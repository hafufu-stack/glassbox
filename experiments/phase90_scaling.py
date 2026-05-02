"""
Phase 90: Test-Time Compute Scaling Law
=========================================
With PRM-guided MCTS, scale rollouts exponentially (8,16,32,64,128)
and measure whether accuracy scales log-linearly.

Prove: "more compute at test time = monotonically better"
(which P87 failed to show without PRM).
"""
import os,sys,json,time,copy,random,math
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from collections import deque
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','training')
RESULTS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results')
FIGURES_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'figures')
os.makedirs(RESULTS_DIR,exist_ok=True);os.makedirs(FIGURES_DIR,exist_ok=True)
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=8;N_COLORS=10

def extract_objects(grid):
    arr=np.array(grid);h,w=arr.shape;vis=np.zeros_like(arr,dtype=bool)
    bg=int(np.bincount(arr.flatten()).argmax());objs=[]
    for r in range(h):
        for c in range(w):
            if not vis[r,c] and arr[r,c]!=bg:
                col=int(arr[r,c]);px=[];q=deque([(r,c)]);vis[r,c]=True
                while q:
                    cr,cc=q.popleft();px.append((cr,cc))
                    for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not vis[nr,nc] and arr[nr,nc]==col:vis[nr,nc]=True;q.append((nr,nc))
                rs=[p[0]for p in px];cs=[p[1]for p in px]
                objs.append({'color':col,'area':len(px),'center':(np.mean(rs),np.mean(cs)),'bbox':(min(rs),min(cs),max(rs),max(cs))})
    return objs,bg

def obj_feats(o,h,w):
    f=[0.]*10;f[min(o['color'],9)]=1.;cr,cc=o['center']
    f.extend([cr/max(h,1),cc/max(w,1),o['area']/max(h*w,1)])
    r0,c0,r1,c1=o['bbox'];f.extend([(r1-r0+1)/max(h,1),(c1-c0+1)/max(w,1),(c1-c0+1)/max(r1-r0+1,1)])
    return f[:NODE_FEAT_DIM]

def extract_op(inp,out):
    ia,oa=np.array(inp),np.array(out);objs,bg=extract_objects(inp)
    if ia.shape==oa.shape and np.array_equal(ia,oa):return 1,0,0,0
    if oa.size>0 and len(np.unique(oa))==1:return 2,0,int(np.unique(oa)[0]),0
    if ia.shape==oa.shape:
        d=ia!=oa
        if d.any():
            oc,nc=set(ia[d].tolist()),set(oa[d].tolist())
            if len(oc)==1 and len(nc)==1:
                o=int(list(oc)[0]);p=0
                for j,obj in enumerate(objs):
                    if obj['color']==o:p=j;break
                return 5,min(p,MAX_OBJECTS-1),o,int(list(nc)[0])
        if np.array_equal(np.flipud(ia),oa):return 6,0,0,0
        if np.array_equal(np.fliplr(ia),oa):return 7,0,0,0
    return 3,0,0,0

def prep(inp,out):
    objs,bg=extract_objects(inp);h,w=len(inp),len(inp[0])
    if not objs:return None
    nf=np.zeros((MAX_OBJECTS,NODE_FEAT_DIM),dtype=np.float32)
    n=min(len(objs),MAX_OBJECTS)
    for j in range(n):f=obj_feats(objs[j],h,w);nf[j,:len(f)]=f[:NODE_FEAT_DIM]
    op,ptr,c1,c2=extract_op(inp,out)
    return{'nf':torch.tensor(nf).unsqueeze(0).float(),'nn':torch.tensor([n]),
           'op':torch.tensor([op]),'c1':torch.tensor([c1]),'c2':torch.tensor([c2]),
           'ptr':torch.tensor([min(ptr,max(n-1,0))])}

def augment_pair(inp,out):
    ia,oa=np.array(inp),np.array(out);aug=[(inp,out)]
    if ia.shape==oa.shape:
        for k in[1,2,3]:aug.append((np.rot90(ia,k).tolist(),np.rot90(oa,k).tolist()))
        aug.append((np.flipud(ia).tolist(),np.flipud(oa).tolist()))
        aug.append((np.fliplr(ia).tolist(),np.fliplr(oa).tolist()))
    for _ in range(2):
        perm=list(range(10));random.shuffle(perm)
        ai=np.vectorize(lambda x:perm[x] if x<10 else x)(ia)
        ao=np.vectorize(lambda x:perm[x] if x<10 else x)(oa)
        aug.append((ai.tolist(),ao.tolist()))
    return aug

class AgentWithValue(nn.Module):
    def __init__(s,hid=64):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid)
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid);s.n2=nn.LayerNorm(hid)
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)
        s.vh=nn.Sequential(nn.Linear(hid+3, hid), nn.ReLU(), nn.Linear(hid, 1), nn.Sigmoid())
    def forward(s,nf,nn_c):
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1);h=s.ne(nf)
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g1(torch.cat([h,msg.expand_as(h)],-1));h=s.n1(h)*mf
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g2(torch.cat([h,msg.expand_as(h)],-1));h=s.n2(h)*mf
        g=(h*mf).sum(1)/mf.sum(1).clamp(min=1)
        pl=((s.pq(g).unsqueeze(1))*s.pk(h)).sum(-1).masked_fill(~mask,-1e9)
        return s.oh(g),s.c1h(g),s.c2h(g),pl,g
    def predict_value(s, g, config_vec):
        return s.vh(torch.cat([g, config_vec], dim=-1))

def compute_loss(model,s):
    ol,cl1,cl2,pl,_=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
    return(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
           F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))

def adapt_model(model,task_samples,steps=10,lr=0.1):
    if not task_samples:return model
    am=copy.deepcopy(model);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    for _ in range(steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
    am.eval();return am

def ablate_l2(model,rate,task_samples):
    am=copy.deepcopy(model);am.train()
    tl=torch.tensor(0.0,device=DEVICE)
    for s in task_samples[:8]:tl=tl+compute_loss(am,s)
    tl=tl/max(len(task_samples[:8]),1);tl.backward()
    l2p=set()
    for p in am.g2.parameters():l2p.add(id(p))
    for p in am.n2.parameters():l2p.add(id(p))
    with torch.no_grad():
        for p in am.parameters():
            if id(p) not in l2p:continue
            if p.grad is not None:
                imp=p.grad.abs();thr=torch.quantile(imp.flatten(),rate)
                p.mul_((imp>thr).float())
    am.eval();return am

def eval_task(model,task):
    ok,total=0,0
    for tp in task.get('test',[]):
        s=prep(tp['input'],tp['output'])
        if s is None:continue
        total+=1
        with torch.no_grad():
            ol,cl1,cl2,pl,_=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
        if(ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
           cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok,total

def demo_loss(model, task_samples):
    if not task_samples: return 999.0
    total = 0.0
    with torch.no_grad():
        for s in task_samples[:8]:
            total += compute_loss(model, s).item()
    return total / min(len(task_samples), 8)

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8') as fp:
                t.append({'id':f[:-5],**json.load(fp)})
    return t

ACTIONS = [
    (0.0,  0.1,  10), (0.0,  0.05, 10), (0.0,  0.15, 10),
    (0.10, 0.1,  10), (0.15, 0.1,  10), (0.20, 0.1,  10),
    (0.15, 0.05, 10), (0.15, 0.15, 10), (0.20, 0.05, 10),
    (0.10, 0.1,  25), (0.15, 0.1,  25), (0.20, 0.1,  25),
]

def prm_mcts(model, task_samples, graph_emb, n_rollouts=16):
    visits = [0]*len(ACTIONS); values = [0.0]*len(ACTIONS); total = 1
    priors = []
    for a_rate, lr, steps in ACTIONS:
        cfg = torch.tensor([[a_rate, lr, steps/25.0]], device=DEVICE)
        with torch.no_grad():
            priors.append(model.predict_value(graph_emb, cfg).item())

    best_loss = float('inf'); best_model = model
    for _ in range(n_rollouts):
        ucbs = []
        for i in range(len(ACTIONS)):
            if visits[i]==0: ucbs.append(float('inf'))
            else: ucbs.append(values[i]/visits[i] + 1.41*math.sqrt(math.log(total)/visits[i]) + 0.5*priors[i])
        idx = max(range(len(ACTIONS)), key=lambda i: ucbs[i])
        a_rate, lr, steps = ACTIONS[idx]
        m = copy.deepcopy(model)
        if a_rate > 0: m = ablate_l2(m, a_rate, task_samples)
        adapted = adapt_model(m, task_samples, steps=steps, lr=lr)
        loss = demo_loss(adapted, task_samples)
        visits[idx] += 1; values[idx] += -loss; total += 1
        if loss < best_loss: best_loss = loss; best_model = adapted
    return best_model, best_loss


def main():
    print("=" * 60)
    print("Phase 90: Test-Time Compute Scaling Law")
    print("Scale rollouts: 8, 16, 32, 64, 128")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    train_tasks = tasks[:split]; test_tasks = tasks[split:]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    all_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep(p['input'], p['output'])
            if s: all_samples.append(s)
    model = AgentWithValue().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(80):
        model.train(); random.shuffle(all_samples)
        for s in all_samples:
            loss = compute_loss(model, s); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    print("  Base model trained.")

    # Reptile meta-init
    print("  Reptile meta-learning...")
    for outer in range(200):
        task = random.choice(train_tasks)
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_s = [s for s in aug_s if s]
        if not aug_s: continue
        adapted = adapt_model(model, aug_s, steps=10, lr=0.1)
        meta_lr = 0.1 * (0.7 ** (outer // 50))
        with torch.no_grad():
            for p_m, p_a in zip(model.parameters(), adapted.parameters()):
                p_m.data += meta_lr * (p_a.data - p_m.data)
    model.eval()
    print("    Reptile done.")

    # PRM data collection + training
    print("  Collecting PRM data...")
    prm_data = []
    for task in train_tasks[:40]:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_s = [prep(ai, ao) for ai, ao in aug_pairs]; aug_s = [s for s in aug_s if s]
        if not aug_s: continue
        with torch.no_grad():
            _,_,_,_,g = model(aug_s[0]['nf'].to(DEVICE), aug_s[0]['nn'].to(DEVICE))
        for _ in range(6):
            a_rate, lr, steps = random.choice(ACTIONS)
            cfg = torch.tensor([[a_rate, lr, steps/25.0]], device=DEVICE)
            m = copy.deepcopy(model)
            if a_rate > 0: m = ablate_l2(m, a_rate, aug_s)
            ad = adapt_model(m, aug_s, steps=steps, lr=lr)
            ok, tot = eval_task(ad, task)
            success = 1.0 if ok == tot and tot > 0 else 0.0
            prm_data.append((g.detach(), cfg.detach(), success))
    print(f"  {len(prm_data)} PRM samples collected")

    for p in model.parameters(): p.requires_grad = False
    for p in model.vh.parameters(): p.requires_grad = True
    vopt = torch.optim.Adam(model.vh.parameters(), lr=1e-3)
    for ep in range(50):
        random.shuffle(prm_data)
        for g, cfg, s in prm_data:
            pred = model.predict_value(g, cfg)
            loss = F.binary_cross_entropy(pred, torch.tensor([[s]], device=DEVICE))
            vopt.zero_grad(); loss.backward(); vopt.step()
    for p in model.parameters(): p.requires_grad = True
    model.eval()
    print("  PRM trained.")

    # Scaling experiment
    rollout_counts = [8, 16, 32, 64, 128]
    scaling_results = {}

    for n_roll in rollout_counts:
        print(f"\n  === Rollouts: {n_roll} ===")
        t1 = time.time()
        ok_total, n_total = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]; aug_s = [s for s in aug_s if s]
            if not aug_s: continue
            with torch.no_grad():
                _,_,_,_,g = model(aug_s[0]['nf'].to(DEVICE), aug_s[0]['nn'].to(DEVICE))
            adapted, _ = prm_mcts(model, aug_s, g, n_rollouts=n_roll)
            ok, tot = eval_task(adapted, task); ok_total += ok; n_total += tot
        acc = ok_total / max(n_total, 1)
        elapsed = time.time() - t1
        scaling_results[n_roll] = {'acc': acc, 'time': elapsed}
        print(f"    PRM-MCTS-{n_roll}: {acc:.1%} ({elapsed:.1f}s)")

    results = {
        'scaling': {str(k): v for k, v in scaling_results.items()},
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase90_scaling.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rolls = sorted(rollout_counts)
    accs = [scaling_results[r]['acc'] for r in rolls]
    times = [scaling_results[r]['time'] for r in rolls]

    ax = axes[0]
    ax.plot(rolls, accs, 's-', lw=3, ms=10, color='#E91E63')
    ax.set_xscale('log', base=2); ax.set_xlabel('Rollouts (log scale)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xticks(rolls); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3); ax.set_ylim(0.7, 1.0)
    ax.set_title('Scaling Law: Accuracy vs Rollouts')

    ax = axes[1]
    ax.plot(times, accs, 'o-', lw=3, ms=10, color='#4CAF50')
    ax.set_xlabel('Compute Time (s)', fontsize=12); ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid(True, alpha=0.3); ax.set_ylim(0.7, 1.0)
    ax.set_title('Compute-Accuracy Tradeoff')

    plt.suptitle('Phase 90: Test-Time Compute Scaling Law', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase90_scaling.png'), dpi=150); plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
