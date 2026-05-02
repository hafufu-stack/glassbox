"""
Phase 91: Expert Iteration (MCTS -> Policy Distillation)
=========================================================
Use MCTS-8's "deep thinking" to teach the base model's
"instant intuition" (Zero-Shot).

AlphaZero loop: MCTS finds good paths -> distill into policy.
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

class Agent(nn.Module):
    def __init__(s,hid=64):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid)
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid);s.n2=nn.LayerNorm(hid)
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)
    def forward(s,nf,nn_c):
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1);h=s.ne(nf)
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g1(torch.cat([h,msg.expand_as(h)],-1));h=s.n1(h)*mf
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g2(torch.cat([h,msg.expand_as(h)],-1));h=s.n2(h)*mf
        g=(h*mf).sum(1)/mf.sum(1).clamp(min=1)
        pl=((s.pq(g).unsqueeze(1))*s.pk(h)).sum(-1).masked_fill(~mask,-1e9)
        return s.oh(g),s.c1h(g),s.c2h(g),pl

def compute_loss(model,s):
    ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
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
            ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
        if(ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
           cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok,total

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

def demo_loss(model, task_samples):
    if not task_samples: return 999.0
    total = 0.0
    with torch.no_grad():
        for s in task_samples[:8]:
            total += compute_loss(model, s).item()
    return total / min(len(task_samples), 8)

def mcts_find_best(model, task_samples, n_rollouts=8):
    """Run MCTS and return best adapted model + the winning action."""
    visits = [0]*len(ACTIONS); values = [0.0]*len(ACTIONS); total = 1
    best_loss = float('inf'); best_model = model; best_action = 0

    for _ in range(n_rollouts):
        ucbs = []
        for i in range(len(ACTIONS)):
            if visits[i]==0: ucbs.append(float('inf'))
            else: ucbs.append(values[i]/visits[i] + 1.41*math.sqrt(math.log(total)/visits[i]))
        idx = max(range(len(ACTIONS)), key=lambda i: ucbs[i])
        a_rate, lr, steps = ACTIONS[idx]
        m = copy.deepcopy(model)
        if a_rate > 0: m = ablate_l2(m, a_rate, task_samples)
        adapted = adapt_model(m, task_samples, steps=steps, lr=lr)
        loss = demo_loss(adapted, task_samples)
        visits[idx] += 1; values[idx] += -loss; total += 1
        if loss < best_loss: best_loss = loss; best_model = adapted; best_action = idx
    return best_model, best_action


def main():
    print("=" * 60)
    print("Phase 91: Expert Iteration (MCTS -> Policy Distillation)")
    print("Distill MCTS's deep thinking into zero-shot intuition")
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
    model = Agent().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(80):
        model.train(); random.shuffle(all_samples)
        for s in all_samples:
            loss = compute_loss(model, s); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    print("  Base model trained.")

    # Baseline zero-shot
    zs_ok, zs_tot = 0, 0
    for task in test_tasks:
        ok, tot = eval_task(model, task); zs_ok += ok; zs_tot += tot
    zs_base = zs_ok / max(zs_tot, 1)
    print(f"  ZS baseline: {zs_base:.1%}")

    # Baseline TTT
    ttt_ok, ttt_tot = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_s = [prep(ai, ao) for ai, ao in aug_pairs]; aug_s = [s for s in aug_s if s]
        ablated = ablate_l2(model, 0.15, aug_s)
        adapted = adapt_model(ablated, aug_s, steps=100, lr=0.1)
        ok, tot = eval_task(adapted, task); ttt_ok += ok; ttt_tot += tot
    ttt_base = ttt_ok / max(ttt_tot, 1)
    print(f"  TTT baseline: {ttt_base:.1%}")

    # Expert Iteration loop
    N_ITER = 5
    zs_history = [zs_base]; ttt_history = [ttt_base]

    for iteration in range(N_ITER):
        print(f"\n  === Expert Iteration {iteration+1}/{N_ITER} ===")

        # Step 1: MCTS expert generates training data
        expert_samples = []
        n_success = 0
        for task in train_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]; aug_s = [s for s in aug_s if s]
            if not aug_s: continue

            # MCTS finds best adaptation
            best_adapted, _ = mcts_find_best(model, aug_s, n_rollouts=8)
            ok, tot = eval_task(best_adapted, task)
            if ok == tot and tot > 0:
                n_success += 1
                # Collect test predictions as expert demonstrations
                for tp in task.get('test', []):
                    s = prep(tp['input'], tp['output'])
                    if s: expert_samples.append(s)
                # Also include augmented train pairs
                for s in aug_s[:4]:
                    expert_samples.append(s)

        print(f"    MCTS solved {n_success}/{len(train_tasks)} train tasks")
        print(f"    Collected {len(expert_samples)} expert samples")

        if not expert_samples:
            zs_history.append(zs_history[-1]); ttt_history.append(ttt_history[-1])
            continue

        # Step 2: Distill expert knowledge into base model
        # Mix with original data (70% original, 30% expert)
        n_expert = len(expert_samples)
        n_original = int(n_expert * 0.7 / 0.3)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        for ep in range(20):
            model.train()
            orig_batch = random.sample(all_samples, min(n_original, len(all_samples)))
            mixed = expert_samples + orig_batch; random.shuffle(mixed)
            for s in mixed:
                loss = compute_loss(model, s); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

        # Step 3: Evaluate
        zs_ok, zs_tot = 0, 0
        for task in test_tasks:
            ok, tot = eval_task(model, task); zs_ok += ok; zs_tot += tot
        zs_acc = zs_ok / max(zs_tot, 1)
        zs_history.append(zs_acc)

        ttt_ok, ttt_tot = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]; aug_s = [s for s in aug_s if s]
            ablated = ablate_l2(model, 0.15, aug_s)
            adapted = adapt_model(ablated, aug_s, steps=100, lr=0.1)
            ok, tot = eval_task(adapted, task); ttt_ok += ok; ttt_tot += tot
        ttt_acc = ttt_ok / max(ttt_tot, 1)
        ttt_history.append(ttt_acc)

        print(f"    Iter {iteration+1}: ZS={zs_acc:.1%} ({zs_acc-zs_base:+.1%}), "
              f"TTT={ttt_acc:.1%} ({ttt_acc-ttt_base:+.1%})")

    results = {
        'zs_baseline': zs_base, 'ttt_baseline': ttt_base,
        'zs_history': zs_history, 'ttt_history': ttt_history,
        'final_zs': zs_history[-1], 'final_ttt': ttt_history[-1],
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase91_expert_iter.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    iters = range(len(zs_history))
    ax.plot(iters, zs_history, 's-', lw=3, ms=10, color='#E91E63', label='Zero-Shot')
    ax.plot(iters, ttt_history, 'o-', lw=3, ms=10, color='#4CAF50', label='TTT-100')
    ax.axhline(y=zs_base, color='#E91E63', ls=':', alpha=0.4, label=f'ZS baseline ({zs_base:.1%})')
    ax.axhline(y=ttt_base, color='#4CAF50', ls=':', alpha=0.4, label=f'TTT baseline ({ttt_base:.1%})')
    ax.set_xlabel('Expert Iteration', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    plt.title('Phase 91: Expert Iteration (MCTS -> Distillation)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase91_expert_iter.png'), dpi=150); plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
