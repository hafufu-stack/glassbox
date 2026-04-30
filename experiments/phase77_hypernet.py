"""
Phase 77: Amortized TTT via Hypernetworks
==========================================
Replace 100-step gradient TTT with a single forward pass.
Train a hypernetwork that predicts optimal weight deltas (DeltaW)
for GNN L2 given demo task features.

Compare: full TTT (100 steps) vs hypernetwork (1 forward pass).
1 seed (architectural exploration).
"""
import os,sys,json,time,copy,random
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

def adapt_model(model,task_samples,steps=100,lr=0.1):
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

def get_task_features(task_samples, model):
    """Extract a fixed-size feature vector from task demos."""
    if not task_samples: return torch.zeros(64, device=DEVICE)
    feats = []
    with torch.no_grad():
        for s in task_samples[:5]:
            nf = s['nf'].to(DEVICE); nn_c = s['nn'].to(DEVICE)
            mask = torch.arange(MAX_OBJECTS, device=DEVICE).unsqueeze(0) < nn_c.unsqueeze(1)
            mf = mask.float().unsqueeze(-1)
            h = model.ne(nf)
            msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
            h = h + model.g1(torch.cat([h, msg.expand_as(h)], -1))
            h = model.n1(h) * mf
            g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)
            feats.append(g.squeeze(0))
    return torch.stack(feats).mean(0)  # (64,)

class HyperNetwork(nn.Module):
    """Predicts weight deltas for L2 given task features."""
    def __init__(self, feat_dim=64, hid=128):
        super().__init__()
        # Count L2 parameters: g2 has Linear(128,64)+ReLU+Linear(64,64) + n2 LayerNorm(64)
        # g2: 128*64+64 + 64*64+64 = 8192+64+4096+64 = 12416
        # n2: 64+64 = 128
        # Total: 12544
        self.n_l2_params = 12544
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, self.n_l2_params)
        )
        # Initialize last layer to near-zero (start with no modification)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, task_feats):
        return self.net(task_feats)

def apply_delta_w(model, delta_w):
    """Apply predicted weight deltas to L2 parameters."""
    am = copy.deepcopy(model)
    offset = 0
    with torch.no_grad():
        for p in list(am.g2.parameters()) + list(am.n2.parameters()):
            n = p.numel()
            if offset + n <= len(delta_w):
                p.add_(delta_w[offset:offset+n].reshape(p.shape) * 0.1)  # Scale down
            offset += n
    am.eval()
    return am

def eval_task(model, task):
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


def main():
    print("=" * 60)
    print("Phase 77: Amortized TTT via Hypernetworks")
    print("Replace 100-step TTT with 1 forward pass")
    print("=" * 60)
    t0=time.time()
    tasks=load_arc_tasks(DATA_DIR)
    split=int(len(tasks)*0.8)
    train_tasks=tasks[:split]; test_tasks=tasks[split:]

    random.seed(42);np.random.seed(42);torch.manual_seed(42)

    # Train base model
    all_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:all_samples.append(s)
    model=Agent().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(80):
        model.train();random.shuffle(all_samples)
        for s in all_samples:
            loss=compute_loss(model,s);opt.zero_grad();loss.backward();opt.step()
    model.eval()
    print("  Base model trained.")

    # Verify L2 param count
    n_l2 = sum(p.numel() for p in model.g2.parameters()) + sum(p.numel() for p in model.n2.parameters())
    print(f"  L2 params: {n_l2}")

    # Train hypernetwork on train tasks
    hypernet = HyperNetwork(feat_dim=64).to(DEVICE)
    hn_opt = torch.optim.Adam(hypernet.parameters(), lr=1e-3)

    print("  Training hypernetwork...")
    for ep in range(50):
        hypernet.train(); random.shuffle(train_tasks)
        ep_loss = 0; n_tasks = 0
        for task in train_tasks[:100]:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_samples = [s for s in aug_samples if s]
            if not aug_samples: continue

            task_feats = get_task_features(aug_samples, model)
            delta_w = hypernet(task_feats)
            adapted = apply_delta_w(model, delta_w)

            # Loss on task demos
            task_loss = sum(compute_loss(adapted, s) for s in aug_samples[:4]) / min(4, len(aug_samples))
            hn_opt.zero_grad(); task_loss.backward(); hn_opt.step()
            ep_loss += task_loss.item(); n_tasks += 1

        if ep % 10 == 0:
            print(f"    Epoch {ep}: loss={ep_loss/max(n_tasks,1):.4f}")
    hypernet.eval()

    # Evaluate: no adapt vs TTT vs hypernetwork
    no_adapt_ok, ttt_ok, hyper_ok, total = 0, 0, 0, 0
    ttt_time, hyper_time = 0, 0

    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]

        # No adaptation
        ok0, tot = eval_task(model, task)
        no_adapt_ok += ok0; total += tot

        # Full TTT (100 steps)
        t1 = time.time()
        ablated = ablate_l2(model, 0.20, aug_samples)
        ttt_adapted = adapt_model(ablated, aug_samples, steps=100, lr=0.1)
        ttt_time += time.time() - t1
        ok1, _ = eval_task(ttt_adapted, task)
        ttt_ok += ok1

        # Hypernetwork (1 forward pass)
        t2 = time.time()
        with torch.no_grad():
            task_feats = get_task_features(aug_samples, model)
            delta_w = hypernet(task_feats)
        hyper_adapted = apply_delta_w(model, delta_w)
        hyper_time += time.time() - t2
        ok2, _ = eval_task(hyper_adapted, task)
        hyper_ok += ok2

    no_adapt_acc = no_adapt_ok / max(total, 1)
    ttt_acc = ttt_ok / max(total, 1)
    hyper_acc = hyper_ok / max(total, 1)

    print(f"\n  No adaptation: {no_adapt_acc:.1%}")
    print(f"  TTT (100 steps): {ttt_acc:.1%} ({ttt_time:.1f}s total)")
    print(f"  Hypernetwork:    {hyper_acc:.1%} ({hyper_time:.1f}s total)")
    print(f"  Speedup: {ttt_time/max(hyper_time,0.001):.1f}x")

    results = {
        'no_adapt_acc': no_adapt_acc, 'ttt_acc': ttt_acc, 'hyper_acc': hyper_acc,
        'ttt_time': ttt_time, 'hyper_time': hyper_time,
        'speedup': ttt_time / max(hyper_time, 0.001),
        'elapsed': time.time()-t0, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    with open(os.path.join(RESULTS_DIR, 'phase77_hypernet.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    bars = ax.bar(['No Adapt', 'TTT\n(100 steps)', 'Hypernetwork\n(1 pass)'],
                  [no_adapt_acc, ttt_acc, hyper_acc], color=['#9E9E9E', '#4CAF50', '#E91E63'], alpha=0.85)
    ax.set_ylabel('Accuracy'); ax.set_title('Accuracy Comparison')
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, a in zip(bars, [no_adapt_acc, ttt_acc, hyper_acc]):
        ax.text(b.get_x()+b.get_width()/2, a+0.02, f'{a:.1%}', ha='center', fontsize=10, fontweight='bold')

    ax = axes[1]
    bars = ax.bar(['TTT', 'Hypernetwork'], [ttt_time, hyper_time], color=['#4CAF50', '#E91E63'], alpha=0.85)
    ax.set_ylabel('Time (seconds)'); ax.set_title('Inference Time')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 77: Amortized TTT', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase77_hypernet.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
