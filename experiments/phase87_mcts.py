"""
Phase 87: Meta-MCTS (Monte Carlo Tree Search with Meta-Init)
==============================================================
Use P83's 9.6x speedup to fuel tree search at test time.
The GlassBox agent serves as the policy network;
MCTS explores multiple ablation+adaptation paths in parallel.

Compare: single-shot TTT vs MCTS with N rollouts.
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

def demo_loss(model, task_samples):
    """Average loss on demo samples (lower = better fit)."""
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

def reptile_meta_train(model, train_tasks, n_outer=200, n_inner=10, meta_lr=0.1, inner_lr=0.1):
    print("  Reptile meta-learning...")
    for outer in range(n_outer):
        task = random.choice(train_tasks)
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_s = [s for s in aug_s if s]
        if not aug_s: continue
        adapted = adapt_model(model, aug_s, steps=n_inner, lr=inner_lr)
        with torch.no_grad():
            for p_m, p_a in zip(model.parameters(), adapted.parameters()):
                p_m.data += meta_lr * (p_a.data - p_m.data)
        if outer > 0 and outer % 50 == 0: meta_lr *= 0.7
    print(f"    Reptile done ({n_outer} outer steps)")
    return model


# ==============================
# MCTS for Test-Time Adaptation
# ==============================

class MCTSNode:
    def __init__(self, model, ablation_rate, lr, steps, parent=None):
        self.model = model
        self.ablation_rate = ablation_rate
        self.lr = lr
        self.steps = steps
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0  # negative loss (higher = better)
        self.adapted_model = None

    def ucb1(self, c=1.41):
        if self.visits == 0: return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return self.value / self.visits + c * math.sqrt(math.log(parent_visits) / self.visits)

def mcts_adapt(model, task_samples, n_rollouts=16, meta_init=True):
    """MCTS over adaptation hyperparameters."""
    # Define action space: (ablation_rate, lr, steps)
    actions = [
        (0.0,  0.1,  10), (0.0,  0.05, 10), (0.0,  0.15, 10),
        (0.10, 0.1,  10), (0.15, 0.1,  10), (0.20, 0.1,  10),
        (0.15, 0.05, 10), (0.15, 0.15, 10), (0.20, 0.05, 10),
        (0.10, 0.1,  25), (0.15, 0.1,  25), (0.20, 0.1,  25),
    ]

    root = MCTSNode(model, 0.0, 0.1, 10)
    root.visits = 1

    # Create child nodes
    for a_rate, lr, steps in actions:
        child = MCTSNode(model, a_rate, lr, steps, parent=root)
        root.children.append(child)

    best_loss = float('inf')
    best_model = model

    for rollout in range(n_rollouts):
        # Selection: pick best child by UCB1
        node = max(root.children, key=lambda n: n.ucb1())

        # Expansion + Simulation: adapt and evaluate
        m = copy.deepcopy(node.model)
        if node.ablation_rate > 0:
            m = ablate_l2(m, node.ablation_rate, task_samples)
        adapted = adapt_model(m, task_samples, steps=node.steps, lr=node.lr)
        loss = demo_loss(adapted, task_samples)

        # Backpropagation
        reward = -loss  # Higher = better
        node.visits += 1
        node.value += reward
        root.visits += 1

        if loss < best_loss:
            best_loss = loss
            best_model = adapted
            node.adapted_model = adapted

    return best_model, best_loss


def main():
    print("=" * 60)
    print("Phase 87: Meta-MCTS")
    print("Monte Carlo Tree Search with Meta-Init for TTT")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    train_tasks = tasks[:split]; test_tasks = tasks[split:]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Train base model
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

    # Meta-init (P83 Reptile)
    model_meta = copy.deepcopy(model)
    model_meta = reptile_meta_train(model_meta, train_tasks, n_outer=200, n_inner=10)
    model_meta.eval()

    # Evaluate: Standard TTT vs MCTS with varying rollouts
    configs = {
        'Standard TTT-100': {'model': model, 'mcts': False, 'steps': 100},
        'Meta TTT-10': {'model': model_meta, 'mcts': False, 'steps': 10},
        'Meta MCTS-8': {'model': model_meta, 'mcts': True, 'rollouts': 8},
        'Meta MCTS-16': {'model': model_meta, 'mcts': True, 'rollouts': 16},
        'Meta MCTS-32': {'model': model_meta, 'mcts': True, 'rollouts': 32},
    }

    eval_results = {}
    for name, cfg in configs.items():
        print(f"\n  Evaluating: {name}")
        t1 = time.time()
        ok_total, n_total = 0, 0

        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_s = [s for s in aug_s if s]

            if cfg['mcts']:
                adapted, _ = mcts_adapt(cfg['model'], aug_s, n_rollouts=cfg['rollouts'])
            else:
                ablated = ablate_l2(cfg['model'], 0.15, aug_s)
                adapted = adapt_model(ablated, aug_s, steps=cfg['steps'], lr=0.1)

            ok, tot = eval_task(adapted, task)
            ok_total += ok; n_total += tot

        acc = ok_total / max(n_total, 1)
        elapsed = time.time() - t1
        eval_results[name] = {'acc': acc, 'time': elapsed}
        print(f"    {name}: {acc:.1%} ({elapsed:.1f}s)")

    results = {
        'configs': eval_results,
        'best': max(eval_results.items(), key=lambda x: x[1]['acc'])[0],
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase87_mcts.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = list(eval_results.keys())
    accs = [eval_results[n]['acc'] for n in names]
    times = [eval_results[n]['time'] for n in names]
    colors = ['#9E9E9E'] + ['#1565C0'] + ['#4CAF50', '#E91E63', '#FF8F00']
    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85, edgecolor='white', lw=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy'); ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    for b, a, t in zip(bars, accs, times):
        ax.text(b.get_x()+b.get_width()/2, a+0.01, f'{a:.1%}\n({t:.0f}s)',
                ha='center', fontsize=9, fontweight='bold')
    plt.title('Phase 87: Meta-MCTS - Accuracy vs Compute', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase87_mcts.png'), dpi=150); plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
