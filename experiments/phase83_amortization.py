"""
Phase 83: Intermediate Amortization
======================================
Bridge the gap between P77's failed hypernetwork (62.1%) and
full TTT (88.9%). Use Reptile meta-learning to find optimal
initialization, then run SHORT TTT (10 steps instead of 100).

Compare: full TTT 100 steps vs meta-init + 10 steps vs meta-init + 25 steps.
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


def reptile_meta_train(model, train_tasks, n_outer=200, n_inner=10, meta_lr=0.1, inner_lr=0.1):
    """Reptile meta-learning: find initialization that adapts fast."""
    print("  Reptile meta-learning...")
    for outer in range(n_outer):
        task = random.choice(train_tasks)
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_s = [s for s in aug_s if s]
        if not aug_s: continue

        # Inner loop: adapt copy
        adapted = adapt_model(model, aug_s, steps=n_inner, lr=inner_lr)

        # Outer step: move model toward adapted weights (Reptile update)
        with torch.no_grad():
            for p_model, p_adapted in zip(model.parameters(), adapted.parameters()):
                p_model.data += meta_lr * (p_adapted.data - p_model.data)

        # Decay meta_lr
        if outer > 0 and outer % 50 == 0:
            meta_lr *= 0.7
            if outer % 100 == 0:
                print(f"    Reptile outer step {outer}/{n_outer}")

    print(f"    Reptile done ({n_outer} outer steps)")
    return model


def main():
    print("=" * 60)
    print("Phase 83: Intermediate Amortization")
    print("Reptile meta-init + Short TTT (10/25 steps)")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    train_tasks = tasks[:split]; test_tasks = tasks[split:]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Phase 1: Standard training
    all_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep(p['input'], p['output'])
            if s: all_samples.append(s)

    model_standard = Agent().to(DEVICE)
    opt = torch.optim.Adam(model_standard.parameters(), lr=1e-3)
    for ep in range(80):
        model_standard.train(); random.shuffle(all_samples)
        for s in all_samples:
            loss = compute_loss(model_standard, s)
            opt.zero_grad(); loss.backward(); opt.step()
    model_standard.eval()
    print("  Standard model trained.")

    # Phase 2: Reptile meta-trained model (same initial training, then meta-adapt)
    model_meta = copy.deepcopy(model_standard)
    model_meta = reptile_meta_train(model_meta, train_tasks, n_outer=200, n_inner=10)
    model_meta.eval()

    # Evaluate all configurations
    configs = [
        ('Standard + TTT 100', model_standard, 100),
        ('Standard + TTT 25', model_standard, 25),
        ('Standard + TTT 10', model_standard, 10),
        ('Meta-Init + TTT 25', model_meta, 25),
        ('Meta-Init + TTT 10', model_meta, 10),
        ('Meta-Init + TTT 5', model_meta, 5),
    ]

    eval_results = {}
    for name, model, steps in configs:
        print(f"\n  Evaluating: {name}")
        t1 = time.time()
        ok_total, n_total = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_s = [s for s in aug_s if s]
            adapted = adapt_model(model, aug_s, steps=steps, lr=0.1)
            ok, tot = eval_task(adapted, task)
            ok_total += ok; n_total += tot
        acc = ok_total / max(n_total, 1)
        elapsed = time.time() - t1
        eval_results[name] = {'acc': acc, 'time': elapsed, 'steps': steps}
        print(f"    {name}: {acc:.1%} ({elapsed:.1f}s)")

    # Compute speedups relative to TTT 100
    base_time = eval_results['Standard + TTT 100']['time']
    base_acc = eval_results['Standard + TTT 100']['acc']
    for name, r in eval_results.items():
        r['speedup'] = base_time / max(r['time'], 0.001)
        r['acc_gap'] = r['acc'] - base_acc

    results = {
        'configs': {k: v for k, v in eval_results.items()},
        'best_meta': max(
            [(k,v) for k,v in eval_results.items() if 'Meta' in k],
            key=lambda x: x[1]['acc']
        )[0],
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase83_amortization.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = list(eval_results.keys())
    accs = [eval_results[n]['acc'] for n in names]
    times = [eval_results[n]['time'] for n in names]
    colors = ['#1565C0' if 'Standard' in n else '#E91E63' for n in names]

    ax = axes[0]
    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy'); ax.set_title('Accuracy Comparison')
    ax.set_ylim(0.6, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, a in zip(bars, accs):
        ax.text(b.get_x()+b.get_width()/2, a+0.01, f'{a:.1%}', ha='center', fontsize=8, fontweight='bold')

    ax = axes[1]
    bars = ax.bar(range(len(names)), times, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Time (s)'); ax.set_title('Inference Time')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 83: Intermediate Amortization (Reptile + Short TTT)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase83_amortization.png'), dpi=150); plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
