"""
Phase 68: Zero-Shot Rule Synthesis
====================================
Can TTT learn rules it has NEVER seen during training?
Hold out specific operations, test if adaptation synthesizes them.

Design: Hold out op=6 (flipV) and op=7 (flipH) from training.
Test if GNN L2 ablation + 100 steps can learn these from scratch.
3 seeds.
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

def adapt_model(model, task_samples, steps=100, lr=1e-2):
    if not task_samples: return model
    am = copy.deepcopy(model)
    opt = torch.optim.SGD(am.parameters(), lr=lr)
    am.train()
    for _ in range(steps):
        batch = random.sample(task_samples, min(8, len(task_samples)))
        tl = sum(compute_loss(am, d) for d in batch) / len(batch)
        opt.zero_grad(); tl.backward()
        torch.nn.utils.clip_grad_norm_(am.parameters(), 1.0); opt.step()
    am.eval(); return am

def ablate_l2(model, rate, task_samples):
    am = copy.deepcopy(model); am.train()
    total_loss = torch.tensor(0.0, device=DEVICE)
    for s in task_samples[:8]:
        total_loss = total_loss + compute_loss(am, s)
    total_loss = total_loss / max(len(task_samples[:8]), 1)
    total_loss.backward()
    l2_params = set()
    for p in am.g2.parameters(): l2_params.add(id(p))
    for p in am.n2.parameters(): l2_params.add(id(p))
    with torch.no_grad():
        for p in am.parameters():
            if id(p) not in l2_params: continue
            if p.grad is not None:
                importance = p.grad.abs()
                threshold = torch.quantile(importance.flatten(), rate)
                p.mul_((importance > threshold).float())
    am.eval(); return am

def eval_task(model, task):
    ok, total = 0, 0
    for tp in task.get('test', []):
        s = prep(tp['input'], tp['output'])
        if s is None: continue
        total += 1
        with torch.no_grad():
            ol, cl1, cl2, pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        if (ol.argmax(1).item() == s['op'].item() and cl1.argmax(1).item() == s['c1'].item() and
            cl2.argmax(1).item() == s['c2'].item() and pl.argmax(1).item() == s['ptr'].item()):
            ok += 1
    return ok, total

def get_task_op(task):
    """Get the dominant operation for a task."""
    ops = []
    for p in task.get('train', []):
        s = prep(p['input'], p['output'])
        if s: ops.append(s['op'].item())
    if ops:
        return max(set(ops), key=ops.count)
    return 3

def load_arc_tasks(d, n=400):
    t = []
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                t.append({'id': f[:-5], **json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 68: Zero-Shot Rule Synthesis")
    print("Can TTT learn rules never seen in training?")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)

    # Classify tasks by operation
    HELD_OUT_OPS = {6, 7}  # flipV, flipH
    OP_NAMES = {1:'identity', 2:'fill', 3:'transform', 5:'recolor', 6:'flipV', 7:'flipH'}

    task_ops = {}
    for task in tasks:
        task_ops[task['id']] = get_task_op(task)

    seen_tasks = [t for t in tasks if task_ops[t['id']] not in HELD_OUT_OPS]
    unseen_tasks = [t for t in tasks if task_ops[t['id']] in HELD_OUT_OPS]

    print(f"  Seen tasks (ops not in {HELD_OUT_OPS}): {len(seen_tasks)}")
    print(f"  Unseen tasks (ops in {HELD_OUT_OPS}): {len(unseen_tasks)}")

    N_SEEDS = 3
    conditions = {
        'seen_no_adapt': [],
        'seen_adapt': [],
        'unseen_no_adapt': [],
        'unseen_adapt': [],
        'unseen_l2_adapt': [],
    }

    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        random.seed(seed * 1000); np.random.seed(seed * 1000); torch.manual_seed(seed * 1000)

        # Train ONLY on seen tasks
        all_samples = []
        for task in seen_tasks:
            for p in task.get('train', []):
                s = prep(p['input'], p['output'])
                if s: all_samples.append(s)

        model = Agent().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(80):
            model.train(); random.shuffle(all_samples)
            for s in all_samples:
                loss = compute_loss(model, s)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        print(f"    Trained on {len(all_samples)} samples (ops {HELD_OUT_OPS} excluded)")

        # Evaluate on seen test tasks
        seen_test = seen_tasks[int(len(seen_tasks)*0.8):]
        for cond_name, test_set, do_adapt, do_l2 in [
            ('seen_no_adapt', seen_test, False, False),
            ('seen_adapt', seen_test, True, False),
            ('unseen_no_adapt', unseen_tasks, False, False),
            ('unseen_adapt', unseen_tasks, True, False),
            ('unseen_l2_adapt', unseen_tasks, True, True),
        ]:
            ok_total, n_total = 0, 0
            for task in test_set:
                demos = task.get('train', [])
                aug_pairs = []
                for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
                aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                aug_samples = [s for s in aug_samples if s]

                current = model
                if do_l2:
                    current = ablate_l2(current, 0.15, aug_samples)
                if do_adapt:
                    current = adapt_model(current, aug_samples, steps=100)
                ok, tot = eval_task(current, task)
                ok_total += ok; n_total += tot

            rate = ok_total / max(n_total, 1)
            conditions[cond_name].append(rate)
            print(f"    {cond_name:<20}: {rate:.1%}")

    # Summary
    results = {}
    print(f"\n  === ZERO-SHOT RULE SYNTHESIS ({N_SEEDS} seeds) ===")
    for cname in conditions:
        vals = np.array(conditions[cname])
        results[f'{cname}_mean'] = float(vals.mean())
        results[f'{cname}_std'] = float(vals.std())
        results[f'{cname}_all'] = [float(v) for v in vals]
        print(f"  {cname:<20}: {vals.mean():.1%} +/- {vals.std():.1%}")

    # Key metric: unseen recovery
    base = results['unseen_no_adapt_mean']
    adapted = results['unseen_adapt_mean']
    l2_adapted = results['unseen_l2_adapt_mean']
    seen_ref = results['seen_adapt_mean']
    print(f"\n  Unseen recovery: {base:.1%} -> {adapted:.1%} -> {l2_adapted:.1%}")
    print(f"  Recovery ratio: {l2_adapted/max(seen_ref,0.01):.1%} of seen performance")

    results['held_out_ops'] = list(HELD_OUT_OPS)
    results['n_seeds'] = N_SEEDS
    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase68_zero_shot.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels = ['Seen\n(no adapt)', 'Seen\n(adapt)', 'Unseen\n(no adapt)', 'Unseen\n(adapt)', 'Unseen\n(L2+adapt)']
    means = [results[f'{c}_mean'] for c in conditions]
    stds = [results[f'{c}_std'] for c in conditions]
    colors = ['#9E9E9E', '#4CAF50', '#FFCDD2', '#FF9800', '#E91E63']
    bars = ax.bar(range(5), means, yerr=stds, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    ax.set_ylabel('Full Match'); ax.set_title('Phase 68: Zero-Shot Rule Synthesis')
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m+0.03, f'{m:.1%}', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase68_zero_shot.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
