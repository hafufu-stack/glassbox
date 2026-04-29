"""
Phase 53: The Singularity Agent
=================================
The FINAL boss weapon. All 52 phases distilled into one script.

Architecture:
  - 77K GNN Agent (same as always)
  - Pre-trained on ARC training set
  - Test-time: Neuro-Apoptosis + Max-budget Adaptation

No triage. No ensemble. No branching. No complexity.
Just: perturb → adapt → answer. For EVERY task, identically.

This is the Kaggle submission script template.
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

def adapt_model(model, task_samples, steps=50, lr=1e-2):
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

def ablate_least_important(model, rate, task_samples):
    am = copy.deepcopy(model); am.train()
    total_loss = torch.tensor(0.0, device=DEVICE)
    for s in task_samples[:8]:
        total_loss = total_loss + compute_loss(am, s)
    total_loss = total_loss / max(len(task_samples[:8]), 1)
    total_loss.backward()
    with torch.no_grad():
        for p in am.parameters():
            if p.grad is not None:
                importance = p.grad.abs()
                threshold = torch.quantile(importance.flatten(), rate)
                p.mul_((importance > threshold).float())
            else:
                p.mul_((torch.rand_like(p) > rate).float())
    am.eval(); return am

def neurogenesis(model, rate=0.15):
    am = copy.deepcopy(model)
    with torch.no_grad():
        for p in am.parameters():
            mask = torch.rand_like(p) < rate
            noise = torch.randn_like(p) * 0.01
            p.add_(mask.float() * noise)
    return am

def singularity_solve(model, task, adapt_steps=500):
    """THE SINGULARITY FORMULA:
    1. Neuro-Apoptosis (ablate 14% + neurogenesis 15%)
    2. Max-budget adaptation
    3. Answer
    """
    demos = task.get('train', [])
    aug_pairs = []
    for p in demos:
        aug_pairs.extend(augment_pair(p['input'], p['output']))
    aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
    aug_samples = [s for s in aug_samples if s]
    if not aug_samples:
        return model

    # Step 1: Neuro-Apoptosis
    perturbed = ablate_least_important(model, 0.14, aug_samples)
    perturbed = neurogenesis(perturbed, 0.15)

    # Step 2: Max-budget adaptation
    adapted = adapt_model(perturbed, aug_samples, steps=adapt_steps)
    return adapted

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

def load_arc_tasks(d, n=400):
    t = []
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                t.append({'id': f[:-5], **json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 53: The Singularity Agent")
    print("The final weapon. 53 phases distilled into one formula.")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)

    all_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep(p['input'], p['output'])
            if s: all_samples.append(s)

    model = Agent().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(80):
        model.train(); random.shuffle(all_samples); el, nb = 0, 0
        for s in all_samples:
            loss = compute_loss(model, s)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}/80: loss={el/nb:.4f}")

    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    # The ultimate showdown
    step_budgets = [50, 100, 200, 500]
    results = {}

    # Baseline
    ok_total, n_total = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        current = adapt_model(model, aug_samples, steps=50)
        ok, tot = eval_task(current, task)
        ok_total += ok; n_total += tot
    results['baseline'] = ok_total / max(n_total, 1)
    print(f"\n  Baseline (Adapt 50):         {results['baseline']:.1%}")

    # The Singularity Agent at various budgets
    print(f"\n  --- THE SINGULARITY AGENT ---")
    for budget in step_budgets:
        ok_total, n_total = 0, 0
        t_start = time.time()
        for task in test_tasks:
            adapted = singularity_solve(model, task, adapt_steps=budget)
            ok, tot = eval_task(adapted, task)
            ok_total += ok; n_total += tot
        rate = ok_total / max(n_total, 1)
        t_total = time.time() - t_start
        t_per = t_total / max(len(test_tasks), 1)
        results[f'singularity_{budget}'] = rate
        results[f'singularity_{budget}_time'] = t_per
        delta = rate - results['baseline']
        # Calculate how many tasks can be solved in 12h at this rate
        tasks_per_12h = int(12 * 3600 / max(t_per, 0.001))
        print(f"  {budget:>5} steps: {rate:.1%} (delta: {delta:>+5.1%}, {t_per:.2f}s/task, max {tasks_per_12h} tasks/12h)")

    # Find best budget
    best_key = max([f'singularity_{b}' for b in step_budgets], key=lambda k: results[k])
    best_budget = int(best_key.split('_')[1])
    best_rate = results[best_key]

    print(f"\n  *** BEST: Singularity Agent @ {best_budget} steps = {best_rate:.1%} ***")
    print(f"  *** Formula: Ablate 14% + Neurogenesis 15% + Adapt {best_budget} ***")

    elapsed = time.time() - t0
    results['best_budget'] = best_budget
    results['best_rate'] = best_rate
    results['elapsed'] = elapsed
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase53_singularity.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Grand visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    budgets = step_budgets
    vals = [results[f'singularity_{b}'] for b in budgets]
    ax.plot(budgets, vals, 'o-', color='#673AB7', linewidth=3, markersize=10, label='Singularity Agent')
    ax.axhline(y=results['baseline'], color='#FF9800', linestyle='--', linewidth=2, label=f'Baseline ({results["baseline"]:.1%})')
    ax.set_xlabel('Adaptation Steps'); ax.set_ylabel('Full Match')
    ax.set_title('The Singularity Agent: Performance Scaling')
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
    for b, v in zip(budgets, vals):
        ax.annotate(f'{v:.1%}', xy=(b, v), xytext=(0, 15), textcoords='offset points',
                   ha='center', fontsize=12, fontweight='bold', color='#673AB7')

    ax = axes[1]
    all_names = ['Baseline\n(Adapt 50)'] + [f'Singularity\n{b} steps' for b in budgets]
    all_vals = [results['baseline']] + vals
    colors = ['#9E9E9E'] + ['#673AB7'] * len(budgets)
    bars = ax.bar(range(len(all_names)), all_vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(all_names))); ax.set_xticklabels(all_names, fontsize=9)
    ax.set_ylabel('Full Match'); ax.set_title('Phase 53: Final Comparison')
    ax.set_ylim(0, 1)
    for b, v in zip(bars, all_vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('Phase 53: The Singularity Agent — The Final Weapon', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase53_singularity.png'), dpi=150)
    plt.close()
    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__': main()
