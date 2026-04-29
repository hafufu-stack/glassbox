"""
Phase 38: Demo-Verified Auto-Triage
=======================================
Phase 37 failed because the model's self-diagnosis (entropy) was
overconfident — 99% of tasks classified as "low confusion" despite
many being wrong. The fix: use ACTUAL demo execution results
instead of subjective confidence.

Pipeline:
  1. Run base model on demos → check if predictions match ground truth
  2. If all demos correct → use base model (no adaptation needed)
  3. If some wrong → full adapt 50 steps → re-check
  4. If still wrong → super-compensation (ablate 15% + adapt 50)

This is "Execution-Based Verification" — trust results, not confidence.
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


def verify_on_demos(model, demo_samples):
    """Execute model on demo samples and check actual correctness.
    Returns (n_correct, n_total) — the OBJECTIVE truth."""
    model.eval()
    ok, total = 0, 0
    for s in demo_samples:
        if s is None: continue
        total += 1
        with torch.no_grad():
            ol, cl1, cl2, pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        if (ol.argmax(1).item() == s['op'].item() and
            cl1.argmax(1).item() == s['c1'].item() and
            cl2.argmax(1).item() == s['c2'].item() and
            pl.argmax(1).item() == s['ptr'].item()):
            ok += 1
    return ok, total


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


def demo_verified_triage(model, task, aug_samples, demo_samples):
    """The core Phase 38 innovation: triage based on actual demo execution.
    Returns (adapted_model, triage_level)."""

    # Step 1: Check base model on demos (0 steps)
    ok, total = verify_on_demos(model, demo_samples)
    if total > 0 and ok == total:
        # Base model already solves all demos -> no adaptation needed!
        return model, 'skip'

    # Step 2: Full adaptation (50 steps), then re-check
    adapted = adapt_model(model, aug_samples, steps=50)
    ok2, total2 = verify_on_demos(adapted, demo_samples)
    if total2 > 0 and ok2 == total2:
        # Adapted model solves all demos -> use it
        return adapted, 'adapt'

    # Step 3: Super-compensation (ablate 15% + adapt 50)
    ablated = ablate_least_important(model, 0.15, aug_samples)
    super_adapted = adapt_model(ablated, aug_samples, steps=50)
    ok3, total3 = verify_on_demos(super_adapted, demo_samples)

    # Pick the best: super-comp vs normal adapt
    if total3 > 0 and ok3 > ok2:
        return super_adapted, 'super'
    elif total2 > 0 and ok2 >= ok3:
        return adapted, 'adapt_fallback'
    else:
        return super_adapted, 'super_fallback'


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
    print("Phase 38: Demo-Verified Auto-Triage")
    print("Trust execution results, not model confidence!")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Train base model
    all_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep(p['input'], p['output'])
            if s: all_samples.append(s)
    print(f"Total training samples: {len(all_samples)}")

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

    # === Compare methods ===
    methods_results = {}
    triage_counts = {'skip': 0, 'adapt': 0, 'super': 0, 'adapt_fallback': 0, 'super_fallback': 0}

    # Method 1: Adapt-only baseline (P28)
    ok_base, n_base = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        adapted = adapt_model(model, aug_samples, steps=50)
        ok, tot = eval_task(adapted, task)
        ok_base += ok; n_base += tot
    base_rate = ok_base / max(n_base, 1)
    methods_results['adapt_only'] = base_rate
    print(f"\n  Adapt-only (P28):          {base_rate:.1%}")

    # Method 2: Fixed ablate 15% + adapt (P34 best)
    ok_abl, n_abl = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        ablated = ablate_least_important(model, 0.15, aug_samples)
        adapted = adapt_model(ablated, aug_samples, steps=50)
        ok, tot = eval_task(adapted, task)
        ok_abl += ok; n_abl += tot
    abl_rate = ok_abl / max(n_abl, 1)
    methods_results['fixed_ablate'] = abl_rate
    print(f"  Fixed ablate 15% (P34):    {abl_rate:.1%}")

    # Method 3: Demo-Verified Triage (P38, the new method)
    ok_triage, n_triage = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        demo_samples = [prep(p['input'], p['output']) for p in demos]
        demo_samples = [s for s in demo_samples if s]
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]

        best_model, level = demo_verified_triage(model, task, aug_samples, demo_samples)
        triage_counts[level] += 1

        ok, tot = eval_task(best_model, task)
        ok_triage += ok; n_triage += tot

    triage_rate = ok_triage / max(n_triage, 1)
    methods_results['demo_triage'] = triage_rate
    print(f"  Demo-Verified Triage (P38): {triage_rate:.1%}")

    # Print triage breakdown
    total_tasks = sum(triage_counts.values())
    print(f"\n  --- Triage Breakdown ---")
    for level, count in sorted(triage_counts.items(), key=lambda x: -x[1]):
        print(f"    {level:<20}: {count:>3d} ({count/max(total_tasks,1):.0%})")

    print(f"\n  --- Summary ---")
    print(f"    Adapt-only:       {base_rate:.1%}")
    print(f"    Fixed ablate 15%: {abl_rate:.1%} (delta: {abl_rate-base_rate:+.1%})")
    print(f"    Demo triage:      {triage_rate:.1%} (delta: {triage_rate-base_rate:+.1%})")

    elapsed = time.time() - t0
    results = {
        'adapt_only': base_rate,
        'fixed_ablate_15': abl_rate,
        'demo_triage': triage_rate,
        'triage_counts': triage_counts,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase38_triage.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    names = ['Adapt-Only\n(P28)', 'Fixed Ablate\n15% (P34)', 'Demo Triage\n(P38)']
    vals = [base_rate, abl_rate, triage_rate]
    colors_bar = ['#FF9800', '#2196F3', '#4CAF50']
    bars = ax.bar(range(3), vals, color=colors_bar, alpha=0.85)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Full Match'); ax.set_title('Demo-Verified Triage vs Baselines')
    ax.set_ylim(0, 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.1%}', ha='center', fontsize=12, fontweight='bold')

    ax = axes[1]
    levels = list(triage_counts.keys())
    counts = [triage_counts[l] for l in levels]
    colors_triage = ['#4CAF50', '#2196F3', '#F44336', '#FF9800', '#9C27B0']
    ax.bar(range(len(levels)), counts, color=colors_triage[:len(levels)], alpha=0.85)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Number of Tasks'); ax.set_title('Triage Decision Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase38_triage.png'), dpi=150)
    plt.close()
    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__': main()
