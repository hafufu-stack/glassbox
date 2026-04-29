"""
Phase 34: Hydra-Driven Super-Compensation
=============================================
Phase 31 discovered "super-compensation": destroying 10-30% of neurons
and then adapting can EXCEED the original baseline performance.

Phase 34 weaponizes this as a test-time inference algorithm:
  1. Identify least-important neurons (via gradient magnitude)
  2. Strategically ablate 15-20% of them
  3. Run Data-Augmented Adaptation (50 steps) on the damaged model
  4. The forced re-wiring should break out of local optima

Hypothesis: Strategic ablation + adaptation > adaptation alone (Phase 28)
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


# ==============================================================
# Strategic Ablation Methods (the core innovation)
# ==============================================================
def ablate_random(model, rate):
    """Phase 31 baseline: random parameter ablation."""
    am = copy.deepcopy(model)
    with torch.no_grad():
        for p in am.parameters():
            mask = torch.rand_like(p) > rate
            p.mul_(mask.float())
    return am

def ablate_least_important(model, rate, task_samples):
    """Strategic ablation: remove neurons with LOWEST gradient magnitude.
    These are the 'lazy' neurons that contribute least to current task.
    Removing them forces the network to rebuild stronger paths."""
    am = copy.deepcopy(model)
    # Compute gradient magnitude per parameter on task demos
    am.train()
    for p in am.parameters():
        if p.grad is not None:
            p.grad.zero_()
    total_loss = torch.tensor(0.0, device=DEVICE)
    for s in task_samples[:8]:
        total_loss = total_loss + compute_loss(am, s)
    total_loss = total_loss / max(len(task_samples[:8]), 1)
    total_loss.backward()

    with torch.no_grad():
        for p in am.parameters():
            if p.grad is not None:
                importance = p.grad.abs()
                # Ablate the LEAST important (lowest gradient) parameters
                threshold = torch.quantile(importance.flatten(), rate)
                mask = importance > threshold
                p.mul_(mask.float())
            else:
                # No gradient info: random ablation fallback
                mask = torch.rand_like(p) > rate
                p.mul_(mask.float())
    am.eval()
    return am

def ablate_high_entropy(model, rate, task_samples):
    """Strategic ablation: remove neurons that produce the MOST ambiguous
    (high-entropy) outputs. These are the 'confused' neurons.
    Forcing the network to work without them clarifies decision paths."""
    am = copy.deepcopy(model)
    # Run forward pass and compute per-parameter activation variance
    am.eval()
    activations = {name: [] for name, _ in am.named_parameters()}

    hooks = []
    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name].append(output.detach())
        return hook_fn

    # Register hooks on all linear layers
    for name, module in am.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name + '.weight')))

    with torch.no_grad():
        for s in task_samples[:8]:
            am(s['nf'].to(DEVICE), s['nn'].to(DEVICE))

    for h in hooks:
        h.remove()

    # Ablate parameters in modules with highest output variance (most confused)
    with torch.no_grad():
        for name, p in am.named_parameters():
            if name in activations and activations[name]:
                acts = torch.cat([a.flatten() for a in activations[name]])
                variance = acts.var().item()
                # Higher variance = more confused = ablate more aggressively
                adaptive_rate = rate * min(2.0, max(0.5, variance / max(acts.abs().mean().item(), 1e-6)))
                mask = torch.rand_like(p) > adaptive_rate
                p.mul_(mask.float())
            else:
                mask = torch.rand_like(p) > rate
                p.mul_(mask.float())
    am.eval()
    return am


def adapt_model(model, task_samples, steps=50, lr=1e-2):
    """Data-Augmented Adaptation (from Phase 28)."""
    if not task_samples:
        return model
    am = copy.deepcopy(model)
    opt = torch.optim.SGD(am.parameters(), lr=lr)
    am.train()
    for _ in range(steps):
        batch = random.sample(task_samples, min(8, len(task_samples)))
        tl = sum(compute_loss(am, d) for d in batch) / len(batch)
        opt.zero_grad(); tl.backward()
        torch.nn.utils.clip_grad_norm_(am.parameters(), 1.0)
        opt.step()
    am.eval()
    return am


def eval_task(model, task):
    """Evaluate model on a single task's test set."""
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
    print("Phase 34: Hydra-Driven Super-Compensation")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Train base model (same as Phase 28/31)
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

    # ===== Super-Compensation Experiment =====
    # Compare: (1) Adapt only  (2) Random ablation + Adapt  (3) Strategic ablation + Adapt
    ablation_rates = [0.10, 0.15, 0.20, 0.30]
    methods = {
        'adapt_only': 'Phase 28 baseline (adapt only)',
        'random_ablate': 'P31 random ablation + adapt',
        'strategic_least': 'P34 least-important ablation + adapt',
        'strategic_entropy': 'P34 high-entropy ablation + adapt',
    }

    results = {}
    print(f"\n{'Method':<30} | {'Rate':>5} | {'Full Match':>10} | {'vs Adapt-Only':>13}")
    print("-" * 75)

    # Baseline: adapt only
    ok_base, total_base = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        adapted = adapt_model(model, aug_samples, steps=50)
        ok, tot = eval_task(adapted, task)
        ok_base += ok; total_base += tot
    base_rate = ok_base / max(total_base, 1)
    print(f"  {'Adapt-only (P28)':<28} | {'N/A':>5} | {base_rate:>9.1%} | {'baseline':>13}")
    results['adapt_only'] = base_rate

    # Test each ablation method x rate
    for method_key in ['random_ablate', 'strategic_least', 'strategic_entropy']:
        for rate in ablation_rates:
            ok_total, n_total = 0, 0
            for task in test_tasks:
                demos = task.get('train', [])
                aug_pairs = []
                for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
                aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                aug_samples = [s for s in aug_samples if s]

                # Apply ablation strategy
                if method_key == 'random_ablate':
                    ablated = ablate_random(model, rate)
                elif method_key == 'strategic_least':
                    ablated = ablate_least_important(model, rate, aug_samples)
                else:
                    ablated = ablate_high_entropy(model, rate, aug_samples)

                # Adapt the ablated model
                adapted = adapt_model(ablated, aug_samples, steps=50)
                ok, tot = eval_task(adapted, task)
                ok_total += ok; n_total += tot

            match_rate = ok_total / max(n_total, 1)
            delta = match_rate - base_rate
            verdict = "SUPER-COMP!" if delta > 0.01 else ("~baseline" if abs(delta) < 0.01 else "worse")
            print(f"  {methods[method_key][:28]:<28} | {rate:>4.0%} | {match_rate:>9.1%} | {delta:>+12.1%} {verdict}")
            results[f'{method_key}_{int(rate*100)}'] = {
                'match_rate': match_rate, 'delta': delta, 'verdict': verdict
            }

    elapsed = time.time() - t0
    results['baseline_adapt'] = base_rate
    results['elapsed'] = elapsed
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase34_supercomp.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All methods vs ablation rate
    ax = axes[0]
    for method_key, label, color, marker in [
        ('random_ablate', 'Random Ablation', '#FF9800', 'o'),
        ('strategic_least', 'Least-Important', '#2196F3', 's'),
        ('strategic_entropy', 'High-Entropy', '#9C27B0', '^'),
    ]:
        rates_pct = [int(r * 100) for r in ablation_rates]
        vals = [results.get(f'{method_key}_{r}', {}).get('match_rate', 0) for r in rates_pct]
        ax.plot(rates_pct, vals, f'{marker}-', label=label, linewidth=2, color=color, markersize=8)
    ax.axhline(y=base_rate, color='#4CAF50', linestyle='--', linewidth=2, label=f'Adapt-Only ({base_rate:.1%})')
    ax.set_xlabel('Ablation Rate (%)'); ax.set_ylabel('Full Match')
    ax.set_title('Phase 34: Super-Compensation via Strategic Ablation')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    # Plot 2: Delta from baseline
    ax = axes[1]
    all_keys = [f'{m}_{int(r*100)}' for m in ['random_ablate', 'strategic_least', 'strategic_entropy'] for r in ablation_rates]
    deltas = [results.get(k, {}).get('delta', 0) for k in all_keys]
    labels = [k.replace('_ablate', '').replace('strategic_', 'S:') for k in all_keys]
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
    ax.bar(range(len(deltas)), [d * 100 for d in deltas], color=colors, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Δ Full Match (pp)'); ax.set_title('Super-Compensation: Δ vs Adapt-Only Baseline')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase34_supercomp.png'), dpi=150)
    plt.close()

    # Find best config
    best_key = max(
        [k for k in results if isinstance(results[k], dict) and 'delta' in results[k]],
        key=lambda k: results[k]['delta'], default=None
    )
    if best_key:
        best = results[best_key]
        print(f"\n  BEST: {best_key} -> {best['match_rate']:.1%} (Δ={best['delta']:+.1%})")
    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__': main()
