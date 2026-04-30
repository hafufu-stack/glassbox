"""
Phase 60: Variance-Asymmetric Portfolio (Deep Think proposal)
==============================================================
Kaggle allows up to 3 answers per task. Instead of 3 identical models,
use 3 models with DIFFERENT variance profiles:
  1. "Iron Wall" (Ablate 15%): std=0.7%, never crashes
  2. "Baseline" (Ablate 0%): std=3.0%, high ceiling potential
  3. "Gambler" (Ablate 25% + Neuro 5%): P55's 90.8% peak, huge variance

The portfolio wins if ANY of the 3 gets the right answer.
Compare: portfolio vs uniform ensemble vs single best.

5 seeds × 3 strategies + portfolio analysis.
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

def add_neurons(model, rate, seed=42):
    """Add randomly initialized neurons (neurogenesis)."""
    am = copy.deepcopy(model)
    # Use manual_seed globally (generator doesn't work with CUDA tensors)
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in am.parameters():
            mask = torch.rand_like(p) < rate
            noise = torch.randn_like(p) * 0.01
            p.add_(mask.float() * noise)
    return am

def eval_task_detailed(model, task):
    """Return per-test-pair correctness (list of bools)."""
    results = []
    for tp in task.get('test', []):
        s = prep(tp['input'], tp['output'])
        if s is None:
            results.append(False)
            continue
        with torch.no_grad():
            ol, cl1, cl2, pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        correct = (ol.argmax(1).item() == s['op'].item() and
                   cl1.argmax(1).item() == s['c1'].item() and
                   cl2.argmax(1).item() == s['c2'].item() and
                   pl.argmax(1).item() == s['ptr'].item())
        results.append(correct)
    return results

def load_arc_tasks(d, n=400):
    t = []
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                t.append({'id': f[:-5], **json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 60: Variance-Asymmetric Portfolio")
    print("Does combining low-var + high-var strategies beat uniform?")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    # Strategy definitions
    strategies = {
        'iron_wall': {'ablate': 0.15, 'neuro': 0.0, 'label': 'Iron Wall (A=15%)'},
        'baseline':  {'ablate': 0.0,  'neuro': 0.0, 'label': 'Baseline (A=0%)'},
        'gambler':   {'ablate': 0.25, 'neuro': 0.05, 'label': 'Gambler (A=25%+N=5%)'},
    }

    N_SEEDS = 5
    # Store per-task, per-strategy correctness: [seed][strategy][task_idx] -> bool
    all_task_results = {}

    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        random.seed(seed * 1000)
        np.random.seed(seed * 1000)
        torch.manual_seed(seed * 1000)

        # Train base model
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
        print(f"    Train loss: {el/nb:.4f}")

        seed_results = {}
        for strat_name, strat_cfg in strategies.items():
            task_correct = []  # per-task boolean
            ok_total, n_total = 0, 0
            for task in test_tasks:
                demos = task.get('train', [])
                aug_pairs = []
                for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
                aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                aug_samples = [s for s in aug_samples if s]

                current = model
                if strat_cfg['ablate'] > 0:
                    current = ablate_least_important(current, strat_cfg['ablate'], aug_samples)
                if strat_cfg['neuro'] > 0:
                    current = add_neurons(current, strat_cfg['neuro'], seed=seed*100+hash(task['id'])%1000)
                current = adapt_model(current, aug_samples, steps=100)

                per_pair = eval_task_detailed(current, task)
                task_ok = all(per_pair) if per_pair else False
                task_correct.append(task_ok)
                ok = sum(per_pair)
                ok_total += ok; n_total += len(per_pair)

            rate = ok_total / max(n_total, 1)
            seed_results[strat_name] = {
                'rate': rate,
                'task_correct': task_correct,
            }
            print(f"    {strat_cfg['label']:<30}: {rate:.1%}")

        all_task_results[seed] = seed_results

    # === Portfolio Analysis ===
    print(f"\n  === PORTFOLIO ANALYSIS ({N_SEEDS} seeds) ===")

    results = {}
    for strat_name in strategies:
        rates = [all_task_results[s][strat_name]['rate'] for s in range(N_SEEDS)]
        results[f'{strat_name}_mean'] = float(np.mean(rates))
        results[f'{strat_name}_std'] = float(np.std(rates))
        results[f'{strat_name}_all'] = rates

    # Portfolio: for each seed, compute "any of 3 correct"
    portfolio_rates = []
    for seed in range(N_SEEDS):
        n_tasks = len(test_tasks)
        portfolio_correct = 0
        portfolio_total = 0
        for ti in range(n_tasks):
            # A task is "solved" if ANY strategy got it right
            any_correct = any(
                all_task_results[seed][sn]['task_correct'][ti]
                for sn in strategies
            )
            portfolio_correct += int(any_correct)
            portfolio_total += 1
        portfolio_rate = portfolio_correct / max(portfolio_total, 1)
        portfolio_rates.append(portfolio_rate)

    results['portfolio_mean'] = float(np.mean(portfolio_rates))
    results['portfolio_std'] = float(np.std(portfolio_rates))
    results['portfolio_all'] = portfolio_rates

    # Uniform ensemble (3x same strategy = iron_wall)
    uniform_rates = [all_task_results[s]['iron_wall']['rate'] for s in range(N_SEEDS)]
    results['uniform_mean'] = float(np.mean(uniform_rates))
    results['uniform_std'] = float(np.std(uniform_rates))

    # Print summary
    print(f"\n  {'Strategy':<30} | {'Mean':>6} | {'Std':>5}")
    print("  " + "-" * 50)
    for strat_name, strat_cfg in strategies.items():
        m = results[f'{strat_name}_mean']
        s = results[f'{strat_name}_std']
        print(f"  {strat_cfg['label']:<30} | {m:>5.1%} | {s:>4.1%}")
    print(f"  {'Uniform (3x Iron Wall)':<30} | {results['uniform_mean']:>5.1%} | {results['uniform_std']:>4.1%}")
    print(f"  {'PORTFOLIO (any-of-3)':<30} | {results['portfolio_mean']:>5.1%} | {results['portfolio_std']:>4.1%}")

    # Improvement
    delta = results['portfolio_mean'] - results['iron_wall_mean']
    print(f"\n  Portfolio vs Iron Wall: {delta:+.1%}")
    print(f"  Portfolio vs Baseline: {results['portfolio_mean'] - results['baseline_mean']:+.1%}")

    results['n_seeds'] = N_SEEDS
    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase60_portfolio.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Strategy comparison bar chart
    ax = axes[0]
    labels = [strategies[s]['label'] for s in strategies] + ['Portfolio\n(any-of-3)']
    means = [results[f'{s}_mean'] for s in strategies] + [results['portfolio_mean']]
    stds = [results[f'{s}_std'] for s in strategies] + [results['portfolio_std']]
    colors = ['#4CAF50', '#9E9E9E', '#FF5722', '#2196F3']
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Task-Level Accuracy')
    ax.set_title('Strategy Comparison: Individual vs Portfolio')
    ax.set_ylim(0.5, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m+0.03, f'{m:.1%}', ha='center', fontsize=10, fontweight='bold')

    # Right: Per-seed lines
    ax = axes[1]
    for sn, cfg in strategies.items():
        vals = [all_task_results[s][sn]['rate'] for s in range(N_SEEDS)]
        ax.plot(range(1, N_SEEDS+1), vals, 'o-', label=cfg['label'], markersize=6)
    ax.plot(range(1, N_SEEDS+1), portfolio_rates, 's-', label='Portfolio', markersize=8, linewidth=2, color='#2196F3')
    ax.set_xlabel('Seed'); ax.set_ylabel('Task-Level Accuracy')
    ax.set_title('Per-Seed Performance')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    plt.suptitle('Phase 60: Variance-Asymmetric Portfolio', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase60_portfolio.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
