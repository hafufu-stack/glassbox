"""
Phase 65: Ablation Topology (Layer-Wise Analysis)
===================================================
P59 showed 15% gradient ablation reduces variance.
P61 showed this is GlassBox-specific, not Transformer.
WHY? Where in the network does ablation help?

Test layer-specific ablation:
  1. Embedding layer only
  2. GNN Layer 1 only
  3. GNN Layer 2 only
  4. Readout heads only
  5. All layers (baseline)

3 seeds × 5 conditions + baseline.
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
    """GlassBox Agent with named layer groups for targeted ablation."""
    def __init__(s,hid=64):
        super().__init__()
        # Group 1: Embedding
        s.ne=nn.Linear(NODE_FEAT_DIM,hid)
        # Group 2: GNN Layer 1
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid)
        # Group 3: GNN Layer 2
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n2=nn.LayerNorm(hid)
        # Group 4: Readout heads
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)

    def get_layer_groups(s):
        """Return named groups of parameters for targeted ablation."""
        return {
            'embed': [s.ne],
            'gnn1': [s.g1, s.n1],
            'gnn2': [s.g2, s.n2],
            'readout': [s.oh, s.c1h, s.c2h, s.pq, s.pk],
        }

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

def ablate_targeted(model, rate, task_samples, target_groups=None):
    """Ablate specific layer groups. If target_groups is None, ablate all."""
    am = copy.deepcopy(model); am.train()
    total_loss = torch.tensor(0.0, device=DEVICE)
    for s in task_samples[:8]:
        total_loss = total_loss + compute_loss(am, s)
    total_loss = total_loss / max(len(task_samples[:8]), 1)
    total_loss.backward()

    # Determine which parameters to ablate
    if target_groups is not None:
        groups = am.get_layer_groups()
        target_params = set()
        for gname in target_groups:
            for module in groups[gname]:
                for p in module.parameters():
                    target_params.add(id(p))
    else:
        target_params = None  # ablate all

    with torch.no_grad():
        for p in am.parameters():
            if target_params is not None and id(p) not in target_params:
                continue  # skip non-target layers
            if p.grad is not None:
                importance = p.grad.abs()
                threshold = torch.quantile(importance.flatten(), rate)
                p.mul_((importance > threshold).float())
            else:
                p.mul_((torch.rand_like(p) > rate).float())
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

def load_arc_tasks(d, n=400):
    t = []
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                t.append({'id': f[:-5], **json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 65: Ablation Topology (Layer-Wise Analysis)")
    print("Which layer benefits most from pruning?")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    # Ablation conditions
    conditions = {
        'baseline': {'groups': None, 'rate': 0.0, 'label': 'No Ablation'},
        'all_15': {'groups': None, 'rate': 0.15, 'label': 'All Layers (15%)'},
        'embed': {'groups': ['embed'], 'rate': 0.15, 'label': 'Embedding Only'},
        'gnn1': {'groups': ['gnn1'], 'rate': 0.15, 'label': 'GNN Layer 1 Only'},
        'gnn2': {'groups': ['gnn2'], 'rate': 0.15, 'label': 'GNN Layer 2 Only'},
        'readout': {'groups': ['readout'], 'rate': 0.15, 'label': 'Readout Only'},
        'gnn_both': {'groups': ['gnn1', 'gnn2'], 'rate': 0.15, 'label': 'Both GNN Layers'},
    }

    N_SEEDS = 5
    all_results = {cname: [] for cname in conditions}

    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        random.seed(seed * 1000)
        np.random.seed(seed * 1000)
        torch.manual_seed(seed * 1000)

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

        # Count params per group
        if seed == 0:
            groups = model.get_layer_groups()
            for gname, modules in groups.items():
                n_params = sum(p.numel() for m in modules for p in m.parameters())
                print(f"    {gname}: {n_params:,} params")

        for cname, cfg in conditions.items():
            ok_total, n_total = 0, 0
            for task in test_tasks:
                demos = task.get('train', [])
                aug_pairs = []
                for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
                aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                aug_samples = [s for s in aug_samples if s]

                current = model
                if cfg['rate'] > 0:
                    current = ablate_targeted(current, cfg['rate'], aug_samples, cfg['groups'])
                current = adapt_model(current, aug_samples, steps=100)
                ok, tot = eval_task(current, task)
                ok_total += ok; n_total += tot

            rate = ok_total / max(n_total, 1)
            all_results[cname].append(rate)
            print(f"    {cfg['label']:<22}: {rate:.1%}")

    # Summary
    results = {}
    print(f"\n  === ABLATION TOPOLOGY ({N_SEEDS} seeds) ===")
    print(f"  {'Condition':<22} | {'Mean':>6} | {'Std':>5} | {'vs BL':>6}")
    print("  " + "-" * 50)

    bl_mean = float(np.mean(all_results['baseline']))

    for cname, cfg in conditions.items():
        vals = np.array(all_results[cname])
        mean, std = float(vals.mean()), float(vals.std())
        results[f'{cname}_mean'] = mean
        results[f'{cname}_std'] = std
        results[f'{cname}_all'] = [float(v) for v in vals]
        delta = mean - bl_mean
        delta_str = f"{delta:+.1%}" if cname != 'baseline' else "  ---"
        print(f"  {cfg['label']:<22} | {mean:>5.1%} | {std:>4.1%} | {delta_str}")

    # Find best single-layer ablation
    single_layers = ['embed', 'gnn1', 'gnn2', 'readout']
    best_layer = max(single_layers, key=lambda l: results[f'{l}_mean'])
    results['best_single_layer'] = best_layer
    results['best_single_mean'] = results[f'{best_layer}_mean']

    # Variance analysis
    bl_std = results['baseline_std']
    print(f"\n  Variance analysis:")
    for cname in conditions:
        if cname == 'baseline': continue
        std = results[f'{cname}_std']
        ratio = bl_std / max(std, 1e-6)
        results[f'{cname}_var_reduction'] = float(ratio)
        print(f"    {conditions[cname]['label']:<22}: std={std:.1%}, reduction={ratio:.1f}×")

    results['n_seeds'] = N_SEEDS
    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase65_topology.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Mean accuracy by condition
    ax = axes[0]
    cnames = list(conditions.keys())
    labels = [conditions[c]['label'] for c in cnames]
    means = [results[f'{c}_mean'] for c in cnames]
    stds = [results[f'{c}_std'] for c in cnames]
    colors = ['#9E9E9E', '#E91E63', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4']
    bars = ax.bar(range(len(cnames)), means, yerr=stds, color=colors[:len(cnames)], alpha=0.85, capsize=3)
    ax.set_xticks(range(len(cnames)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Full Match'); ax.set_title('Accuracy by Ablation Target')
    ax.set_ylim(0.7, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m+0.02, f'{m:.1%}', ha='center', fontsize=8, fontweight='bold')

    # Right: Variance reduction by layer
    ax = axes[1]
    layer_names = ['embed', 'gnn1', 'gnn2', 'readout', 'gnn_both', 'all_15']
    layer_labels = ['Embed', 'GNN1', 'GNN2', 'Readout', 'Both\nGNN', 'All\nLayers']
    var_reductions = [results.get(f'{l}_var_reduction', 1.0) for l in layer_names]
    colors_v = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4', '#E91E63']
    bars = ax.bar(range(len(layer_names)), var_reductions, color=colors_v, alpha=0.85)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(layer_names))); ax.set_xticklabels(layer_labels)
    ax.set_ylabel('Variance Reduction (×, ↑ better)')
    ax.set_title('Which Layer Reduces Variance Most?')
    ax.grid(True, alpha=0.3, axis='y')
    for b, r in zip(bars, var_reductions):
        ax.text(b.get_x()+b.get_width()/2, r+0.05, f'{r:.1f}×', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Phase 65: Ablation Topology', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase65_topology.png'), dpi=150)
    plt.close()

    print(f"\n  *** Best single layer: {best_layer} ({results[f'{best_layer}_mean']:.1%}) ***")
    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
