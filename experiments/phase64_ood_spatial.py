"""
Phase 64: OOD Spatial Generalization
======================================
Prove Structure > Scale for out-of-distribution grid sizes.

Train on small grids (<=15x15), test on large grids (>15x15).
Compare GlassBox (object-based, size-invariant) vs Transformer
(position-dependent, size-brittle).

Design: 2 architectures x 2 test conditions x 3 seeds.
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

# ============================================================
# Architecture A: GlassBox (object-based, size-invariant)
# ============================================================
class GlassBoxAgent(nn.Module):
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

# ============================================================
# Architecture B: Transformer (position-dependent)
# ============================================================
class TransformerAgent(nn.Module):
    def __init__(self, hid=128, n_layers=4, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(NODE_FEAT_DIM, hid)
        self.pos_embed = nn.Embedding(MAX_OBJECTS, hid)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid, nhead=n_heads, dim_feedforward=hid*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.Linear(hid, hid)
        self.oh = nn.Linear(hid, N_OPS)
        self.c1h = nn.Linear(hid, N_COLORS)
        self.c2h = nn.Linear(hid, N_COLORS)
        self.pq = nn.Linear(hid, hid)
        self.pk = nn.Linear(hid, hid)
    def forward(self, nf, nn_c):
        B = nf.size(0)
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        pos = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0).expand(B, -1)
        h = self.embed(nf) + self.pos_embed(pos)
        pad_mask = ~mask
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        h = h * mf
        g = F.relu(self.pool((h * mf).sum(1) / mf.sum(1).clamp(min=1)))
        pl = ((self.pq(g).unsqueeze(1)) * self.pk(h)).sum(-1).masked_fill(~mask, -1e9)
        return self.oh(g), self.c1h(g), self.c2h(g), pl

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

def max_grid_size(task):
    """Get max dimension across all grids in a task."""
    max_dim = 0
    for split in ['train', 'test']:
        for pair in task.get(split, []):
            for key in ['input', 'output']:
                g = pair.get(key, [])
                if g:
                    h, w = len(g), len(g[0]) if g else 0
                    max_dim = max(max_dim, h, w)
    return max_dim


def main():
    print("=" * 60)
    print("Phase 64: OOD Spatial Generalization")
    print("Train small (<=15), test large (>15)")
    print("Structure > Scale for spatial robustness")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)

    # Split by grid size
    SIZE_THRESH = 15
    small_tasks = [t for t in tasks if max_grid_size(t) <= SIZE_THRESH]
    large_tasks = [t for t in tasks if max_grid_size(t) > SIZE_THRESH]
    all_test = tasks[int(len(tasks)*0.8):]
    small_test = [t for t in all_test if max_grid_size(t) <= SIZE_THRESH]
    large_test = [t for t in all_test if max_grid_size(t) > SIZE_THRESH]

    print(f"  Small tasks (<={SIZE_THRESH}): {len(small_tasks)} total, {len(small_test)} test")
    print(f"  Large tasks (>{SIZE_THRESH}): {len(large_tasks)} total, {len(large_test)} test")

    architectures = {
        'glassbox': {'class': GlassBoxAgent, 'label': 'GlassBox (77K)'},
        'transformer': {'class': TransformerAgent, 'label': 'Transformer (~850K)'},
    }

    N_SEEDS = 3
    all_results = {}

    for arch_name, arch_cfg in architectures.items():
        print(f"\n  ===== {arch_cfg['label']} =====")
        for test_label, test_set in [('small', small_test), ('large', large_test)]:
            key = f'{arch_name}_{test_label}'
            all_results[key] = []

        for seed in range(N_SEEDS):
            print(f"\n    --- Seed {seed+1}/{N_SEEDS} ---")
            random.seed(seed * 1000)
            np.random.seed(seed * 1000)
            torch.manual_seed(seed * 1000)

            # Train ONLY on small tasks
            train_small = small_tasks[:int(len(small_tasks)*0.8)]
            all_samples = []
            for task in train_small:
                for p in task.get('train', []):
                    s = prep(p['input'], p['output'])
                    if s: all_samples.append(s)

            model = arch_cfg['class']().to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters())
            if seed == 0:
                print(f"      Parameters: {n_params:,}")
                print(f"      Training on {len(all_samples)} samples from {len(train_small)} small tasks")

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for ep in range(80):
                model.train(); random.shuffle(all_samples); el, nb = 0, 0
                for s in all_samples:
                    loss = compute_loss(model, s)
                    opt.zero_grad(); loss.backward(); opt.step()
                    el += loss.item(); nb += 1
            print(f"      Train loss: {el/nb:.4f}")

            # Evaluate on both small and large test sets
            for test_label, test_set in [('small', small_test), ('large', large_test)]:
                if not test_set:
                    all_results[f'{arch_name}_{test_label}'].append(0.0)
                    print(f"      {test_label} test: N/A (no tasks)")
                    continue

                ok_total, n_total = 0, 0
                for task in test_set:
                    demos = task.get('train', [])
                    aug_pairs = []
                    for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
                    aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                    aug_samples = [s for s in aug_samples if s]

                    current = ablate_least_important(model, 0.15, aug_samples)
                    current = adapt_model(current, aug_samples, steps=100)
                    ok, tot = eval_task(current, task)
                    ok_total += ok; n_total += tot

                rate = ok_total / max(n_total, 1)
                all_results[f'{arch_name}_{test_label}'].append(rate)
                print(f"      {test_label} test: {rate:.1%}")

    # Summary
    results = {}
    print(f"\n  === OOD SPATIAL GENERALIZATION ({N_SEEDS} seeds) ===")
    print(f"  {'Architecture':<25} | {'Test Set':<8} | {'Mean':>6} | {'Std':>5}")
    print("  " + "-" * 55)

    for arch_name, arch_cfg in architectures.items():
        for test_label in ['small', 'large']:
            key = f'{arch_name}_{test_label}'
            vals = np.array(all_results[key])
            mean, std = float(vals.mean()), float(vals.std())
            results[f'{key}_mean'] = mean
            results[f'{key}_std'] = std
            results[f'{key}_all'] = [float(v) for v in vals]
            print(f"  {arch_cfg['label']:<25} | {test_label:<8} | {mean:>5.1%} | {std:>4.1%}")

    # OOD gap calculation
    gb_gap = results.get('glassbox_small_mean', 0) - results.get('glassbox_large_mean', 0)
    tf_gap = results.get('transformer_small_mean', 0) - results.get('transformer_large_mean', 0)
    results['glassbox_ood_gap'] = gb_gap
    results['transformer_ood_gap'] = tf_gap

    print(f"\n  OOD Gap (small->large drop):")
    print(f"    GlassBox:    {gb_gap:+.1%}")
    print(f"    Transformer: {tf_gap:+.1%}")
    if abs(gb_gap) < abs(tf_gap):
        print(f"  *** GlassBox is more robust to OOD spatial changes! ***")

    results['n_seeds'] = N_SEEDS
    results['size_threshold'] = SIZE_THRESH
    results['n_small_test'] = len(small_test)
    results['n_large_test'] = len(large_test)
    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase64_ood_spatial.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Grouped bar chart
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    gb_means = [results.get('glassbox_small_mean', 0), results.get('glassbox_large_mean', 0)]
    gb_stds = [results.get('glassbox_small_std', 0), results.get('glassbox_large_std', 0)]
    tf_means = [results.get('transformer_small_mean', 0), results.get('transformer_large_mean', 0)]
    tf_stds = [results.get('transformer_small_std', 0), results.get('transformer_large_std', 0)]

    bars1 = ax.bar(x - width/2, gb_means, width, yerr=gb_stds, label='GlassBox', color='#4CAF50', alpha=0.85, capsize=4)
    bars2 = ax.bar(x + width/2, tf_means, width, yerr=tf_stds, label='Transformer', color='#FF9800', alpha=0.85, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(['Small (<=15)\nIn-Distribution', 'Large (>15)\nOut-of-Distribution'])
    ax.set_ylabel('Full Match'); ax.set_title('OOD Spatial Generalization')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    for b, m in zip(bars1, gb_means):
        ax.text(b.get_x()+b.get_width()/2, m+0.03, f'{m:.1%}', ha='center', fontsize=10, fontweight='bold')
    for b, m in zip(bars2, tf_means):
        ax.text(b.get_x()+b.get_width()/2, m+0.03, f'{m:.1%}', ha='center', fontsize=10, fontweight='bold')

    # Right: OOD gap comparison
    ax = axes[1]
    gaps = [abs(gb_gap), abs(tf_gap)]
    gap_labels = ['GlassBox', 'Transformer']
    gap_colors = ['#4CAF50', '#FF9800']
    bars = ax.bar(range(2), gaps, color=gap_colors, alpha=0.85)
    ax.set_xticks(range(2)); ax.set_xticklabels(gap_labels)
    ax.set_ylabel('OOD Performance Gap (lower=better)')
    ax.set_title('Spatial Robustness Gap')
    ax.grid(True, alpha=0.3, axis='y')
    for b, g in zip(bars, gaps):
        ax.text(b.get_x()+b.get_width()/2, g+0.01, f'{g:.1%}', ha='center', fontsize=12, fontweight='bold')

    plt.suptitle('Phase 64: OOD Spatial Generalization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase64_ood_spatial.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
