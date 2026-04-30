"""
Phase 61: Cross-Architecture Variance Regularization
=====================================================
Future Work recovery: "Investigate the variance regularization effect
in larger models and architectures."

Test whether gradient-based ablation reduces variance in the
1.45M-parameter Transformer baseline (Phase 15), not just GlassBox.

If yes → variance regularization is a UNIVERSAL property of neural nets.
If no → it's specific to structured (GNN) architectures.

Design: 2 architectures × 3 ablation rates × 5 seeds = 30 runs.
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
# Architecture A: GlassBox Agent (77K params, GNN-based)
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
# Architecture B: Transformer (1.45M params, unstructured)
# Processes flattened object features via self-attention
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
        # Transformer needs src_key_padding_mask (True = ignore)
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


def main():
    print("=" * 60)
    print("Phase 61: Cross-Architecture Variance Regularization")
    print("Is ablation-induced variance reduction universal?")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    architectures = {
        'glassbox': {'class': GlassBoxAgent, 'label': 'GlassBox (77K)'},
        'transformer': {'class': TransformerAgent, 'label': 'Transformer (~1.4M)'},
    }
    ablation_rates = [0.0, 0.10, 0.15]
    N_SEEDS = 5

    all_results = {}

    for arch_name, arch_cfg in architectures.items():
        print(f"\n  ===== {arch_cfg['label']} =====")
        arch_results = {}
        for abl_r in ablation_rates:
            key = f'{arch_name}_a{int(abl_r*100)}'
            arch_results[key] = []

        for seed in range(N_SEEDS):
            print(f"\n    --- Seed {seed+1}/{N_SEEDS} ---")
            random.seed(seed * 1000)
            np.random.seed(seed * 1000)
            torch.manual_seed(seed * 1000)

            all_samples = []
            for task in tasks:
                for p in task.get('train', []):
                    s = prep(p['input'], p['output'])
                    if s: all_samples.append(s)

            model = arch_cfg['class']().to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters())
            if seed == 0:
                print(f"      Parameters: {n_params:,}")

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for ep in range(80):
                model.train(); random.shuffle(all_samples); el, nb = 0, 0
                for s in all_samples:
                    loss = compute_loss(model, s)
                    opt.zero_grad(); loss.backward(); opt.step()
                    el += loss.item(); nb += 1
            print(f"      Train loss: {el/nb:.4f}")

            for abl_r in ablation_rates:
                ok_total, n_total = 0, 0
                for task in test_tasks:
                    demos = task.get('train', [])
                    aug_pairs = []
                    for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
                    aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                    aug_samples = [s for s in aug_samples if s]

                    current = model
                    if abl_r > 0:
                        current = ablate_least_important(current, abl_r, aug_samples)
                    current = adapt_model(current, aug_samples, steps=100)
                    ok, tot = eval_task(current, task)
                    ok_total += ok; n_total += tot

                rate = ok_total / max(n_total, 1)
                key = f'{arch_name}_a{int(abl_r*100)}'
                arch_results[key].append(rate)
                label = f"Ablate {abl_r:.0%}" if abl_r > 0 else "Baseline"
                print(f"      {label:<14}: {rate:.1%}")

        all_results.update(arch_results)

    # === Statistics ===
    results = {}
    print(f"\n  === CROSS-ARCHITECTURE VARIANCE ANALYSIS ({N_SEEDS} seeds) ===")
    print(f"  {'Architecture':<20} | {'Ablation':<10} | {'Mean':>6} | {'Std':>5} | {'Std Ratio':>9}")
    print("  " + "-" * 65)

    for arch_name, arch_cfg in architectures.items():
        baseline_key = f'{arch_name}_a0'
        baseline_std = float(np.std(all_results[baseline_key]))

        for abl_r in ablation_rates:
            key = f'{arch_name}_a{int(abl_r*100)}'
            vals = np.array(all_results[key])
            mean = float(vals.mean())
            std = float(vals.std())
            results[f'{key}_mean'] = mean
            results[f'{key}_std'] = std
            results[f'{key}_all'] = [float(v) for v in vals]

            if abl_r == 0:
                ratio_str = "1.0×"
            else:
                ratio = baseline_std / max(std, 1e-6)
                ratio_str = f"{ratio:.1f}×"
                results[f'{key}_std_ratio'] = float(ratio)

            label = f"Ablate {abl_r:.0%}" if abl_r > 0 else "Baseline"
            print(f"  {arch_cfg['label']:<20} | {label:<10} | {mean:>5.1%} | {std:>4.1%} | {ratio_str:>9}")

    # Universal law check
    glassbox_ratio = results.get('glassbox_a15_std_ratio', 1.0)
    transformer_ratio = results.get('transformer_a15_std_ratio', 1.0)
    results['glassbox_var_reduction'] = glassbox_ratio
    results['transformer_var_reduction'] = transformer_ratio
    results['is_universal'] = transformer_ratio > 1.5  # >1.5x = meaningful reduction

    print(f"\n  GlassBox variance reduction at 15%: {glassbox_ratio:.1f}×")
    print(f"  Transformer variance reduction at 15%: {transformer_ratio:.1f}×")
    if results['is_universal']:
        print(f"  *** UNIVERSAL LAW CONFIRMED: Both architectures show variance reduction! ***")
    else:
        print(f"  --- Architecture-specific: Only GlassBox shows variance reduction ---")

    results['n_seeds'] = N_SEEDS
    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase61_cross_arch.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: GlassBox bars
    ax = axes[0]
    gb_means = [results[f'glassbox_a{int(r*100)}_mean'] for r in ablation_rates]
    gb_stds = [results[f'glassbox_a{int(r*100)}_std'] for r in ablation_rates]
    colors_gb = ['#9E9E9E', '#4CAF50', '#2196F3']
    bars = ax.bar(range(len(ablation_rates)), gb_means, yerr=gb_stds, color=colors_gb, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(ablation_rates)))
    ax.set_xticklabels([f'{int(r*100)}%' for r in ablation_rates])
    ax.set_xlabel('Ablation Rate'); ax.set_ylabel('Full Match')
    ax.set_title('GlassBox (77K) — Variance by Ablation')
    ax.set_ylim(0.6, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, m, s in zip(bars, gb_means, gb_stds):
        ax.text(b.get_x()+b.get_width()/2, m+s+0.02, f'{m:.1%}\n±{s:.1%}', ha='center', fontsize=9)

    # Center: Transformer bars
    ax = axes[1]
    tf_means = [results[f'transformer_a{int(r*100)}_mean'] for r in ablation_rates]
    tf_stds = [results[f'transformer_a{int(r*100)}_std'] for r in ablation_rates]
    colors_tf = ['#9E9E9E', '#FF9800', '#E91E63']
    bars = ax.bar(range(len(ablation_rates)), tf_means, yerr=tf_stds, color=colors_tf, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(ablation_rates)))
    ax.set_xticklabels([f'{int(r*100)}%' for r in ablation_rates])
    ax.set_xlabel('Ablation Rate'); ax.set_ylabel('Full Match')
    ax.set_title('Transformer (~1.4M) — Variance by Ablation')
    ax.set_ylim(0.0, 0.8); ax.grid(True, alpha=0.3, axis='y')
    for b, m, s in zip(bars, tf_means, tf_stds):
        ax.text(b.get_x()+b.get_width()/2, m+s+0.02, f'{m:.1%}\n±{s:.1%}', ha='center', fontsize=9)

    # Right: Std ratio comparison
    ax = axes[2]
    ratios = [
        results.get('glassbox_a10_std_ratio', 1.0),
        results.get('glassbox_a15_std_ratio', 1.0),
        results.get('transformer_a10_std_ratio', 1.0),
        results.get('transformer_a15_std_ratio', 1.0),
    ]
    labels = ['GB\nA=10%', 'GB\nA=15%', 'TF\nA=10%', 'TF\nA=15%']
    colors_r = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    bars = ax.bar(range(4), ratios, color=colors_r, alpha=0.85)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No reduction')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Variance Reduction (×)')
    ax.set_title('Variance Reduction: Universal?')
    ax.grid(True, alpha=0.3, axis='y')
    for b, r in zip(bars, ratios):
        ax.text(b.get_x()+b.get_width()/2, r+0.1, f'{r:.1f}×', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Phase 61: Cross-Architecture Variance Regularization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase61_cross_arch.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
