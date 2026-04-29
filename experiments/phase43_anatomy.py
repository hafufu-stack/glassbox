"""
Phase 43: Anatomy of Super-Compensation
==========================================
WHY does one-shot ablation + adaptation beat everything else?
This phase dissects the mechanism:

1. Extract weight matrices BEFORE and AFTER super-compensation
2. Compute ΔW (weight change) per layer
3. Measure "pointer attention" pattern shift (Cross-Attention Map)
4. Visualize: which connections were destroyed and which NEW ones formed

The hypothesis: ablation kills overfit shortcuts, and adaptation
builds NEW, more general causal paths (Re-wiring).
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

def get_pointer_attention(model, sample):
    """Extract the pointer attention pattern: softmax(Q*K^T)."""
    model.eval()
    with torch.no_grad():
        nf = sample['nf'].to(DEVICE)
        nn_c = sample['nn'].to(DEVICE)
        mask = torch.arange(MAX_OBJECTS, device=DEVICE).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        h = model.ne(nf)
        msg = (h * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + model.g1(torch.cat([h, msg.expand_as(h)], -1))
        h = model.n1(h) * mf
        msg = (h * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + model.g2(torch.cat([h, msg.expand_as(h)], -1))
        h = model.n2(h) * mf
        g = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
        # Pointer attention scores
        q = model.pq(g)  # (1, hid)
        k = model.pk(h)  # (1, MAX_OBJECTS, hid)
        scores = (q.unsqueeze(1) * k).sum(-1)  # (1, MAX_OBJECTS)
        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)  # (1, MAX_OBJECTS)
    return attn.cpu().numpy()[0], nn_c.item()

def load_arc_tasks(d, n=400):
    t = []
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                t.append({'id': f[:-5], **json.load(fp)})
    return t

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


def main():
    print("=" * 60)
    print("Phase 43: Anatomy of Super-Compensation")
    print("Why does one-punch work?")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    all_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep(p['input'], p['output'])
            if s: all_samples.append(s)
    print(f"Total training samples: {len(all_samples)}")

    # Train base model
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

    # Save base model weights
    base_weights = {name: p.data.clone() for name, p in model.named_parameters()}

    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    # ===== Anatomy Experiment =====
    # For each test task, compare 3 models:
    #   A) Adapt-only (no ablation)
    #   B) Ablate 15% + Adapt (super-compensation)
    # Track: ΔW per layer, pointer attention shifts, accuracy

    layer_delta_adapt = {name: [] for name in base_weights}
    layer_delta_super = {name: [] for name in base_weights}
    attn_shifts = []
    sparsity_before = []
    sparsity_after = []

    ok_adapt, ok_super, n_total = 0, 0, 0
    n_tasks_analyzed = 0

    for task in test_tasks[:40]:  # Analyze 40 tasks for detail
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        demo_samples = [prep(p['input'], p['output']) for p in demos]
        demo_samples = [s for s in demo_samples if s]
        if not aug_samples or not demo_samples:
            continue

        # A) Adapt-only
        adapted = adapt_model(model, aug_samples, steps=50)
        ok_a, tot = eval_task(adapted, task)
        ok_adapt += ok_a; n_total += tot

        # B) Ablate 15% + Adapt
        ablated = ablate_least_important(model, 0.15, aug_samples)

        # Measure sparsity after ablation (before recovery)
        total_params = sum(p.numel() for p in ablated.parameters())
        zero_params = sum((p.data == 0).sum().item() for p in ablated.parameters())
        sparsity_before.append(zero_params / total_params)

        super_adapted = adapt_model(ablated, aug_samples, steps=50)
        ok_s, _ = eval_task(super_adapted, task)
        ok_super += ok_s

        # Measure sparsity after recovery
        zero_after = sum((p.data == 0).sum().item() for p in super_adapted.parameters())
        sparsity_after.append(zero_after / total_params)

        # Track ΔW per layer
        for name, p_base in base_weights.items():
            p_adapt = dict(adapted.named_parameters())[name].data
            p_super = dict(super_adapted.named_parameters())[name].data
            dw_adapt = (p_adapt - p_base).abs().mean().item()
            dw_super = (p_super - p_base).abs().mean().item()
            layer_delta_adapt[name].append(dw_adapt)
            layer_delta_super[name].append(dw_super)

        # Track pointer attention shift
        s0 = demo_samples[0]
        attn_base, n_obj = get_pointer_attention(model, s0)
        attn_adapt, _ = get_pointer_attention(adapted, s0)
        attn_super, _ = get_pointer_attention(super_adapted, s0)

        # KL divergence of attention distributions
        eps = 1e-8
        attn_b = attn_base[:n_obj] + eps
        attn_a = attn_adapt[:n_obj] + eps
        attn_s = attn_super[:n_obj] + eps
        attn_b /= attn_b.sum()
        attn_a /= attn_a.sum()
        attn_s /= attn_s.sum()

        kl_adapt = float(np.sum(attn_a * np.log(attn_a / attn_b)))
        kl_super = float(np.sum(attn_s * np.log(attn_s / attn_b)))
        attn_shifts.append({'kl_adapt': kl_adapt, 'kl_super': kl_super,
                           'attn_base': attn_base[:n_obj].tolist(),
                           'attn_super': attn_super[:n_obj].tolist()})
        n_tasks_analyzed += 1

    adapt_rate = ok_adapt / max(n_total, 1)
    super_rate = ok_super / max(n_total, 1)

    print(f"\n  Adapt-only:       {adapt_rate:.1%}")
    print(f"  Super-comp (15%): {super_rate:.1%} (delta: {super_rate - adapt_rate:+.1%})")

    # ===== Analysis =====
    # 1. Average ΔW per layer
    layer_names = list(base_weights.keys())
    mean_dw_adapt = [np.mean(layer_delta_adapt[n]) for n in layer_names]
    mean_dw_super = [np.mean(layer_delta_super[n]) for n in layer_names]
    rewiring_ratio = [s / max(a, 1e-8) for a, s in zip(mean_dw_adapt, mean_dw_super)]

    print(f"\n  --- Rewiring Analysis ---")
    print(f"  {'Layer':<30} | {'dW Adapt':>10} | {'dW Super':>10} | {'Ratio':>6}")
    print("  " + "-" * 65)
    for i, name in enumerate(layer_names):
        short = name.replace('.', '_')[:28]
        print(f"  {short:<30} | {mean_dw_adapt[i]:>10.4f} | {mean_dw_super[i]:>10.4f} | {rewiring_ratio[i]:>5.1f}x")

    # 2. Sparsity analysis
    mean_sparsity_pre = np.mean(sparsity_before)
    mean_sparsity_post = np.mean(sparsity_after)
    print(f"\n  Sparsity after ablation:  {mean_sparsity_pre:.1%}")
    print(f"  Sparsity after recovery:  {mean_sparsity_post:.1%}")
    print(f"  Neurons 're-activated':   {mean_sparsity_pre - mean_sparsity_post:.1%}")

    # 3. Attention shift analysis
    mean_kl_adapt = np.mean([a['kl_adapt'] for a in attn_shifts])
    mean_kl_super = np.mean([a['kl_super'] for a in attn_shifts])
    print(f"\n  Attention KL divergence (from base):")
    print(f"    Adapt-only: {mean_kl_adapt:.4f}")
    print(f"    Super-comp: {mean_kl_super:.4f}")
    print(f"    Ratio:      {mean_kl_super / max(mean_kl_adapt, 1e-8):.1f}x more re-wiring")

    elapsed = time.time() - t0
    results = {
        'adapt_rate': adapt_rate, 'super_rate': super_rate,
        'mean_dw_adapt': {n: float(v) for n, v in zip(layer_names, mean_dw_adapt)},
        'mean_dw_super': {n: float(v) for n, v in zip(layer_names, mean_dw_super)},
        'rewiring_ratio': {n: float(v) for n, v in zip(layer_names, rewiring_ratio)},
        'sparsity_pre': mean_sparsity_pre, 'sparsity_post': mean_sparsity_post,
        'kl_adapt': mean_kl_adapt, 'kl_super': mean_kl_super,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase43_anatomy.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ===== Visualization =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: ΔW per layer comparison
    ax = axes[0, 0]
    x = np.arange(len(layer_names))
    w = 0.35
    short_names = [n.split('.')[-1][:10] for n in layer_names]
    ax.bar(x - w/2, mean_dw_adapt, w, label='Adapt-only', color='#FF9800', alpha=0.8)
    ax.bar(x + w/2, mean_dw_super, w, label='Super-comp', color='#2196F3', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean |dW|'); ax.set_title('Weight Change per Layer')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 2: Rewiring ratio
    ax = axes[0, 1]
    colors = ['#4CAF50' if r > 1.5 else '#FF9800' if r > 1.0 else '#F44336' for r in rewiring_ratio]
    ax.bar(range(len(layer_names)), rewiring_ratio, color=colors, alpha=0.8)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('dW_super / dW_adapt'); ax.set_title('Rewiring Ratio (>1 = more change)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Sparsity before/after recovery
    ax = axes[1, 0]
    ax.bar(['After Ablation', 'After Recovery'], [mean_sparsity_pre, mean_sparsity_post],
           color=['#F44336', '#4CAF50'], alpha=0.8)
    ax.set_ylabel('Sparsity (% zero weights)')
    ax.set_title('Neuron Re-activation During Recovery')
    for i, v in enumerate([mean_sparsity_pre, mean_sparsity_post]):
        ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=14, fontweight='bold')

    # Plot 4: Attention shift examples (first 5 tasks)
    ax = axes[1, 1]
    for i, shift in enumerate(attn_shifts[:5]):
        n_obj = len(shift['attn_base'])
        x_pos = np.arange(n_obj) + i * (n_obj + 1)
        ax.bar(x_pos - 0.2, shift['attn_base'], 0.4, alpha=0.5, color='#9E9E9E', label='Base' if i == 0 else '')
        ax.bar(x_pos + 0.2, shift['attn_super'], 0.4, alpha=0.7, color='#2196F3', label='Super' if i == 0 else '')
    ax.set_xlabel('Object Index (per task)'); ax.set_ylabel('Attention Weight')
    ax.set_title('Pointer Attention Shift: Base vs Super-Compensated')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase43_anatomy.png'), dpi=150)
    plt.close()
    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__': main()
