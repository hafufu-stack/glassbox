"""
Phase 35: Attribution-Guided Targeted Surgery
================================================
Phase 33 proved 82.8% of predictions are fully traceable.
This means we KNOW where the model is confused (high entropy nodes).

Phase 35 exploits this: instead of adapting the entire model,
we surgically update ONLY the confused modules, leaving confident
paths untouched. True white-box self-debugging.

Method:
  1. Forward pass with causal tracing
  2. Identify high-entropy decision points (confused modules)
  3. Freeze confident modules, update only confused ones
  4. Compare: full-model adapt vs surgical adapt
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
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=8;N_COLORS=10;HID=64

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

class SurgicalAgent(nn.Module):
    """Agent with module-level confidence tracking for surgical updates."""
    def __init__(s, hid=HID):
        super().__init__()
        s.ne = nn.Linear(NODE_FEAT_DIM, hid)
        s.g1 = nn.Sequential(nn.Linear(hid*2, hid), nn.ReLU(), nn.Linear(hid, hid))
        s.g2 = nn.Sequential(nn.Linear(hid*2, hid), nn.ReLU(), nn.Linear(hid, hid))
        s.n1 = nn.LayerNorm(hid); s.n2 = nn.LayerNorm(hid)
        s.oh = nn.Linear(hid, N_OPS)
        s.c1h = nn.Linear(hid, N_COLORS)
        s.c2h = nn.Linear(hid, N_COLORS)
        s.pq = nn.Linear(hid, hid); s.pk = nn.Linear(hid, hid)

    def forward(s, nf, nn_c):
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1); h = s.ne(nf)
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + s.g1(torch.cat([h, msg.expand_as(h)], -1)); h = s.n1(h) * mf
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + s.g2(torch.cat([h, msg.expand_as(h)], -1)); h = s.n2(h) * mf
        g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)
        pl = ((s.pq(g).unsqueeze(1)) * s.pk(h)).sum(-1).masked_fill(~mask, -1e9)
        return s.oh(g), s.c1h(g), s.c2h(g), pl

    def diagnose(s, nf, nn_c):
        """Diagnose which output heads are confused (high entropy)."""
        with torch.no_grad():
            ol, cl1, cl2, pl = s.forward(nf, nn_c)
            n_obj = nn_c.item()

            # Compute entropy for each output head
            def entropy(logits):
                p = F.softmax(logits, dim=-1)
                return -(p * (p + 1e-8).log()).sum(-1).item()

            diagnosis = {
                'op_entropy': entropy(ol),
                'c1_entropy': entropy(cl1),
                'c2_entropy': entropy(cl2),
                'ptr_entropy': entropy(pl[..., :max(n_obj, 1)]) if n_obj > 0 else 0.0,
                'op_conf': F.softmax(ol, -1).max().item(),
                'c1_conf': F.softmax(cl1, -1).max().item(),
                'c2_conf': F.softmax(cl2, -1).max().item(),
                'ptr_conf': F.softmax(pl, -1)[0, :max(n_obj, 1)].max().item() if n_obj > 0 else 0.0,
            }
            # Identify which heads are confused (entropy > threshold)
            confused = []
            if diagnosis['op_entropy'] > 1.0: confused.append('op')
            if diagnosis['c1_entropy'] > 1.0: confused.append('c1')
            if diagnosis['c2_entropy'] > 1.0: confused.append('c2')
            if diagnosis['ptr_entropy'] > 1.0: confused.append('ptr')
            diagnosis['confused_heads'] = confused
            return diagnosis

    def get_head_params(s, head_name):
        """Get parameters for a specific output head."""
        if head_name == 'op': return list(s.oh.parameters())
        elif head_name == 'c1': return list(s.c1h.parameters())
        elif head_name == 'c2': return list(s.c2h.parameters())
        elif head_name == 'ptr': return list(s.pq.parameters()) + list(s.pk.parameters())
        return []


def compute_loss(model, s):
    ol, cl1, cl2, pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
    return (F.cross_entropy(ol, s['op'].to(DEVICE)) + F.cross_entropy(cl1, s['c1'].to(DEVICE)) +
            F.cross_entropy(cl2, s['c2'].to(DEVICE)) + F.cross_entropy(pl, s['ptr'].to(DEVICE)))


def adapt_full(model, task_samples, steps=50):
    """Standard full-model adaptation (Phase 28 baseline)."""
    if not task_samples: return model
    am = copy.deepcopy(model)
    opt = torch.optim.SGD(am.parameters(), lr=1e-2)
    am.train()
    for _ in range(steps):
        batch = random.sample(task_samples, min(8, len(task_samples)))
        tl = sum(compute_loss(am, d) for d in batch) / len(batch)
        opt.zero_grad(); tl.backward()
        torch.nn.utils.clip_grad_norm_(am.parameters(), 1.0); opt.step()
    am.eval(); return am


def adapt_surgical(model, task_samples, steps=50):
    """Surgical adaptation: only update confused heads.
    The core GlassBox Phase 35 innovation."""
    if not task_samples: return model
    am = copy.deepcopy(model)

    # Diagnose: which heads are confused on this task's demos?
    confused_counts = {'op': 0, 'c1': 0, 'c2': 0, 'ptr': 0}
    for s in task_samples[:4]:
        diag = am.diagnose(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        for h in diag['confused_heads']:
            confused_counts[h] += 1

    # If no heads are confused, do minimal adaptation (the model is confident)
    confused = [h for h, c in confused_counts.items() if c > 0]
    if not confused:
        # Model is confident -> only 5 steps of light adaptation
        return adapt_full(model, task_samples, steps=5)

    # Freeze all parameters first
    for p in am.parameters():
        p.requires_grad = False

    # Unfreeze ONLY confused heads + shared GNN backbone
    # (backbone must be unfrozen for gradient flow)
    for p in am.ne.parameters(): p.requires_grad = True
    for p in am.g1.parameters(): p.requires_grad = True
    for p in am.g2.parameters(): p.requires_grad = True
    for p in am.n1.parameters(): p.requires_grad = True
    for p in am.n2.parameters(): p.requires_grad = True

    for head in confused:
        for p in am.get_head_params(head):
            p.requires_grad = True

    # Higher learning rate for confused heads (they need more correction)
    param_groups = [
        {'params': [p for n, p in am.named_parameters()
                    if p.requires_grad and not any(h in n for h in ['oh', 'c1h', 'c2h', 'pq', 'pk'])],
         'lr': 5e-3},  # backbone: gentle
        {'params': [p for head in confused for p in am.get_head_params(head)],
         'lr': 2e-2},  # confused heads: aggressive
    ]
    # Remove empty groups
    param_groups = [g for g in param_groups if g['params']]

    opt = torch.optim.SGD(param_groups)
    am.train()
    for _ in range(steps):
        batch = random.sample(task_samples, min(8, len(task_samples)))
        tl = sum(compute_loss(am, d) for d in batch) / len(batch)
        opt.zero_grad(); tl.backward()
        torch.nn.utils.clip_grad_norm_([p for p in am.parameters() if p.requires_grad], 1.0)
        opt.step()
    am.eval(); return am


def adapt_entropy_weighted(model, task_samples, steps=50):
    """Entropy-weighted adaptation: weight loss terms by head confusion.
    Confused heads get higher loss weight, forcing more focus on them."""
    if not task_samples: return model
    am = copy.deepcopy(model)

    # Diagnose to get entropy weights
    entropies = {'op': [], 'c1': [], 'c2': [], 'ptr': []}
    for s in task_samples[:4]:
        diag = am.diagnose(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        entropies['op'].append(diag['op_entropy'])
        entropies['c1'].append(diag['c1_entropy'])
        entropies['c2'].append(diag['c2_entropy'])
        entropies['ptr'].append(diag['ptr_entropy'])

    # Higher entropy = higher loss weight (more confused = more attention)
    weights = {}
    for k, vals in entropies.items():
        avg_e = np.mean(vals) if vals else 0.0
        weights[k] = max(0.5, min(3.0, avg_e))  # clamp to [0.5, 3.0]

    opt = torch.optim.SGD(am.parameters(), lr=1e-2)
    am.train()
    for _ in range(steps):
        batch = random.sample(task_samples, min(8, len(task_samples)))
        tl = torch.tensor(0.0, device=DEVICE)
        for d in batch:
            ol, cl1, cl2, pl = am(d['nf'].to(DEVICE), d['nn'].to(DEVICE))
            tl = tl + (
                weights['op'] * F.cross_entropy(ol, d['op'].to(DEVICE)) +
                weights['c1'] * F.cross_entropy(cl1, d['c1'].to(DEVICE)) +
                weights['c2'] * F.cross_entropy(cl2, d['c2'].to(DEVICE)) +
                weights['ptr'] * F.cross_entropy(pl, d['ptr'].to(DEVICE))
            )
        tl = tl / len(batch)
        opt.zero_grad(); tl.backward()
        torch.nn.utils.clip_grad_norm_(am.parameters(), 1.0); opt.step()
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
    print("Phase 35: Attribution-Guided Targeted Surgery")
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

    model = SurgicalAgent().to(DEVICE)
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

    # Compare adaptation methods
    methods = {
        'full_adapt': ('Full-Model Adapt (P28)', adapt_full),
        'surgical': ('Surgical Adapt (P35)', adapt_surgical),
        'entropy_weighted': ('Entropy-Weighted (P35b)', adapt_entropy_weighted),
    }

    results = {}
    print(f"\n{'Method':<35} | {'Full Match':>10} | {'vs Full-Adapt':>13} | {'Avg Confused Heads':>18}")
    print("-" * 90)

    # Collect diagnosis statistics
    all_diagnoses = []
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        if aug_samples:
            diag = model.diagnose(aug_samples[0]['nf'].to(DEVICE), aug_samples[0]['nn'].to(DEVICE))
            all_diagnoses.append(diag)

    avg_confused = np.mean([len(d['confused_heads']) for d in all_diagnoses]) if all_diagnoses else 0

    for method_key, (label, adapt_fn) in methods.items():
        ok_total, n_total = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_samples = [s for s in aug_samples if s]

            adapted = adapt_fn(model, aug_samples, steps=50)
            ok, tot = eval_task(adapted, task)
            ok_total += ok; n_total += tot

        match_rate = ok_total / max(n_total, 1)
        results[method_key] = match_rate

        if method_key == 'full_adapt':
            base_rate = match_rate
            print(f"  {label:<33} | {match_rate:>9.1%} | {'baseline':>13} | {avg_confused:>17.1f}")
        else:
            delta = match_rate - base_rate
            print(f"  {label:<33} | {match_rate:>9.1%} | {delta:>+12.1%} | {avg_confused:>17.1f}")

    # Diagnosis statistics
    print(f"\n  --- Diagnosis Statistics ---")
    print(f"  Tasks with confused op head: {sum(1 for d in all_diagnoses if 'op' in d['confused_heads'])}/{len(all_diagnoses)}")
    print(f"  Tasks with confused c1 head: {sum(1 for d in all_diagnoses if 'c1' in d['confused_heads'])}/{len(all_diagnoses)}")
    print(f"  Tasks with confused c2 head: {sum(1 for d in all_diagnoses if 'c2' in d['confused_heads'])}/{len(all_diagnoses)}")
    print(f"  Tasks with confused ptr head: {sum(1 for d in all_diagnoses if 'ptr' in d['confused_heads'])}/{len(all_diagnoses)}")
    print(f"  Tasks with NO confusion: {sum(1 for d in all_diagnoses if not d['confused_heads'])}/{len(all_diagnoses)}")

    elapsed = time.time() - t0
    results['diagnosis'] = {
        'avg_confused_heads': avg_confused,
        'op_confused_pct': sum(1 for d in all_diagnoses if 'op' in d['confused_heads']) / max(len(all_diagnoses), 1),
        'c1_confused_pct': sum(1 for d in all_diagnoses if 'c1' in d['confused_heads']) / max(len(all_diagnoses), 1),
        'c2_confused_pct': sum(1 for d in all_diagnoses if 'c2' in d['confused_heads']) / max(len(all_diagnoses), 1),
        'ptr_confused_pct': sum(1 for d in all_diagnoses if 'ptr' in d['confused_heads']) / max(len(all_diagnoses), 1),
    }
    results['elapsed'] = elapsed
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase35_surgery.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Method comparison
    ax = axes[0]
    names = list(methods.keys())
    vals = [results[k] for k in names]
    colors_bar = ['#FF9800', '#4CAF50', '#2196F3']
    ax.bar(range(len(names)), vals, color=colors_bar, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(['Full Adapt\n(P28)', 'Surgical\n(P35)', 'Entropy-\nWeighted'], fontsize=10)
    ax.set_ylabel('Full Match'); ax.set_title('Adaptation Method Comparison')
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12, fontweight='bold')

    # Plot 2: Confusion diagnosis
    ax = axes[1]
    head_names = ['Operation', 'Color 1', 'Color 2', 'Pointer']
    confused_pcts = [
        results['diagnosis']['op_confused_pct'],
        results['diagnosis']['c1_confused_pct'],
        results['diagnosis']['c2_confused_pct'],
        results['diagnosis']['ptr_confused_pct'],
    ]
    ax.bar(range(4), [p * 100 for p in confused_pcts], color=['#E91E63', '#9C27B0', '#673AB7', '#3F51B5'])
    ax.set_xticks(range(4)); ax.set_xticklabels(head_names, fontsize=10)
    ax.set_ylabel('% Tasks Confused'); ax.set_title('Where Does the Model Struggle?')
    ax.set_ylim(0, 100)

    # Plot 3: Entropy distribution
    ax = axes[2]
    if all_diagnoses:
        op_ents = [d['op_entropy'] for d in all_diagnoses]
        ptr_ents = [d['ptr_entropy'] for d in all_diagnoses]
        ax.hist(op_ents, bins=20, alpha=0.6, label='Op Head', color='#E91E63')
        ax.hist(ptr_ents, bins=20, alpha=0.6, label='Ptr Head', color='#3F51B5')
        ax.axvline(x=1.0, color='red', linestyle='--', label='Confusion Threshold')
        ax.set_xlabel('Entropy'); ax.set_ylabel('Count')
        ax.set_title('Entropy Distribution per Head'); ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase35_surgery.png'), dpi=150)
    plt.close()
    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__': main()
