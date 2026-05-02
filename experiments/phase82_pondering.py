"""
Phase 82: Dynamic Latent Pondering
====================================
Multi-seed validation of P79 Latent Graph Dynamics +
Adaptive Computation Time (PonderNet-style halt probability).

The model learns to "think longer" on hard tasks and "answer quickly"
on easy ones. Each latent step outputs a halt probability; the model
stops when cumulative halt > threshold.

5 seeds x {fixed 5-step, adaptive pondering} comparison.
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
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=8;N_COLORS=10;MAX_PONDER_STEPS=10

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


class PonderingAgent(nn.Module):
    """GNN with PonderNet-style adaptive computation in latent space."""
    def __init__(s, hid=64):
        super().__init__()
        s.hid = hid
        s.ne = nn.Linear(NODE_FEAT_DIM, hid)
        s.g1 = nn.Sequential(nn.Linear(hid*2, hid), nn.ReLU(), nn.Linear(hid, hid))
        s.g2 = nn.Sequential(nn.Linear(hid*2, hid), nn.ReLU(), nn.Linear(hid, hid))
        s.n1 = nn.LayerNorm(hid); s.n2 = nn.LayerNorm(hid)
        # Latent transition
        s.transition = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid), nn.Tanh()
        )
        # Halt probability predictor (PonderNet)
        s.halt_head = nn.Sequential(
            nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        # Readout heads
        s.oh = nn.Linear(hid, N_OPS); s.c1h = nn.Linear(hid, N_COLORS)
        s.c2h = nn.Linear(hid, N_COLORS)
        s.pq = nn.Linear(hid, hid); s.pk = nn.Linear(hid, hid)

    def encode(s, nf, nn_c):
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        h = s.ne(nf)
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + s.g1(torch.cat([h, msg.expand_as(h)], -1)); h = s.n1(h) * mf
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + s.g2(torch.cat([h, msg.expand_as(h)], -1)); h = s.n2(h) * mf
        return h, mask, mf

    def decode(s, h, mask, mf):
        g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)
        pl = ((s.pq(g).unsqueeze(1)) * s.pk(h)).sum(-1).masked_fill(~mask, -1e9)
        return s.oh(g), s.c1h(g), s.c2h(g), pl

    def forward_fixed(s, nf, nn_c, n_steps=5):
        """Fixed-step latent dynamics (P79 style)."""
        h, mask, mf = s.encode(nf, nn_c)
        for _ in range(n_steps):
            delta = s.transition(h) * 0.1
            h = h + delta * mf
        return s.decode(h, mask, mf)

    def forward_pondering(s, nf, nn_c, max_steps=MAX_PONDER_STEPS):
        """PonderNet: adaptive steps with halt probability."""
        h, mask, mf = s.encode(nf, nn_c)
        # Collect outputs and halt probs at each step
        all_outputs = []
        halt_probs = []
        for step in range(max_steps):
            delta = s.transition(h) * 0.1
            h = h + delta * mf
            outputs = s.decode(h, mask, mf)
            all_outputs.append(outputs)
            # Halt probability from global pooled state
            g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)
            p_halt = s.halt_head(g).squeeze(-1)  # (B,)
            halt_probs.append(p_halt)
        return all_outputs, halt_probs

    def forward_adaptive_inference(s, nf, nn_c, threshold=0.8, max_steps=MAX_PONDER_STEPS):
        """At inference: stop when cumulative halt > threshold."""
        h, mask, mf = s.encode(nf, nn_c)
        cum_halt = torch.zeros(nf.size(0), device=nf.device)
        n_steps_used = 0
        for step in range(max_steps):
            delta = s.transition(h) * 0.1
            h = h + delta * mf
            g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)
            p_halt = s.halt_head(g).squeeze(-1)
            cum_halt = cum_halt + p_halt
            n_steps_used = step + 1
            if (cum_halt >= threshold).all():
                break
        return s.decode(h, mask, mf), n_steps_used


def compute_loss_fixed(model, s, n_steps=5):
    ol,cl1,cl2,pl = model.forward_fixed(s['nf'].to(DEVICE), s['nn'].to(DEVICE), n_steps=n_steps)
    return (F.cross_entropy(ol, s['op'].to(DEVICE)) + F.cross_entropy(cl1, s['c1'].to(DEVICE)) +
            F.cross_entropy(cl2, s['c2'].to(DEVICE)) + F.cross_entropy(pl, s['ptr'].to(DEVICE)))

def compute_loss_pondering(model, s):
    """PonderNet loss: weighted sum of per-step losses + KL regularizer."""
    all_out, halt_probs = model.forward_pondering(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
    N = len(all_out)
    # Geometric prior for regularization (encourage using fewer steps)
    lambda_p = 0.5
    geo_prior = [(1-lambda_p)**(i) * lambda_p for i in range(N-1)] + [(1-lambda_p)**(N-1)]
    geo_prior = torch.tensor(geo_prior, device=DEVICE)
    geo_prior = geo_prior / geo_prior.sum()

    # Compute halting distribution
    p_continue = torch.ones(1, device=DEVICE)
    halt_dist = []
    for i, p_h in enumerate(halt_probs):
        p_h_mean = p_h.mean()  # Average over batch
        if i < N - 1:
            halt_dist.append(p_continue * p_h_mean)
            p_continue = p_continue * (1 - p_h_mean)
        else:
            halt_dist.append(p_continue)  # Must halt at last step
    halt_dist = torch.stack(halt_dist)
    halt_dist = halt_dist / halt_dist.sum().clamp(min=1e-8)

    # Weighted task loss
    total_loss = torch.tensor(0.0, device=DEVICE)
    for i, (ol,cl1,cl2,pl) in enumerate(all_out):
        step_loss = (F.cross_entropy(ol, s['op'].to(DEVICE)) +
                     F.cross_entropy(cl1, s['c1'].to(DEVICE)) +
                     F.cross_entropy(cl2, s['c2'].to(DEVICE)) +
                     F.cross_entropy(pl, s['ptr'].to(DEVICE)))
        total_loss = total_loss + halt_dist[i].detach() * step_loss

    # KL divergence between halt distribution and geometric prior
    kl = F.kl_div(halt_dist.log().clamp(min=-20), geo_prior, reduction='sum')
    return total_loss + 0.01 * kl

def adapt_model(model, task_samples, steps=100, lr=0.1, mode='fixed', n_steps=5):
    if not task_samples: return model
    am = copy.deepcopy(model); opt = torch.optim.SGD(am.parameters(), lr=lr); am.train()
    for _ in range(steps):
        batch = random.sample(task_samples, min(8, len(task_samples)))
        if mode == 'fixed':
            tl = sum(compute_loss_fixed(am, d, n_steps=n_steps) for d in batch) / len(batch)
        else:
            tl = sum(compute_loss_pondering(am, d) for d in batch) / len(batch)
        opt.zero_grad(); tl.backward()
        torch.nn.utils.clip_grad_norm_(am.parameters(), 1.0); opt.step()
    am.eval(); return am

def eval_task(model, task, mode='fixed', n_steps=5):
    ok, total, steps_used = 0, 0, []
    for tp in task.get('test', []):
        s = prep(tp['input'], tp['output'])
        if s is None: continue
        total += 1
        with torch.no_grad():
            if mode == 'fixed':
                ol,cl1,cl2,pl = model.forward_fixed(s['nf'].to(DEVICE), s['nn'].to(DEVICE), n_steps=n_steps)
                steps_used.append(n_steps)
            else:
                (ol,cl1,cl2,pl), ns = model.forward_adaptive_inference(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
                steps_used.append(ns)
        if (ol.argmax(1).item() == s['op'].item() and cl1.argmax(1).item() == s['c1'].item() and
            cl2.argmax(1).item() == s['c2'].item() and pl.argmax(1).item() == s['ptr'].item()):
            ok += 1
    return ok, total, steps_used

def load_arc_tasks(d, n=400):
    t = []
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                t.append({'id': f[:-5], **json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 82: Dynamic Latent Pondering")
    print("Multi-seed P79 validation + PonderNet adaptive computation")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    SEEDS = [42, 123, 456, 789, 1024]
    results_fixed = []
    results_ponder = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  === Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        all_samples = []
        for task in tasks:
            for p in task.get('train', []):
                s = prep(p['input'], p['output'])
                if s: all_samples.append(s)

        # Train model with pondering capability
        model = PonderingAgent().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(80):
            model.train(); random.shuffle(all_samples)
            for s in all_samples:
                # Alternate between fixed and pondering loss
                if ep < 40:
                    loss = compute_loss_fixed(model, s, n_steps=5)
                else:
                    loss = compute_loss_pondering(model, s)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

        # Evaluate: Fixed 5-step
        ok_f, n_f = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_s = [s for s in aug_s if s]
            adapted = adapt_model(model, aug_s, steps=100, lr=0.1, mode='fixed', n_steps=5)
            ok, tot, _ = eval_task(adapted, task, mode='fixed', n_steps=5)
            ok_f += ok; n_f += tot
        acc_f = ok_f / max(n_f, 1)
        results_fixed.append(acc_f)
        print(f"    Fixed 5-step: {acc_f:.1%}")

        # Evaluate: Adaptive pondering
        ok_p, n_p = 0, 0
        all_steps = []
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_s = [s for s in aug_s if s]
            adapted = adapt_model(model, aug_s, steps=100, lr=0.1, mode='ponder')
            ok, tot, steps = eval_task(adapted, task, mode='ponder')
            ok_p += ok; n_p += tot; all_steps.extend(steps)
        acc_p = ok_p / max(n_p, 1)
        results_ponder.append(acc_p)
        avg_steps = np.mean(all_steps) if all_steps else 0
        print(f"    Pondering:    {acc_p:.1%} (avg {avg_steps:.1f} steps)")

    # Statistics
    fixed_mean = np.mean(results_fixed); fixed_std = np.std(results_fixed)
    ponder_mean = np.mean(results_ponder); ponder_std = np.std(results_ponder)

    print(f"\n  === Multi-Seed Summary ===")
    print(f"  Fixed 5-step:  {fixed_mean:.1%} +/- {fixed_std:.1%}")
    print(f"  Pondering:     {ponder_mean:.1%} +/- {ponder_std:.1%}")

    results = {
        'seeds': SEEDS,
        'fixed_accs': results_fixed, 'fixed_mean': fixed_mean, 'fixed_std': fixed_std,
        'ponder_accs': results_ponder, 'ponder_mean': ponder_mean, 'ponder_std': ponder_std,
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase82_pondering.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    x = np.arange(len(SEEDS))
    ax.bar(x-0.2, results_fixed, 0.35, label='Fixed 5-step', color='#1565C0', alpha=0.85)
    ax.bar(x+0.2, results_ponder, 0.35, label='Adaptive Pondering', color='#E91E63', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f'Seed {s}' for s in SEEDS], fontsize=8)
    ax.set_ylabel('Accuracy'); ax.set_title('Per-Seed Comparison')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0.7, 1.0)

    ax = axes[1]
    ax.bar(['Fixed 5-step', 'Pondering'], [fixed_mean, ponder_mean],
           yerr=[fixed_std, ponder_std], color=['#1565C0', '#E91E63'],
           alpha=0.85, capsize=5, edgecolor='white', linewidth=2)
    ax.set_ylabel('Accuracy'); ax.set_title('Mean +/- Std')
    ax.set_ylim(0.7, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate([(fixed_mean, fixed_std), (ponder_mean, ponder_std)]):
        ax.text(i, m + s + 0.01, f'{m:.1%}\n+/-{s:.1%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 82: Dynamic Latent Pondering', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase82_pondering.png'), dpi=150); plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
