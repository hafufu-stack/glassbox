"""
Phase 63: Cross-Demo Consistency Loss
=======================================
ARC's essence: find ONE universal rule across ALL demos.
Current adaptation may overfit to individual demos.

Add Consistency Loss: force the model to predict the SAME
operation distribution across all demo pairs in a task.

Design: Compare standard TTT vs TTT + Consistency Loss.
3 seeds × 2 methods (standard, consistency).
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

def consistency_loss(model, samples):
    """Compute variance of Op distribution across samples.
    Low variance = model predicts same operation for all demos = consistent.
    """
    if len(samples) < 2:
        return torch.tensor(0.0, device=DEVICE)

    op_dists = []
    for s in samples[:8]:
        ol, _, _, _ = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        op_prob = F.softmax(ol, dim=-1)
        op_dists.append(op_prob)

    if len(op_dists) < 2:
        return torch.tensor(0.0, device=DEVICE)

    # Stack and compute variance across demos
    stacked = torch.cat(op_dists, dim=0)  # (N, N_OPS)
    mean_dist = stacked.mean(dim=0, keepdim=True)  # (1, N_OPS)
    variance = ((stacked - mean_dist) ** 2).mean()
    return variance

def adapt_standard(model, task_samples, steps=100, lr=1e-2):
    """Standard TTT (no consistency)."""
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

def adapt_consistent(model, task_samples, raw_samples, steps=100, lr=1e-2, alpha=0.5):
    """TTT with Consistency Loss: force same Op across demos."""
    if not task_samples: return model
    am = copy.deepcopy(model)
    opt = torch.optim.SGD(am.parameters(), lr=lr)
    am.train()
    for _ in range(steps):
        batch = random.sample(task_samples, min(8, len(task_samples)))
        tl = sum(compute_loss(am, d) for d in batch) / len(batch)
        # Add consistency loss using raw (non-augmented) samples
        cl = consistency_loss(am, raw_samples)
        total = tl + alpha * cl
        opt.zero_grad(); total.backward()
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
    print("Phase 63: Cross-Demo Consistency Loss")
    print("Force universal rule extraction across demos")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    N_SEEDS = 3
    alphas = [0.0, 0.1, 0.3, 0.5, 1.0]  # 0.0 = standard (no consistency)
    all_results = {f'a{a}': [] for a in alphas}

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

        for alpha in alphas:
            ok_total, n_total = 0, 0
            for task in test_tasks:
                demos = task.get('train', [])
                # Raw samples (non-augmented) for consistency loss
                raw_samples = [prep(p['input'], p['output']) for p in demos]
                raw_samples = [s for s in raw_samples if s]

                aug_pairs = []
                for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
                aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                aug_samples = [s for s in aug_samples if s]

                current = ablate_least_important(model, 0.15, aug_samples)
                if alpha == 0.0:
                    current = adapt_standard(current, aug_samples, steps=100)
                else:
                    current = adapt_consistent(current, aug_samples, raw_samples, steps=100, alpha=alpha)

                ok, tot = eval_task(current, task)
                ok_total += ok; n_total += tot

            rate = ok_total / max(n_total, 1)
            all_results[f'a{alpha}'].append(rate)
            label = f"α={alpha}" if alpha > 0 else "Standard"
            print(f"    {label:<14}: {rate:.1%}")

    # Summary
    results = {}
    print(f"\n  === CONSISTENCY LOSS SWEEP ({N_SEEDS} seeds) ===")
    print(f"  {'Method':<14} | {'Mean':>6} | {'Std':>5}")
    print("  " + "-" * 35)

    best_mean, best_alpha = 0, 0
    for alpha in alphas:
        key = f'a{alpha}'
        vals = np.array(all_results[key])
        mean, std = float(vals.mean()), float(vals.std())
        results[f'{key}_mean'] = mean
        results[f'{key}_std'] = std
        results[f'{key}_all'] = [float(v) for v in vals]
        label = f"α={alpha}" if alpha > 0 else "Standard"
        print(f"  {label:<14} | {mean:>5.1%} | {std:>4.1%}")
        if mean > best_mean:
            best_mean = mean; best_alpha = alpha

    results['best_alpha'] = best_alpha
    results['best_mean'] = best_mean
    results['n_seeds'] = N_SEEDS
    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase63_consistency.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    alpha_labels = ['Standard\n(α=0)'] + [f'α={a}' for a in alphas[1:]]
    means = [results[f'a{a}_mean'] for a in alphas]
    stds = [results[f'a{a}_std'] for a in alphas]
    colors = ['#9E9E9E'] + ['#2196F3']*(len(alphas)-1)
    colors[alphas.index(best_alpha)] = '#E91E63'
    bars = ax.bar(range(len(alphas)), means, yerr=stds, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(alphas))); ax.set_xticklabels(alpha_labels)
    ax.set_ylabel('Full Match'); ax.set_title('Phase 63: Consistency Loss Sweep')
    ax.set_ylim(0.7, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m+0.02, f'{m:.1%}', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase63_consistency.png'), dpi=150)
    plt.close()

    print(f"\n  *** BEST: α={best_alpha}, Mean = {best_mean:.1%} ***")
    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
