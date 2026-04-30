"""
Phase 62: GlassBox-Guided A* Search (Neuro-Symbolic Fusion)
============================================================
Phase 20 showed brute-force DSL search covers only 2.5% of tasks.
Now use GlassBox's softmax probabilities as A* heuristics to
prune the search space dramatically.

Key idea: GlassBox predicts (Op, Color1, Color2, Pointer) with
confidence scores. Use these as beam search priorities instead
of blind enumeration.

Design: Compare 3 approaches on test tasks:
  1. Blind DSL search (Phase 20 style)
  2. GlassBox neural-only (Phase 59 style)
  3. GlassBox-Guided search (neural heuristic + DSL verification)
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

# ============================================================
# DSL Operations (simplified from Phase 1/20)
# ============================================================
def dsl_identity(grid):
    return [row[:] for row in grid]

def dsl_fill(grid, color):
    h, w = len(grid), len(grid[0])
    return [[color]*w for _ in range(h)]

def dsl_recolor(grid, old_c, new_c):
    return [[new_c if v == old_c else v for v in row] for row in grid]

def dsl_flipv(grid):
    return grid[::-1]

def dsl_fliph(grid):
    return [row[::-1] for row in grid]

def dsl_rot90(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[h-1-r][c] for r in range(h)] for c in range(w)]

def dsl_rot180(grid):
    return [row[::-1] for row in grid[::-1]]

DSL_PROGRAMS = []
# Single-step programs
DSL_PROGRAMS.append(('identity', dsl_identity, []))
DSL_PROGRAMS.append(('flipv', dsl_flipv, []))
DSL_PROGRAMS.append(('fliph', dsl_fliph, []))
DSL_PROGRAMS.append(('rot90', dsl_rot90, []))
DSL_PROGRAMS.append(('rot180', dsl_rot180, []))
for c in range(10):
    DSL_PROGRAMS.append((f'fill_{c}', dsl_fill, [c]))
for old_c in range(10):
    for new_c in range(10):
        if old_c != new_c:
            DSL_PROGRAMS.append((f'recolor_{old_c}_{new_c}', dsl_recolor, [old_c, new_c]))

def run_dsl_program(grid, prog):
    name, func, args = prog
    try:
        return func(grid, *args)
    except Exception:
        return None

def blind_dsl_search(inp, out, max_programs=None):
    """Blind search: try all programs."""
    programs = DSL_PROGRAMS if max_programs is None else DSL_PROGRAMS[:max_programs]
    for prog in programs:
        result = run_dsl_program(inp, prog)
        if result is not None and np.array_equal(np.array(result), np.array(out)):
            return prog[0]
    return None

def guided_dsl_search(inp, out, model, top_k=20):
    """GlassBox-guided search: use neural confidence to rank programs."""
    s = prep(inp, out)
    if s is None:
        return blind_dsl_search(inp, out)

    with torch.no_grad():
        ol, cl1, cl2, pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        op_probs = F.softmax(ol, dim=-1).cpu().numpy()[0]
        c1_probs = F.softmax(cl1, dim=-1).cpu().numpy()[0]
        c2_probs = F.softmax(cl2, dim=-1).cpu().numpy()[0]

    # Score each DSL program by neural confidence
    scored = []
    for prog in DSL_PROGRAMS:
        name, func, args = prog
        score = 0.0
        # Map DSL to op type
        if 'identity' in name:
            score = op_probs[1]  # op=1
        elif 'fill' in name:
            c = args[0]
            score = op_probs[2] * c1_probs[min(c, 9)]
        elif 'recolor' in name:
            oc, nc = args
            score = op_probs[5] * c1_probs[min(oc, 9)] * c2_probs[min(nc, 9)]
        elif 'flipv' in name:
            score = op_probs[6]
        elif 'fliph' in name:
            score = op_probs[7]
        elif 'rot' in name:
            score = op_probs[3]  # generic transform
        else:
            score = 0.01

        scored.append((score, prog))

    # Sort by neural confidence (highest first)
    scored.sort(key=lambda x: -x[0])

    # Try top-K programs
    for score, prog in scored[:top_k]:
        result = run_dsl_program(inp, prog)
        if result is not None and np.array_equal(np.array(result), np.array(out)):
            return prog[0]
    return None


def eval_task_neural(model, task):
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
    print("Phase 62: GlassBox-Guided A* Search")
    print("Neural heuristics + DSL verification")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    N_SEEDS = 3
    results_per_seed = {'blind': [], 'neural': [], 'guided': [], 'hybrid': []}

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

        # Evaluate 3 approaches
        blind_ok = 0; neural_ok = 0; guided_ok = 0; hybrid_ok = 0
        total = 0

        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_samples = [s for s in aug_samples if s]

            # Adapted model for neural/guided
            adapted = ablate_least_important(model, 0.15, aug_samples)
            adapted = adapt_model(adapted, aug_samples, steps=100)

            for tp in task.get('test', []):
                total += 1
                inp, out = tp['input'], tp['output']

                # 1. Blind DSL search
                if blind_dsl_search(inp, out) is not None:
                    blind_ok += 1

                # 2. Neural only
                s = prep(inp, out)
                neural_correct = False
                if s is not None:
                    with torch.no_grad():
                        ol, cl1, cl2, pl = adapted(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
                    if (ol.argmax(1).item() == s['op'].item() and cl1.argmax(1).item() == s['c1'].item() and
                        cl2.argmax(1).item() == s['c2'].item() and pl.argmax(1).item() == s['ptr'].item()):
                        neural_ok += 1
                        neural_correct = True

                # 3. Guided DSL search
                guided_found = guided_dsl_search(inp, out, adapted, top_k=20) is not None
                if guided_found:
                    guided_ok += 1

                # 4. Hybrid: neural OR guided
                if neural_correct or guided_found:
                    hybrid_ok += 1

        blind_rate = blind_ok / max(total, 1)
        neural_rate = neural_ok / max(total, 1)
        guided_rate = guided_ok / max(total, 1)
        hybrid_rate = hybrid_ok / max(total, 1)

        results_per_seed['blind'].append(blind_rate)
        results_per_seed['neural'].append(neural_rate)
        results_per_seed['guided'].append(guided_rate)
        results_per_seed['hybrid'].append(hybrid_rate)

        print(f"    Blind DSL:      {blind_rate:.1%}")
        print(f"    Neural Only:    {neural_rate:.1%}")
        print(f"    Guided DSL:     {guided_rate:.1%}")
        print(f"    HYBRID (OR):    {hybrid_rate:.1%}")

    # Summary
    results = {}
    print(f"\n  === RESULTS ({N_SEEDS} seeds) ===")
    for method in ['blind', 'neural', 'guided', 'hybrid']:
        vals = np.array(results_per_seed[method])
        results[f'{method}_mean'] = float(vals.mean())
        results[f'{method}_std'] = float(vals.std())
        results[f'{method}_all'] = [float(v) for v in vals]
        print(f"  {method:<16}: {vals.mean():.1%} ± {vals.std():.1%}")

    results['n_seeds'] = N_SEEDS
    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase62_guided_search.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    methods = ['blind', 'neural', 'guided', 'hybrid']
    labels = ['Blind DSL\n(P20 style)', 'Neural Only\n(GlassBox)', 'Guided DSL\n(Neural+DSL)', 'HYBRID\n(Neural OR DSL)']
    means = [results[f'{m}_mean'] for m in methods]
    stds = [results[f'{m}_std'] for m in methods]
    colors = ['#9E9E9E', '#4CAF50', '#FF9800', '#2196F3']
    bars = ax.bar(range(4), means, yerr=stds, color=colors, alpha=0.85, capsize=5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy'); ax.set_title('Phase 62: Neuro-Symbolic Fusion')
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m+0.03, f'{m:.1%}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase62_guided_search.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
