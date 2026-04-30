"""
Phase 74: TTT-Guided Self-Play
================================
Use TTT as an oracle to generate curriculum learning data.
1. Generate synthetic ARC tasks using simple DSL rules
2. Filter: keep only tasks that base model fails but TTT solves
3. Train base model on these "zone of proximal development" tasks
4. Repeat. Measure base model improvement.

2 iterations of self-play. 1 seed.
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

def adapt_model(model, task_samples, steps=100, lr=0.1):
    if not task_samples: return model
    am=copy.deepcopy(model);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    for _ in range(steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
    am.eval();return am

def ablate_l2(model, rate, task_samples):
    am=copy.deepcopy(model);am.train()
    total_loss=torch.tensor(0.0,device=DEVICE)
    for s in task_samples[:8]:total_loss=total_loss+compute_loss(am,s)
    total_loss=total_loss/max(len(task_samples[:8]),1);total_loss.backward()
    l2_params=set()
    for p in am.g2.parameters():l2_params.add(id(p))
    for p in am.n2.parameters():l2_params.add(id(p))
    with torch.no_grad():
        for p in am.parameters():
            if id(p) not in l2_params:continue
            if p.grad is not None:
                imp=p.grad.abs();thr=torch.quantile(imp.flatten(),rate)
                p.mul_((imp>thr).float())
    am.eval();return am

def eval_model(model, task):
    ok,total=0,0
    for tp in task.get('test',[]):
        s=prep(tp['input'],tp['output'])
        if s is None:continue
        total+=1
        with torch.no_grad():
            ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
        if(ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
           cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok,total

def generate_synthetic_task():
    """Generate a simple synthetic ARC-like task."""
    h, w = random.randint(3, 10), random.randint(3, 10)
    bg = random.randint(0, 9)
    n_objs = random.randint(1, 3)
    grid = [[bg]*w for _ in range(h)]

    # Place random rectangles
    for _ in range(n_objs):
        color = random.choice([c for c in range(10) if c != bg])
        r0, c0 = random.randint(0, h-1), random.randint(0, w-1)
        r1 = min(r0 + random.randint(1, 3), h)
        c1 = min(c0 + random.randint(1, 3), w)
        for r in range(r0, r1):
            for c in range(c0, c1):
                grid[r][c] = color

    # Apply random operation
    arr = np.array(grid)
    op = random.choice(['identity', 'recolor', 'flipv', 'fliph', 'fill'])
    if op == 'identity':
        out_grid = grid
    elif op == 'recolor':
        old_c = random.choice([c for c in range(10) if c != bg and c in arr])
        new_c = random.choice([c for c in range(10) if c != old_c])
        out = arr.copy(); out[out == old_c] = new_c
        out_grid = out.tolist()
    elif op == 'flipv':
        out_grid = np.flipud(arr).tolist()
    elif op == 'fliph':
        out_grid = np.fliplr(arr).tolist()
    elif op == 'fill':
        fill_c = random.randint(0, 9)
        out_grid = [[fill_c]*w for _ in range(h)]
    else:
        out_grid = grid

    return {'train': [{'input': grid, 'output': out_grid}],
            'test': [{'input': grid, 'output': out_grid}]}

def load_arc_tasks(d, n=400):
    t = []
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                t.append({'id': f[:-5], **json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 74: TTT-Guided Self-Play")
    print("Synthetic curriculum: TTT as oracle teacher")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Initial training on real data
    all_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep(p['input'], p['output'])
            if s: all_samples.append(s)

    model = Agent().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(80):
        model.train(); random.shuffle(all_samples)
        for s in all_samples:
            loss = compute_loss(model, s)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    # Evaluate baseline
    base_ok, base_total = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        adapted = adapt_model(model, aug_samples, steps=100, lr=0.1)
        ok, tot = eval_model(adapted, task)
        base_ok += ok; base_total += tot
    base_acc = base_ok / max(base_total, 1)
    print(f"  Baseline (real data only): {base_acc:.1%}")

    iteration_accs = [base_acc]

    # Self-play iterations
    N_ITERATIONS = 2
    N_SYNTHETIC = 200  # synthetic tasks per iteration

    for iteration in range(N_ITERATIONS):
        print(f"\n  === Self-Play Iteration {iteration+1}/{N_ITERATIONS} ===")

        # Generate synthetic tasks
        good_samples = []
        n_tried, n_base_fail_ttt_pass = 0, 0

        for _ in range(N_SYNTHETIC * 5):
            syn_task = generate_synthetic_task()
            train_pair = syn_task['train'][0]
            s = prep(train_pair['input'], train_pair['output'])
            if s is None: continue
            n_tried += 1

            # Check base model (0-shot)
            with torch.no_grad():
                ol, cl1, cl2, pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
            base_correct = (ol.argmax(1).item() == s['op'].item() and
                           cl1.argmax(1).item() == s['c1'].item())

            if not base_correct:
                # TTT can solve?
                aug_pairs = augment_pair(train_pair['input'], train_pair['output'])
                aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                aug_samples = [s2 for s2 in aug_samples if s2]
                ablated = ablate_l2(model, 0.20, aug_samples)
                adapted = adapt_model(ablated, aug_samples, steps=100, lr=0.1)
                with torch.no_grad():
                    ol2, cl12, cl22, pl2 = adapted(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
                ttt_correct = (ol2.argmax(1).item() == s['op'].item() and
                              cl12.argmax(1).item() == s['c1'].item())
                if ttt_correct:
                    n_base_fail_ttt_pass += 1
                    good_samples.append(s)
                    if len(good_samples) >= N_SYNTHETIC:
                        break

        print(f"    Generated {n_tried} tasks, found {n_base_fail_ttt_pass} ZPD samples")

        if not good_samples:
            print("    No ZPD samples found, skipping iteration")
            iteration_accs.append(iteration_accs[-1])
            continue

        # Train model on ZPD samples + original data
        combined_samples = all_samples + good_samples
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        for ep in range(30):
            model.train(); random.shuffle(combined_samples)
            for s in combined_samples:
                loss = compute_loss(model, s)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

        # Re-evaluate
        iter_ok, iter_total = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_samples = [s2 for s2 in aug_samples if s2]
            adapted = adapt_model(model, aug_samples, steps=100, lr=0.1)
            ok, tot = eval_model(adapted, task)
            iter_ok += ok; iter_total += tot
        iter_acc = iter_ok / max(iter_total, 1)
        iteration_accs.append(iter_acc)
        print(f"    After iteration {iteration+1}: {iter_acc:.1%} ({iter_acc-base_acc:+.1%} vs baseline)")

    results = {
        'baseline_acc': base_acc,
        'iteration_accs': iteration_accs,
        'final_acc': iteration_accs[-1],
        'improvement': iteration_accs[-1] - base_acc,
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    with open(os.path.join(RESULTS_DIR, 'phase74_selfplay.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(len(iteration_accs)), iteration_accs, 'o-', linewidth=2, color='#E91E63', markersize=8)
    ax.set_xlabel('Self-Play Iteration'); ax.set_ylabel('Accuracy')
    ax.set_title('Phase 74: TTT-Guided Self-Play')
    ax.set_xticks(range(len(iteration_accs)))
    ax.set_xticklabels(['Baseline'] + [f'Iter {i+1}' for i in range(N_ITERATIONS)])
    ax.grid(True, alpha=0.3); ax.set_ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase74_selfplay.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
