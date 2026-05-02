"""
Phase 88: Test-Time Ablation Search (GA)
==========================================
Optimize WHICH 15% of neurons to ablate per task using
Genetic Algorithm search at test time.

Instead of "gradient-smallest 15%", evolve task-specific
ablation masks via GA + Meta-Init fast TTT (10 steps).
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

def adapt_model(model,task_samples,steps=10,lr=0.1):
    if not task_samples:return model
    am=copy.deepcopy(model);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    for _ in range(steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
    am.eval();return am

def ablate_l2_gradient(model,rate,task_samples):
    """Standard gradient-based ablation (baseline)."""
    am=copy.deepcopy(model);am.train()
    tl=torch.tensor(0.0,device=DEVICE)
    for s in task_samples[:8]:tl=tl+compute_loss(am,s)
    tl=tl/max(len(task_samples[:8]),1);tl.backward()
    l2p=set()
    for p in am.g2.parameters():l2p.add(id(p))
    for p in am.n2.parameters():l2p.add(id(p))
    with torch.no_grad():
        for p in am.parameters():
            if id(p) not in l2p:continue
            if p.grad is not None:
                imp=p.grad.abs();thr=torch.quantile(imp.flatten(),rate)
                p.mul_((imp>thr).float())
    am.eval();return am

def get_l2_param_count(model):
    return sum(p.numel() for p in model.g2.parameters()) + sum(p.numel() for p in model.n2.parameters())

def apply_mask(model, mask_binary):
    """Apply a binary mask to L2 parameters (1=keep, 0=ablate)."""
    am = copy.deepcopy(model)
    offset = 0
    with torch.no_grad():
        for p in list(am.g2.parameters()) + list(am.n2.parameters()):
            n = p.numel()
            m = mask_binary[offset:offset+n].reshape(p.shape).to(p.device)
            p.mul_(m)
            offset += n
    am.eval()
    return am

def demo_loss(model, task_samples):
    if not task_samples: return 999.0
    total = 0.0
    with torch.no_grad():
        for s in task_samples[:8]:
            total += compute_loss(model, s).item()
    return total / min(len(task_samples), 8)

def eval_task(model,task):
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

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8') as fp:
                t.append({'id':f[:-5],**json.load(fp)})
    return t


# ==============================
# Genetic Algorithm Ablation Search
# ==============================

def ga_ablation_search(model, task_samples, n_params, target_rate=0.15,
                       pop_size=20, n_generations=10, ttt_steps=10, lr=0.1):
    """Evolve optimal ablation mask using GA + fast TTT."""
    n_ablate = int(n_params * target_rate)

    # Initialize population: random masks with ~15% zeros
    population = []
    for _ in range(pop_size):
        mask = torch.ones(n_params)
        indices = random.sample(range(n_params), n_ablate)
        for idx in indices: mask[idx] = 0.0
        population.append(mask)

    # Add gradient-based mask as seed (warm start)
    grad_model = ablate_l2_gradient(model, target_rate, task_samples)
    grad_mask = torch.ones(n_params)
    offset = 0
    for p_orig, p_abl in zip(
        list(model.g2.parameters()) + list(model.n2.parameters()),
        list(grad_model.g2.parameters()) + list(grad_model.n2.parameters())
    ):
        n = p_orig.numel()
        orig_flat = p_orig.data.flatten()
        abl_flat = p_abl.data.flatten()
        for i in range(n):
            if abs(orig_flat[i].item()) > 1e-10 and abs(abl_flat[i].item()) < 1e-10:
                grad_mask[offset + i] = 0.0
        offset += n
    population[0] = grad_mask  # Replace first individual

    best_mask = population[0]
    best_fitness = float('inf')

    for gen in range(n_generations):
        # Evaluate fitness (demo loss after TTT)
        fitness = []
        for mask in population:
            masked_model = apply_mask(model, mask)
            adapted = adapt_model(masked_model, task_samples, steps=ttt_steps, lr=lr)
            loss = demo_loss(adapted, task_samples)
            fitness.append(loss)
            if loss < best_fitness:
                best_fitness = loss
                best_mask = mask.clone()

        # Selection: tournament
        sorted_idx = np.argsort(fitness)
        survivors = [population[i] for i in sorted_idx[:pop_size//2]]

        # Crossover + Mutation
        new_pop = list(survivors)
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            # Uniform crossover
            child = torch.where(torch.rand(n_params) < 0.5, p1, p2)
            # Mutation: flip ~2% of bits
            for _ in range(max(1, n_params // 50)):
                idx = random.randint(0, n_params - 1)
                child[idx] = 1.0 - child[idx]
            # Enforce ~15% ablation rate
            n_zeros = int((child == 0).sum().item())
            if n_zeros < n_ablate:
                ones = (child == 1).nonzero(as_tuple=True)[0]
                flip = random.sample(ones.tolist(), min(n_ablate - n_zeros, len(ones)))
                for idx in flip: child[idx] = 0.0
            elif n_zeros > n_ablate:
                zeros = (child == 0).nonzero(as_tuple=True)[0]
                flip = random.sample(zeros.tolist(), min(n_zeros - n_ablate, len(zeros)))
                for idx in flip: child[idx] = 1.0
            new_pop.append(child)

        population = new_pop[:pop_size]

    return best_mask, best_fitness


def main():
    print("=" * 60)
    print("Phase 88: Test-Time Ablation Search (GA)")
    print("Evolve task-specific ablation masks")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

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
            loss = compute_loss(model, s); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    print("  Base model trained.")

    n_l2 = get_l2_param_count(model)
    print(f"  L2 params: {n_l2}")

    # Evaluate: Gradient ablation vs GA ablation
    grad_ok, ga_ok, total = 0, 0, 0
    grad_time, ga_time = 0, 0
    ga_improvements = 0

    for ti, task in enumerate(test_tasks):
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_s = [s for s in aug_s if s]
        if not aug_s: continue

        # Gradient-based ablation (baseline)
        t1 = time.time()
        grad_model = ablate_l2_gradient(model, 0.15, aug_s)
        grad_adapted = adapt_model(grad_model, aug_s, steps=10, lr=0.1)
        grad_time += time.time() - t1
        ok_g, tot = eval_task(grad_adapted, task)
        grad_ok += ok_g; total += tot

        # GA ablation search
        t2 = time.time()
        best_mask, _ = ga_ablation_search(
            model, aug_s, n_l2,
            target_rate=0.15, pop_size=16, n_generations=5, ttt_steps=10
        )
        ga_model = apply_mask(model, best_mask)
        ga_adapted = adapt_model(ga_model, aug_s, steps=10, lr=0.1)
        ga_time += time.time() - t2
        ok_ga, _ = eval_task(ga_adapted, task)
        ga_ok += ok_ga
        if ok_ga > ok_g: ga_improvements += 1

        if (ti+1) % 20 == 0:
            print(f"    {ti+1}/{len(test_tasks)}: Grad={grad_ok}/{total}, GA={ga_ok}/{total}")

    grad_acc = grad_ok / max(total, 1)
    ga_acc = ga_ok / max(total, 1)

    print(f"\n  === Results ===")
    print(f"  Gradient Ablation: {grad_acc:.1%} ({grad_time:.1f}s)")
    print(f"  GA Ablation:       {ga_acc:.1%} ({ga_time:.1f}s)")
    print(f"  GA improved on {ga_improvements}/{len(test_tasks)} tasks")

    results = {
        'grad_acc': grad_acc, 'ga_acc': ga_acc,
        'grad_time': grad_time, 'ga_time': ga_time,
        'ga_improvements': ga_improvements,
        'n_tasks': len(test_tasks),
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase88_ablation_search.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    bars = ax.bar(['Gradient\nAblation', 'GA\nAblation'], [grad_acc, ga_acc],
                  color=['#1565C0', '#E91E63'], alpha=0.85, edgecolor='white', lw=2)
    ax.set_ylabel('Accuracy'); ax.set_title('Ablation Strategy Comparison')
    ax.set_ylim(0.7, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, a in zip(bars, [grad_acc, ga_acc]):
        ax.text(b.get_x()+b.get_width()/2, a+0.01, f'{a:.1%}', ha='center', fontweight='bold')

    ax = axes[1]
    bars = ax.bar(['Gradient', 'GA Search'], [grad_time, ga_time],
                  color=['#1565C0', '#E91E63'], alpha=0.85)
    ax.set_ylabel('Time (s)'); ax.set_title('Compute Cost')
    ax.grid(True, alpha=0.3, axis='y')
    for b, t in zip(bars, [grad_time, ga_time]):
        ax.text(b.get_x()+b.get_width()/2, t+1, f'{t:.0f}s', ha='center', fontweight='bold')

    plt.suptitle('Phase 88: Test-Time Ablation Search (GA)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase88_ablation_search.png'), dpi=150); plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
