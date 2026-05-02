"""
Phase 93: Dynamic Macro-Action Discovery
==========================================
Break through P90's action space saturation (12 actions -> plateau at 32 rollouts)
by discovering and adding macro-actions (frequent action sequences) to the DSL.

Inspired by DreamCoder: if the agent keeps picking action A then B,
fuse them into a single macro-action "AB" to expand the vocabulary.
"""
import os,sys,json,time,copy,random,math
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from collections import deque, Counter
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','training')
RESULTS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results')
FIGURES_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'figures')
os.makedirs(RESULTS_DIR,exist_ok=True);os.makedirs(FIGURES_DIR,exist_ok=True)
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=8;N_COLORS=10

# ========== Reuse from P89 ==========
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
        return s.oh(g),s.c1h(g),s.c2h(g),pl,g

def compute_loss(model,s):
    ol,cl1,cl2,pl,_=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
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

def ablate_l2(model,rate,task_samples):
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

def eval_task(model,task):
    ok,total=0,0
    for tp in task.get('test',[]):
        s=prep(tp['input'],tp['output'])
        if s is None:continue
        total+=1
        with torch.no_grad():
            ol,cl1,cl2,pl,_=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
        if(ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
           cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok,total

def demo_loss(model, task_samples):
    if not task_samples: return 999.0
    total = 0.0
    with torch.no_grad():
        for s in task_samples[:8]:
            total += compute_loss(model, s).item()
    return total / min(len(task_samples), 8)

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8') as fp:
                t.append({'id':f[:-5],**json.load(fp)})
    return t

def reptile_meta_train(model, train_tasks, n_outer=200, n_inner=10, meta_lr=0.1, inner_lr=0.1):
    print("  Reptile meta-learning...")
    for outer in range(n_outer):
        task = random.choice(train_tasks)
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_s = [s for s in aug_s if s]
        if not aug_s: continue
        adapted = adapt_model(model, aug_s, steps=n_inner, lr=inner_lr)
        with torch.no_grad():
            for p_m, p_a in zip(model.parameters(), adapted.parameters()):
                p_m.data += meta_lr * (p_a.data - p_m.data)
        if outer > 0 and outer % 50 == 0: meta_lr *= 0.7
    print(f"    Reptile done ({n_outer} outer steps)")
    return model


# ========================================
# Macro-Action Discovery
# ========================================
BASE_ACTIONS = [
    (0.0,  0.1,  10), (0.0,  0.05, 10), (0.0,  0.15, 10),
    (0.10, 0.1,  10), (0.15, 0.1,  10), (0.20, 0.1,  10),
    (0.15, 0.05, 10), (0.15, 0.15, 10), (0.20, 0.05, 10),
    (0.10, 0.1,  25), (0.15, 0.1,  25), (0.20, 0.1,  25),
]

def execute_action(model, action, task_samples):
    """Execute a single base action: ablate + adapt."""
    a_rate, lr, steps = action
    m = copy.deepcopy(model)
    if a_rate > 0:
        m = ablate_l2(m, a_rate, task_samples)
    return adapt_model(m, task_samples, steps=steps, lr=lr)

def execute_macro(model, macro_actions, task_samples):
    """Execute a macro-action: sequential application of base actions.

    This is the key insight: instead of independently adapting from scratch
    each time, chain multiple adaptations sequentially. This allows
    compound strategies like 'ablate aggressively then fine-tune gently'.
    """
    current = model
    for action in macro_actions:
        current = execute_action(current, action, task_samples)
    return current

def discover_macros(model, train_tasks, n_tasks=30, n_rollouts=16):
    """Run MCTS on tasks, track which action sequences lead to success.

    For each task:
    1. Run n_rollouts of random 2-step sequences
    2. Track which pairs succeed
    3. Return the most frequent successful pairs as macro-actions
    """
    pair_successes = Counter()  # (action_i, action_j) -> success count
    pair_attempts = Counter()

    for task in train_tasks[:n_tasks]:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_s = [s for s in aug_s if s]
        if not aug_s: continue

        for _ in range(n_rollouts):
            # Pick two random base actions
            i = random.randint(0, len(BASE_ACTIONS)-1)
            j = random.randint(0, len(BASE_ACTIONS)-1)
            pair = (i, j)
            pair_attempts[pair] += 1

            # Execute as macro (sequential)
            macro = [BASE_ACTIONS[i], BASE_ACTIONS[j]]
            adapted = execute_macro(model, macro, aug_s)
            ok, tot = eval_task(adapted, task)
            if ok == tot and tot > 0:
                pair_successes[pair] += 1

    # Find pairs with highest success rate (min 3 attempts)
    macro_candidates = []
    for pair, successes in pair_successes.items():
        attempts = pair_attempts[pair]
        if attempts >= 3:
            rate = successes / attempts
            macro_candidates.append((pair, rate, successes, attempts))

    macro_candidates.sort(key=lambda x: (-x[1], -x[2]))
    return macro_candidates[:8]  # Top 8 macros


def mcts_with_macros(model, task_samples, action_library, n_rollouts=16):
    """MCTS using expanded action library (base + macro actions)."""
    n_actions = len(action_library)
    visits = np.zeros(n_actions, dtype=np.float32)
    values = np.zeros(n_actions, dtype=np.float32)
    total_visits = 1
    best_loss = float('inf'); best_model = model

    for _ in range(n_rollouts):
        ucbs = []
        for i in range(n_actions):
            if visits[i] == 0:
                ucbs.append(float('inf'))
            else:
                ucbs.append(values[i]/visits[i] + 1.41*math.sqrt(math.log(total_visits)/visits[i]))
        idx = max(range(n_actions), key=lambda i: ucbs[i])

        action_entry = action_library[idx]
        if action_entry['type'] == 'base':
            adapted = execute_action(model, action_entry['action'], task_samples)
        else:  # macro
            adapted = execute_macro(model, action_entry['actions'], task_samples)

        loss = demo_loss(adapted, task_samples)
        visits[idx] += 1; values[idx] += -loss; total_visits += 1
        if loss < best_loss:
            best_loss = loss; best_model = adapted

    return best_model, best_loss


def main():
    print("=" * 60)
    print("Phase 93: Dynamic Macro-Action Discovery")
    print("Break through 12-action saturation via macro-actions")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    train_tasks = tasks[:split]; test_tasks = tasks[split:]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Train base model
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

    model = reptile_meta_train(model, train_tasks, n_outer=200, n_inner=10)
    model.eval()

    # Step 1: Discover macro-actions
    print("\n  Discovering macro-actions...")
    macros = discover_macros(model, train_tasks, n_tasks=30, n_rollouts=16)
    print(f"  Found {len(macros)} macro-action candidates:")
    for (i, j), rate, succ, att in macros:
        a1 = BASE_ACTIONS[i]; a2 = BASE_ACTIONS[j]
        print(f"    [{i}->{j}] rate={rate:.1%} ({succ}/{att}) | "
              f"abl={a1[0]:.2f},lr={a1[1]:.2f},st={a1[2]} -> abl={a2[0]:.2f},lr={a2[1]:.2f},st={a2[2]}")

    # Step 2: Build expanded action library
    base_library = [{'type': 'base', 'action': a, 'name': f'base_{i}'}
                    for i, a in enumerate(BASE_ACTIONS)]
    macro_library = []
    for (i, j), rate, succ, att in macros:
        if rate >= 0.3:  # Only add macros with >= 30% success rate
            macro_library.append({
                'type': 'macro',
                'actions': [BASE_ACTIONS[i], BASE_ACTIONS[j]],
                'name': f'macro_{i}_{j}',
                'success_rate': rate,
            })
    expanded_library = base_library + macro_library
    print(f"\n  Action library: {len(base_library)} base + {len(macro_library)} macros = {len(expanded_library)} total")

    # Step 3: Evaluate scaling with expanded library
    rollout_counts = [8, 16, 32, 64]
    results_base_only = {}
    results_expanded = {}

    for n_roll in rollout_counts:
        print(f"\n  === Rollouts: {n_roll} ===")

        # Base-only MCTS
        t1 = time.time()
        ok_b, tot_b = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_s = [s for s in aug_s if s]
            adapted, _ = mcts_with_macros(model, aug_s, base_library, n_rollouts=n_roll)
            ok, tot = eval_task(adapted, task); ok_b += ok; tot_b += tot
        acc_b = ok_b / max(tot_b, 1)
        time_b = time.time() - t1
        results_base_only[n_roll] = {'acc': acc_b, 'time': time_b}
        print(f"    Base-only ({len(base_library)} actions): {acc_b:.1%} ({time_b:.1f}s)")

        # Expanded MCTS
        t2 = time.time()
        ok_e, tot_e = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_s = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_s = [s for s in aug_s if s]
            adapted, _ = mcts_with_macros(model, aug_s, expanded_library, n_rollouts=n_roll)
            ok, tot = eval_task(adapted, task); ok_e += ok; tot_e += tot
        acc_e = ok_e / max(tot_e, 1)
        time_e = time.time() - t2
        results_expanded[n_roll] = {'acc': acc_e, 'time': time_e}
        print(f"    Expanded  ({len(expanded_library)} actions): {acc_e:.1%} ({time_e:.1f}s)")

    # Save results
    results = {
        'n_base_actions': len(base_library),
        'n_macro_actions': len(macro_library),
        'n_total_actions': len(expanded_library),
        'macros_discovered': [(list(m[0]), m[1], m[2], m[3]) for m in macros],
        'macros_used': [m['name'] for m in macro_library],
        'base_only': {str(k): v for k, v in results_base_only.items()},
        'expanded': {str(k): v for k, v in results_expanded.items()},
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, 'phase93_macro_actions.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot: Scaling curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    rolls = sorted(rollout_counts)
    base_accs = [results_base_only[r]['acc'] for r in rolls]
    exp_accs = [results_expanded[r]['acc'] for r in rolls]
    ax.plot(rolls, base_accs, 'o--', lw=2, ms=10, color='#9E9E9E',
            label=f'Base only ({len(base_library)} actions)')
    ax.plot(rolls, exp_accs, 's-', lw=3, ms=10, color='#2196F3',
            label=f'Expanded ({len(expanded_library)} actions)')
    ax.set_xlabel('Rollouts', fontsize=12); ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xscale('log', base=2); ax.set_xticks(rolls)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_ylim(0.7, 1.0)
    plt.title('Phase 93: Macro-Actions Unlock Scaling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase93_macro_actions.png'), dpi=150); plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
