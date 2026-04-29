"""
Phase 41: Progressive Ablation (My own addition)
====================================================
Phase 34 showed one-shot 15% ablation works. But what about
PROGRESSIVE stress — like gradually increasing workout intensity?

Instead of destroying 15% at once:
  Ablate 5% -> Adapt 15 steps -> Ablate 5% more -> Adapt 15 steps -> Ablate 5% more -> Adapt 20 steps
  Total: 15% destroyed, 50 steps adapted, but spread over 3 rounds.

Hypothesis: Gradual stress allows the network to build resilience
incrementally, like progressive overload in strength training.
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

def ablate_random(model, rate):
    am = copy.deepcopy(model)
    with torch.no_grad():
        for p in am.parameters():
            p.mul_((torch.rand_like(p) > rate).float())
    return am

def oneshot_ablate_adapt(model, total_rate, aug_samples, total_steps=50):
    """Standard one-shot: destroy total_rate at once, then adapt total_steps."""
    ablated = ablate_random(model, total_rate)
    return adapt_model(ablated, aug_samples, steps=total_steps)

def progressive_ablate_adapt(model, total_rate, aug_samples, rounds=3, total_steps=50):
    """Progressive ablation: spread destruction across multiple rounds.
    Each round: ablate (total_rate/rounds)% -> adapt (total_steps/rounds) steps.
    Like progressive overload in weight training."""
    rate_per_round = total_rate / rounds
    steps_per_round = total_steps // rounds
    # Give last round any remaining steps
    last_round_steps = total_steps - steps_per_round * (rounds - 1)

    am = copy.deepcopy(model)
    for r in range(rounds):
        # Ablate
        with torch.no_grad():
            for p in am.parameters():
                p.mul_((torch.rand_like(p) > rate_per_round).float())
        # Adapt
        steps = last_round_steps if r == rounds - 1 else steps_per_round
        am = adapt_model(am, aug_samples, steps=steps)

    return am

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
    print("Phase 41: Progressive Ablation")
    print("Gradual stress vs sudden shock")
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

    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    # Compare methods
    total_rates = [0.10, 0.15, 0.20]
    round_options = [1, 2, 3, 5]  # 1 = one-shot
    results = {}

    # Baseline: adapt only
    ok_base, n_base = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        adapted = adapt_model(model, aug_samples, steps=50)
        ok, tot = eval_task(adapted, task)
        ok_base += ok; n_base += tot
    base_rate = ok_base / max(n_base, 1)
    results['adapt_only'] = base_rate
    print(f"\n  Adapt-only: {base_rate:.1%}")

    print(f"\n{'Rate':<6} | {'Rounds':<7} | {'Full Match':<11} | {'Delta':<10} | {'Type'}")
    print("-" * 55)

    for total_rate in total_rates:
        for rounds in round_options:
            ok_total, n_total = 0, 0
            for task in test_tasks:
                demos = task.get('train', [])
                aug_pairs = []
                for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
                aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
                aug_samples = [s for s in aug_samples if s]

                if rounds == 1:
                    adapted = oneshot_ablate_adapt(model, total_rate, aug_samples)
                else:
                    adapted = progressive_ablate_adapt(model, total_rate, aug_samples, rounds=rounds)

                ok, tot = eval_task(adapted, task)
                ok_total += ok; n_total += tot

            rate = ok_total / max(n_total, 1)
            delta = rate - base_rate
            label = "one-shot" if rounds == 1 else f"progressive"
            results[f'rate{int(total_rate*100)}_rounds{rounds}'] = rate
            print(f"  {total_rate:.0%}   | {rounds:<7} | {rate:.1%}      | {delta:+.1%}    | {label}")

    elapsed = time.time() - t0
    results['elapsed'] = elapsed
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase41_progressive.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot: heatmap of rate x rounds
    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.zeros((len(total_rates), len(round_options)))
    for i, tr in enumerate(total_rates):
        for j, ro in enumerate(round_options):
            data[i, j] = results.get(f'rate{int(tr*100)}_rounds{ro}', 0)

    im = ax.imshow(data, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(round_options)))
    ax.set_xticklabels([f'{r} rounds' if r > 1 else '1 (one-shot)' for r in round_options])
    ax.set_yticks(range(len(total_rates)))
    ax.set_yticklabels([f'{int(r*100)}%' for r in total_rates])
    ax.set_xlabel('Number of Rounds'); ax.set_ylabel('Total Ablation Rate')
    ax.set_title('Progressive Ablation: Rate x Rounds Heatmap')
    for i in range(len(total_rates)):
        for j in range(len(round_options)):
            ax.text(j, i, f'{data[i,j]:.1%}', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im, label='Full Match')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase41_progressive.png'), dpi=150)
    plt.close()
    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__': main()
