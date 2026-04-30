"""
Phase 73: Auto-Regressive Execution (Visual Chain of Thought)
===============================================================
Instead of 1-shot prediction, let the model output a sequence of
operations and apply them iteratively. Add STOP token.

Design: Model predicts op -> apply to grid -> feed back -> repeat.
Max 5 steps. Compare 1-shot vs auto-regressive.
1 seed (architectural experiment).
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
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=9;N_COLORS=10  # +1 for STOP

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
    if ia.shape==oa.shape and np.array_equal(ia,oa):return 0,0,0,0  # STOP (op=0)
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

class AutoRegressiveAgent(nn.Module):
    """Agent with STOP token (op=0). Can be called iteratively."""
    def __init__(s,hid=64):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid)
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid);s.n2=nn.LayerNorm(hid)
        # N_OPS=9 includes STOP=0
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)
        # Step embedding (which step in the chain)
        s.step_emb=nn.Embedding(5,hid)

    def forward(s,nf,nn_c,step=0):
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1);h=s.ne(nf)
        # Add step embedding
        step_t=torch.tensor([step],device=nf.device).clamp(0,4)
        h=h+s.step_emb(step_t).unsqueeze(1)
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g1(torch.cat([h,msg.expand_as(h)],-1));h=s.n1(h)*mf
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g2(torch.cat([h,msg.expand_as(h)],-1));h=s.n2(h)*mf
        g=(h*mf).sum(1)/mf.sum(1).clamp(min=1)
        pl=((s.pq(g).unsqueeze(1))*s.pk(h)).sum(-1).masked_fill(~mask,-1e9)
        return s.oh(g),s.c1h(g),s.c2h(g),pl

def compute_loss(model,s,step=0):
    ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE),step=step)
    return(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
           F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))

def adapt_model(model, task_samples, steps=100, lr=1e-2):
    if not task_samples: return model
    am=copy.deepcopy(model);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    for _ in range(steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
    am.eval();return am

def apply_op(grid, op, ptr, c1, c2):
    """Apply a predicted operation to the grid (simplified DSL)."""
    arr = np.array(grid)
    if op == 0:  # STOP
        return grid, True
    if op == 2:  # Fill
        return [[c1]*len(grid[0]) for _ in range(len(grid))], False
    if op == 5:  # Recolor
        out = arr.copy()
        out[out == c1] = c2
        return out.tolist(), False
    if op == 6:  # FlipV
        return np.flipud(arr).tolist(), False
    if op == 7:  # FlipH
        return np.fliplr(arr).tolist(), False
    # Unknown ops: return unchanged
    return grid, False

def eval_task_oneshot(model, task):
    """Standard 1-shot evaluation."""
    ok, total = 0, 0
    for tp in task.get('test', []):
        s = prep(tp['input'], tp['output'])
        if s is None: continue
        total += 1
        with torch.no_grad():
            ol,cl1,cl2,pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        if (ol.argmax(1).item() == s['op'].item() and cl1.argmax(1).item() == s['c1'].item() and
            cl2.argmax(1).item() == s['c2'].item() and pl.argmax(1).item() == s['ptr'].item()):
            ok += 1
    return ok, total

def eval_task_autoregressive(model, task, max_steps=5):
    """Auto-regressive: predict op, apply, re-extract, repeat."""
    ok, total = 0, 0
    for tp in task.get('test', []):
        s_gt = prep(tp['input'], tp['output'])
        if s_gt is None: continue
        total += 1
        current_grid = [list(row) for row in tp['input']]
        predicted_ops = []
        for step in range(max_steps):
            s = prep(current_grid, tp['output'])
            if s is None: break
            with torch.no_grad():
                ol, cl1, cl2, pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE), step=step)
            op = ol.argmax(1).item()
            c1_pred = cl1.argmax(1).item()
            c2_pred = cl2.argmax(1).item()
            ptr_pred = pl.argmax(1).item()
            predicted_ops.append(op)
            if op == 0:  # STOP
                break
            current_grid, _ = apply_op(current_grid, op, ptr_pred, c1_pred, c2_pred)
        # Check final prediction matches ground truth op chain
        if (len(predicted_ops) > 0 and predicted_ops[0] == s_gt['op'].item() and
            cl1.argmax(1).item() == s_gt['c1'].item() and
            cl2.argmax(1).item() == s_gt['c2'].item() and
            pl.argmax(1).item() == s_gt['ptr'].item()):
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
    print("Phase 73: Auto-Regressive Execution")
    print("Visual Chain of Thought: multi-step prediction")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Prepare training data
    all_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep(p['input'], p['output'])
            if s: all_samples.append(s)

    # Train auto-regressive model
    model = AutoRegressiveAgent().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(80):
        model.train(); random.shuffle(all_samples)
        for s in all_samples:
            # Train with step=0 (first step) for base prediction
            loss = compute_loss(model, s, step=0)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    # Also train with step augmentation (steps 1-4 see same data but with step embedding)
    for ep in range(20):
        model.train(); random.shuffle(all_samples)
        for s in all_samples:
            step = random.randint(0, 4)
            loss = compute_loss(model, s, step=step)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    # Evaluate
    oneshot_ok, oneshot_total = 0, 0
    ar_ok, ar_total = 0, 0
    for task in test_tasks:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        adapted = adapt_model(model, aug_samples, steps=100)

        ok1, tot1 = eval_task_oneshot(adapted, task)
        ok2, tot2 = eval_task_autoregressive(adapted, task, max_steps=5)
        oneshot_ok += ok1; oneshot_total += tot1
        ar_ok += ok2; ar_total += tot2

    oneshot_acc = oneshot_ok / max(oneshot_total, 1)
    ar_acc = ar_ok / max(ar_total, 1)

    print(f"\n  1-Shot accuracy:          {oneshot_acc:.1%}")
    print(f"  Auto-Regressive accuracy: {ar_acc:.1%}")

    results = {
        'oneshot_acc': oneshot_acc, 'ar_acc': ar_acc,
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    with open(os.path.join(RESULTS_DIR, 'phase73_autoregressive.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    bars = ax.bar(['1-Shot', 'Auto-Regressive\n(5 steps)'],
                  [oneshot_acc, ar_acc], color=['#9E9E9E', '#E91E63'], alpha=0.85)
    ax.set_ylabel('Accuracy'); ax.set_title('Phase 73: Auto-Regressive Execution')
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, a in zip(bars, [oneshot_acc, ar_acc]):
        ax.text(b.get_x()+b.get_width()/2, a+0.02, f'{a:.1%}', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase73_autoregressive.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
