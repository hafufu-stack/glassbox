"""
Phase 47: Skill Patch Memory Bank
=====================================
P44 failed because it averaged ALL super-compensation knowledge into one model.
Instead: store each task's ΔW as an independent "skill patch" (like a LoRA adapter).

At test time:
  1. Encode the new task's features
  2. Find the most similar training task(s) in the memory bank
  3. Inject that task's skill patch into the base model
  4. THEN run one-punch on top

This is Lifelong Modular Learning: each experience is a reusable cartridge.
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

def task_fingerprint(task):
    """Create a simple feature vector for task similarity matching."""
    feats = []
    for p in task.get('train', []):
        ia = np.array(p['input']); oa = np.array(p['output'])
        feats.extend([ia.shape[0], ia.shape[1], len(np.unique(ia)),
                      oa.shape[0] if len(oa.shape) > 0 else 0,
                      oa.shape[1] if len(oa.shape) > 1 else 0,
                      len(np.unique(oa)),
                      1.0 if ia.shape == oa.shape else 0.0,
                      ia.size, oa.size])
    # Pad/truncate to fixed length
    fp = np.zeros(50, dtype=np.float32)
    fp[:min(len(feats), 50)] = feats[:50]
    return fp / (np.linalg.norm(fp) + 1e-8)


def main():
    print("=" * 60)
    print("Phase 47: Skill Patch Memory Bank")
    print("Store each task's super-comp as a reusable patch")
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

    split = int(len(tasks) * 0.8)
    train_tasks = tasks[:split]
    test_tasks = tasks[split:]

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

    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ===== Build Skill Patch Memory Bank =====
    print("\n  Building skill patch memory bank...")
    memory_bank = []  # list of (fingerprint, skill_patch_dict)

    for i, task in enumerate(train_tasks):
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]
        if not aug_samples: continue

        # Run one-punch to get super-compensated model
        ablated = ablate_least_important(model, 0.15, aug_samples)
        adapted = adapt_model(ablated, aug_samples, steps=50)

        # Compute ΔW = adapted - base
        delta_w = {}
        for k in base_state:
            delta_w[k] = (adapted.state_dict()[k] - base_state[k]).cpu()

        fp = task_fingerprint(task)
        memory_bank.append((fp, delta_w))

        if (i + 1) % 50 == 0:
            print(f"    Built {i+1}/{len(train_tasks)} patches")

    print(f"  Memory bank size: {len(memory_bank)} patches")

    # ===== Test-time: Retrieve and Inject =====
    def find_closest_patches(test_task, k=1):
        """Find k most similar training tasks by fingerprint cosine similarity."""
        test_fp = task_fingerprint(test_task)
        sims = []
        for j, (fp, dw) in enumerate(memory_bank):
            sim = float(np.dot(test_fp, fp))
            sims.append((sim, j))
        sims.sort(reverse=True)
        return [memory_bank[j][1] for _, j in sims[:k]]

    def inject_patch(model, patches, alpha=1.0):
        """Inject averaged skill patches into model."""
        patched = copy.deepcopy(model)
        state = patched.state_dict()
        for k in state:
            avg_delta = torch.stack([p[k] for p in patches]).mean(0).to(DEVICE)
            state[k] = state[k] + alpha * avg_delta
        patched.load_state_dict(state)
        return patched

    # Compare methods
    configs = [
        ('adapt_only',     'Adapt 50 (P28)',            False, False, 0),
        ('one_punch',      'One Punch (P45)',           True,  False, 0),
        ('patch_1',        'Patch (k=1) + Adapt',       False, True,  1),
        ('patch_3',        'Patch (k=3) + Adapt',       False, True,  3),
        ('patch_1_punch',  'Patch (k=1) + One Punch',   True,  True,  1),
        ('patch_3_punch',  'Patch (k=3) + One Punch',   True,  True,  3),
    ]
    results = {}

    print(f"\n  {'Method':<35} | {'Full Match':>10}")
    print("  " + "-" * 50)

    for key, label, do_punch, do_patch, k_patches in configs:
        ok_total, n_total = 0, 0
        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_samples = [s for s in aug_samples if s]

            current = model
            if do_patch:
                patches = find_closest_patches(task, k=k_patches)
                current = inject_patch(current, patches, alpha=0.5)

            if do_punch:
                current = ablate_least_important(current, 0.15, aug_samples)
                current = adapt_model(current, aug_samples, steps=50)
            else:
                current = adapt_model(current, aug_samples, steps=50)

            ok, tot = eval_task(current, task)
            ok_total += ok; n_total += tot

        rate = ok_total / max(n_total, 1)
        results[key] = rate
        print(f"  {label:<35} | {rate:>9.1%}")

    elapsed = time.time() - t0
    results['elapsed'] = elapsed
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    results['memory_bank_size'] = len(memory_bank)

    with open(os.path.join(RESULTS_DIR, 'phase47_skillpatch.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [c[1] for c in configs]
    vals = [results[c[0]] for c in configs]
    colors = ['#FF9800', '#E91E63', '#4CAF50', '#2196F3', '#9C27B0', '#673AB7']
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Full Match'); ax.set_title('Phase 47: Skill Patch Memory Bank')
    ax.set_ylim(0, 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase47_skillpatch.png'), dpi=150)
    plt.close()
    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__': main()
