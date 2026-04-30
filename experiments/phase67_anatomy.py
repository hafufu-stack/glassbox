"""
Phase 67: Mechanistic Anatomy of GNN Layers
=============================================
WHY does GNN Layer 2 ablation work best (P65)?
Use concept probing to analyze what L1 vs L2 encode.

Hypothesis: L1 = low-level features (color, size)
            L2 = high-level rules (operation type)
Destroying L2's "assumptions" and re-learning = optimal.

Probe L1 and L2 hidden states for concept classification
before and after adaptation.
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

class Agent(nn.Module):
    def __init__(s,hid=64):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid)
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid);s.n2=nn.LayerNorm(hid)
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)

    def forward_with_intermediates(s,nf,nn_c):
        """Forward pass returning intermediate layer activations."""
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1);h=s.ne(nf)
        h_embed = h.clone()

        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g1(torch.cat([h,msg.expand_as(h)],-1));h=s.n1(h)*mf
        h_l1 = h.clone()

        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g2(torch.cat([h,msg.expand_as(h)],-1));h=s.n2(h)*mf
        h_l2 = h.clone()

        g=(h*mf).sum(1)/mf.sum(1).clamp(min=1)
        return {'embed': h_embed, 'l1': h_l1, 'l2': h_l2, 'global': g, 'mask': mask}

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

def ablate_l2(model, rate, task_samples):
    """Ablate only GNN Layer 2."""
    am = copy.deepcopy(model); am.train()
    total_loss = torch.tensor(0.0, device=DEVICE)
    for s in task_samples[:8]:
        total_loss = total_loss + compute_loss(am, s)
    total_loss = total_loss / max(len(task_samples[:8]), 1)
    total_loss.backward()
    l2_params = set()
    for p in am.g2.parameters(): l2_params.add(id(p))
    for p in am.n2.parameters(): l2_params.add(id(p))
    with torch.no_grad():
        for p in am.parameters():
            if id(p) not in l2_params: continue
            if p.grad is not None:
                importance = p.grad.abs()
                threshold = torch.quantile(importance.flatten(), rate)
                p.mul_((importance > threshold).float())
    am.eval(); return am

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

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.fc(x)

def train_probe(features, labels, n_classes, epochs=50):
    """Train a linear probe and return accuracy."""
    probe = LinearProbe(features.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    n = len(features)
    split = int(n * 0.8)
    train_f, train_l = features[:split], labels[:split]
    test_f, test_l = features[split:], labels[split:]
    if len(test_f) == 0: return 0.0

    for _ in range(epochs):
        logits = probe(train_f)
        loss = F.cross_entropy(logits, train_l)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        preds = probe(test_f).argmax(1)
        acc = (preds == test_l).float().mean().item()
    return acc

def load_arc_tasks(d, n=400):
    t = []
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d, f), 'r', encoding='utf-8') as fp:
                t.append({'id': f[:-5], **json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 67: Mechanistic Anatomy of GNN Layers")
    print("Why does L2 ablation work best?")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

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
            loss = compute_loss(model, s)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    print("  Base model trained.")

    # Collect hidden states and labels for probing
    concepts = {
        'op': {'labels': [], 'n_classes': N_OPS},     # Operation type (high-level rule)
        'color': {'labels': [], 'n_classes': N_COLORS}, # Dominant color (low-level feature)
    }
    layer_features = {'embed': [], 'l1': [], 'l2': []}

    print("  Collecting hidden states...")
    with torch.no_grad():
        for s in all_samples[:500]:
            intermediates = model.forward_with_intermediates(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
            mask = intermediates['mask']
            mf = mask.float().unsqueeze(-1)

            for lname in ['embed', 'l1', 'l2']:
                h = intermediates[lname]
                # Global pool
                g = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
                layer_features[lname].append(g.cpu())

            concepts['op']['labels'].append(s['op'].item())
            concepts['color']['labels'].append(s['c1'].item())

    # Stack features
    for lname in layer_features:
        layer_features[lname] = torch.cat(layer_features[lname], dim=0).to(DEVICE)

    results = {}

    # Probe each layer for each concept
    print("\n  === CONCEPT PROBING (Base Model) ===")
    print(f"  {'Layer':<8} | {'Op (Rule)':<12} | {'Color (Feature)':<15}")
    print("  " + "-" * 40)

    for lname in ['embed', 'l1', 'l2']:
        feats = layer_features[lname]
        for cname, cdata in concepts.items():
            labels = torch.tensor(cdata['labels'], device=DEVICE)
            acc = train_probe(feats, labels, cdata['n_classes'])
            results[f'base_{lname}_{cname}'] = acc

        op_acc = results[f'base_{lname}_op']
        color_acc = results[f'base_{lname}_color']
        print(f"  {lname:<8} | {op_acc:<11.1%} | {color_acc:<14.1%}")

    # Now probe AFTER L2 ablation + adaptation on some test tasks
    print("\n  === CONCEPT PROBING (After L2 Ablation + Adapt) ===")

    adapted_features = {'embed': [], 'l1': [], 'l2': []}
    adapted_labels = {'op': [], 'color': []}

    for task in test_tasks[:20]:
        demos = task.get('train', [])
        aug_pairs = []
        for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
        aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
        aug_samples = [s for s in aug_samples if s]

        adapted = ablate_l2(model, 0.15, aug_samples)
        adapted = adapt_model(adapted, aug_samples, steps=100)
        adapted.eval()

        with torch.no_grad():
            for s in aug_samples[:5]:
                intermediates = adapted.forward_with_intermediates(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
                mask = intermediates['mask']
                mf = mask.float().unsqueeze(-1)
                for lname in ['embed', 'l1', 'l2']:
                    h = intermediates[lname]
                    g = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
                    adapted_features[lname].append(g.cpu())
                adapted_labels['op'].append(s['op'].item())
                adapted_labels['color'].append(s['c1'].item())

    for lname in adapted_features:
        adapted_features[lname] = torch.cat(adapted_features[lname], dim=0).to(DEVICE)

    print(f"  {'Layer':<8} | {'Op (Rule)':<12} | {'Color (Feature)':<15}")
    print("  " + "-" * 40)

    for lname in ['embed', 'l1', 'l2']:
        feats = adapted_features[lname]
        for cname in concepts:
            labels = torch.tensor(adapted_labels[cname], device=DEVICE)
            acc = train_probe(feats, labels, concepts[cname]['n_classes'])
            results[f'adapted_{lname}_{cname}'] = acc

        op_acc = results[f'adapted_{lname}_op']
        color_acc = results[f'adapted_{lname}_color']
        print(f"  {lname:<8} | {op_acc:<11.1%} | {color_acc:<14.1%}")

    # Analysis: L1 vs L2 specialization
    print("\n  === LAYER SPECIALIZATION ANALYSIS ===")
    for lname in ['l1', 'l2']:
        base_op = results[f'base_{lname}_op']
        base_color = results[f'base_{lname}_color']
        adapt_op = results[f'adapted_{lname}_op']
        adapt_color = results[f'adapted_{lname}_color']
        print(f"  {lname}: Op {base_op:.1%}->{adapt_op:.1%} ({adapt_op-base_op:+.1%}), "
              f"Color {base_color:.1%}->{adapt_color:.1%} ({adapt_color-base_color:+.1%})")

    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase67_anatomy.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    layers = ['embed', 'l1', 'l2']
    x = np.arange(len(layers))
    width = 0.35

    # Left: Base model probing
    ax = axes[0]
    op_vals = [results[f'base_{l}_op'] for l in layers]
    color_vals = [results[f'base_{l}_color'] for l in layers]
    ax.bar(x - width/2, op_vals, width, label='Op (Rule)', color='#E91E63', alpha=0.85)
    ax.bar(x + width/2, color_vals, width, label='Color (Feature)', color='#4CAF50', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['Embedding', 'GNN L1', 'GNN L2'])
    ax.set_ylabel('Probe Accuracy'); ax.set_title('Base Model: What Each Layer Encodes')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.0)

    # Right: After adaptation
    ax = axes[1]
    op_vals = [results[f'adapted_{l}_op'] for l in layers]
    color_vals = [results[f'adapted_{l}_color'] for l in layers]
    ax.bar(x - width/2, op_vals, width, label='Op (Rule)', color='#E91E63', alpha=0.85)
    ax.bar(x + width/2, color_vals, width, label='Color (Feature)', color='#4CAF50', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['Embedding', 'GNN L1', 'GNN L2'])
    ax.set_ylabel('Probe Accuracy'); ax.set_title('After L2 Ablation + Adaptation')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.0)

    plt.suptitle('Phase 67: Mechanistic Anatomy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase67_anatomy.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
