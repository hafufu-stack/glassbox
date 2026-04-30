"""
Phase 72: Differentiable Object Discovery (Slot Attention)
============================================================
Replace handcrafted extract_objects with a learned module.
Simplified Slot Attention: learn to group pixels into K slots
via iterative attention, then feed slots to GNN.

Compare: BFS objects (handcrafted) vs Slot Attention (learned).
1 seed (architectural exploration).
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
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=8;N_COLORS=10;MAX_GRID=30

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

def prep_bfs(inp,out):
    objs,bg=extract_objects(inp);h,w=len(inp),len(inp[0])
    if not objs:return None
    nf=np.zeros((MAX_OBJECTS,NODE_FEAT_DIM),dtype=np.float32)
    n=min(len(objs),MAX_OBJECTS)
    for j in range(n):f=obj_feats(objs[j],h,w);nf[j,:len(f)]=f[:NODE_FEAT_DIM]
    op,ptr,c1,c2=extract_op(inp,out)
    return{'nf':torch.tensor(nf).unsqueeze(0).float(),'nn':torch.tensor([n]),
           'op':torch.tensor([op]),'c1':torch.tensor([c1]),'c2':torch.tensor([c2]),
           'ptr':torch.tensor([min(ptr,max(n-1,0))])}

def grid_to_tensor(grid):
    """Convert grid to one-hot pixel tensor for slot attention."""
    arr = np.array(grid)
    h, w = arr.shape
    # Pad to MAX_GRID x MAX_GRID
    padded = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int64)
    padded[:min(h,MAX_GRID), :min(w,MAX_GRID)] = arr[:min(h,MAX_GRID), :min(w,MAX_GRID)]
    # One-hot encode colors
    onehot = np.zeros((N_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
    for c in range(N_COLORS):
        onehot[c] = (padded == c).astype(np.float32)
    return torch.tensor(onehot), h, w

def prep_slot(inp, out):
    """Prepare data for slot attention model."""
    grid_t, h, w = grid_to_tensor(inp)
    op, ptr, c1, c2 = extract_op(inp, out)
    return {'grid': grid_t.unsqueeze(0), 'h': h, 'w': w,
            'op': torch.tensor([op]), 'c1': torch.tensor([c1]),
            'c2': torch.tensor([c2]), 'ptr': torch.tensor([min(ptr, MAX_OBJECTS-1)])}

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

# BFS-based GlassBox (baseline)
class BFSAgent(nn.Module):
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

# Simplified Slot Attention Agent
class SlotAttentionAgent(nn.Module):
    def __init__(self, hid=64, n_slots=MAX_OBJECTS, n_iters=3):
        super().__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.hid = hid
        # Pixel encoder: 10 colors -> hid
        self.pixel_enc = nn.Sequential(
            nn.Conv2d(N_COLORS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, hid, 3, padding=1), nn.ReLU(),
        )
        # Slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, 1, hid) * 0.02)
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, hid) * 0.02)
        # Slot attention components
        self.k_proj = nn.Linear(hid, hid)
        self.q_proj = nn.Linear(hid, hid)
        self.v_proj = nn.Linear(hid, hid)
        self.gru = nn.GRUCell(hid, hid)
        self.mlp = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.norm_slots = nn.LayerNorm(hid)
        self.norm_inputs = nn.LayerNorm(hid)
        # Readout (same as BFS agent)
        self.oh = nn.Linear(hid, N_OPS)
        self.c1h = nn.Linear(hid, N_COLORS)
        self.c2h = nn.Linear(hid, N_COLORS)
        self.pq = nn.Linear(hid, hid)
        self.pk = nn.Linear(hid, hid)

    def forward(self, grid_t):
        B = grid_t.size(0)
        # Encode pixels: (B, hid, H, W) -> (B, H*W, hid)
        pixel_feats = self.pixel_enc(grid_t)
        pixel_feats = pixel_feats.flatten(2).permute(0, 2, 1)  # (B, N, hid)
        pixel_feats = self.norm_inputs(pixel_feats)
        # Initialize slots
        slots = self.slot_mu + self.slot_sigma * torch.randn(B, self.n_slots, self.hid, device=grid_t.device)
        # Iterative attention
        k = self.k_proj(pixel_feats)  # (B, N, hid)
        v = self.v_proj(pixel_feats)
        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.q_proj(slots)  # (B, K, hid)
            # Attention: (B, K, N)
            attn = torch.einsum('bkd,bnd->bkn', q, k) / (self.hid ** 0.5)
            attn = F.softmax(attn, dim=1)  # Normalize over slots (competition)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum('bkn,bnd->bkd', attn_norm, v)
            slots = self.gru(updates.reshape(-1, self.hid), slots_prev.reshape(-1, self.hid))
            slots = slots.reshape(B, self.n_slots, self.hid)
            slots = slots + self.mlp(slots)
        # Global pool and readout
        g = slots.mean(dim=1)  # (B, hid)
        pl = (self.pq(g).unsqueeze(1) * self.pk(slots)).sum(-1)
        return self.oh(g), self.c1h(g), self.c2h(g), pl

def compute_loss_bfs(model, s):
    ol,cl1,cl2,pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
    return (F.cross_entropy(ol, s['op'].to(DEVICE)) + F.cross_entropy(cl1, s['c1'].to(DEVICE)) +
            F.cross_entropy(cl2, s['c2'].to(DEVICE)) + F.cross_entropy(pl, s['ptr'].to(DEVICE)))

def compute_loss_slot(model, s):
    ol, cl1, cl2, pl = model(s['grid'].to(DEVICE))
    return (F.cross_entropy(ol, s['op'].to(DEVICE)) + F.cross_entropy(cl1, s['c1'].to(DEVICE)) +
            F.cross_entropy(cl2, s['c2'].to(DEVICE)) + F.cross_entropy(pl, s['ptr'].to(DEVICE)))

def eval_task_bfs(model, task):
    ok, total = 0, 0
    for tp in task.get('test', []):
        s = prep_bfs(tp['input'], tp['output'])
        if s is None: continue
        total += 1
        with torch.no_grad():
            ol, cl1, cl2, pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
        if (ol.argmax(1).item() == s['op'].item() and cl1.argmax(1).item() == s['c1'].item() and
            cl2.argmax(1).item() == s['c2'].item() and pl.argmax(1).item() == s['ptr'].item()):
            ok += 1
    return ok, total

def eval_task_slot(model, task):
    ok, total = 0, 0
    for tp in task.get('test', []):
        s = prep_slot(tp['input'], tp['output'])
        if s is None: continue
        total += 1
        with torch.no_grad():
            ol, cl1, cl2, pl = model(s['grid'].to(DEVICE))
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
    print("Phase 72: Differentiable Object Discovery")
    print("Slot Attention vs BFS handcrafted objects")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Train BFS agent (baseline)
    print("\n  === BFS Agent (Handcrafted Objects) ===")
    bfs_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep_bfs(p['input'], p['output'])
            if s: bfs_samples.append(s)
    bfs_model = BFSAgent().to(DEVICE)
    opt = torch.optim.Adam(bfs_model.parameters(), lr=1e-3)
    for ep in range(80):
        bfs_model.train(); random.shuffle(bfs_samples)
        for s in bfs_samples:
            loss = compute_loss_bfs(bfs_model, s)
            opt.zero_grad(); loss.backward(); opt.step()
    bfs_model.eval()
    bfs_ok, bfs_total = 0, 0
    for task in test_tasks:
        ok, tot = eval_task_bfs(bfs_model, task)
        bfs_ok += ok; bfs_total += tot
    bfs_acc = bfs_ok / max(bfs_total, 1)
    print(f"  BFS accuracy: {bfs_acc:.1%}")

    # Train Slot Attention agent
    print("\n  === Slot Attention Agent (Learned Objects) ===")
    slot_samples = []
    for task in tasks:
        for p in task.get('train', []):
            s = prep_slot(p['input'], p['output'])
            if s: slot_samples.append(s)
    slot_model = SlotAttentionAgent().to(DEVICE)
    n_params = sum(p.numel() for p in slot_model.parameters())
    print(f"  Slot params: {n_params:,}")
    opt = torch.optim.Adam(slot_model.parameters(), lr=1e-3)
    for ep in range(80):
        slot_model.train(); random.shuffle(slot_samples)
        el, nb = 0, 0
        for s in slot_samples:
            loss = compute_loss_slot(slot_model, s)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        if ep % 20 == 0:
            print(f"    Epoch {ep}: loss={el/nb:.4f}")
    slot_model.eval()
    slot_ok, slot_total = 0, 0
    for task in test_tasks:
        ok, tot = eval_task_slot(slot_model, task)
        slot_ok += ok; slot_total += tot
    slot_acc = slot_ok / max(slot_total, 1)
    print(f"  Slot accuracy: {slot_acc:.1%}")

    results = {
        'bfs_acc': bfs_acc, 'slot_acc': slot_acc,
        'bfs_params': sum(p.numel() for p in bfs_model.parameters()),
        'slot_params': n_params,
        'elapsed': time.time() - t0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    with open(os.path.join(RESULTS_DIR, 'phase72_slot.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    bars = ax.bar(['BFS\n(Handcrafted)', 'Slot Attention\n(Learned)'],
                  [bfs_acc, slot_acc], color=['#4CAF50', '#E91E63'], alpha=0.85)
    ax.set_ylabel('Accuracy'); ax.set_title('Phase 72: Object Discovery')
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, a in zip(bars, [bfs_acc, slot_acc]):
        ax.text(b.get_x()+b.get_width()/2, a+0.02, f'{a:.1%}', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase72_slot.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
