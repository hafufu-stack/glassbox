"""
Phase 22: Task Embedding (Gradient-Free Adaptation)
=====================================================
Phase 21: 20 gradient steps -> 79.3%. But gradients are SLOW.
Can we get the same effect WITHOUT gradients?

Method: Encode demo pairs into a "task embedding" vector,
then condition the model on it. No test-time gradients needed.
"""
import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MAX_OBJECTS = 20; NODE_FEAT_DIM = 16; N_OPS = 8; N_COLORS = 10; CONCEPT_DIM = 7; MAX_GRID = 30

def extract_objects(grid):
    arr = np.array(grid); h, w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    bg = int(np.bincount(arr.flatten()).argmax())
    objects = []
    for r in range(h):
        for c in range(w):
            if not visited[r,c] and arr[r,c] != bg:
                color = int(arr[r,c]); pixels = []; queue = deque([(r,c)]); visited[r,c] = True
                while queue:
                    cr,cc = queue.popleft(); pixels.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and arr[nr,nc]==color:
                            visited[nr,nc]=True; queue.append((nr,nc))
                rows=[p[0] for p in pixels]; cols=[p[1] for p in pixels]
                objects.append({'color':color,'area':len(pixels),'center':(np.mean(rows),np.mean(cols)),
                    'bbox':(min(rows),min(cols),max(rows),max(cols))})
    return objects, bg

def object_to_features(obj, h, w):
    feats = [0.0]*10; feats[min(obj['color'],9)] = 1.0
    cr,cc = obj['center']
    feats.extend([cr/max(h,1), cc/max(w,1), obj['area']/max(h*w,1)])
    r0,c0,r1,c1 = obj['bbox']
    feats.extend([(r1-r0+1)/max(h,1),(c1-c0+1)/max(w,1),(c1-c0+1)/max(r1-r0+1,1)])
    return feats[:NODE_FEAT_DIM]

def extract_concepts(grid):
    arr = np.array(grid); h,w = arr.shape
    bg = int(np.bincount(arr.flatten()).argmax())
    h_sym = float(np.array_equal(arr, np.flipud(arr)))
    v_sym = float(np.array_equal(arr, np.fliplr(arr)))
    n_colors = len(set(arr.flatten().tolist()))
    return np.array([0, h_sym, v_sym, n_colors/10, max(h,w)/30,
                     float(np.sum(arr==bg))/max(arr.size,1), bg/10], dtype=np.float32)

def extract_op_label(inp, out):
    ia, oa = np.array(inp), np.array(out)
    objects, bg = extract_objects(inp)
    if ia.shape==oa.shape and np.array_equal(ia,oa): return 1,0,0,0
    if oa.size>0 and len(np.unique(oa))==1: return 2,0,int(np.unique(oa)[0]),0
    if ia.shape==oa.shape:
        diff=ia!=oa
        if diff.any():
            oc,nc=set(ia[diff].tolist()),set(oa[diff].tolist())
            if len(oc)==1 and len(nc)==1:
                o=int(list(oc)[0]); ptr=0
                for j,obj in enumerate(objects):
                    if obj['color']==o: ptr=j; break
                return 5,min(ptr,MAX_OBJECTS-1),o,int(list(nc)[0])
        if np.array_equal(np.flipud(ia),oa): return 6,0,0,0
        if np.array_equal(np.fliplr(ia),oa): return 7,0,0,0
    return 3,0,0,0

def pad_grid(grid, pad_val=10):
    h,w = len(grid), len(grid[0])
    p = np.full((MAX_GRID,MAX_GRID), pad_val, dtype=np.int64)
    p[:h,:w] = np.array(grid); return p

def prepare_sample(inp, out):
    objects, bg = extract_objects(inp)
    h,w = len(inp), len(inp[0])
    if not objects: return None
    nf = np.zeros((MAX_OBJECTS, NODE_FEAT_DIM), dtype=np.float32)
    n = min(len(objects), MAX_OBJECTS)
    for j in range(n):
        f = object_to_features(objects[j],h,w); nf[j,:len(f)] = f[:NODE_FEAT_DIM]
    op,ptr,c1,c2 = extract_op_label(inp, out)
    return {'nf':torch.tensor(nf).unsqueeze(0), 'nn':torch.tensor([n]),
            'con':torch.tensor(extract_concepts(inp)).unsqueeze(0),
            'op':torch.tensor([op]), 'c1':torch.tensor([c1]),
            'c2':torch.tensor([c2]), 'ptr':torch.tensor([min(ptr,max(n-1,0))]),
            'inp_flat':torch.tensor(pad_grid(inp)).unsqueeze(0),
            'out_flat':torch.tensor(pad_grid(out)).unsqueeze(0)}

class TaskConditionedAgent(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.grid_embed = nn.Embedding(11, 32)
        self.demo_encoder = nn.Sequential(nn.Linear(32*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.gnn1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden); self.norm2 = nn.LayerNorm(hidden)
        self.task_fuse = nn.Linear(hidden*2, hidden)
        self.op_head = nn.Linear(hidden, N_OPS)
        self.c1_head = nn.Linear(hidden, N_COLORS)
        self.c2_head = nn.Linear(hidden, N_COLORS)
        self.ptr_q = nn.Linear(hidden, hidden); self.ptr_k = nn.Linear(hidden, hidden)

    def encode_demo(self, inp_flat, out_flat):
        ie = self.grid_embed(inp_flat.clamp(0,10)).mean(dim=[1,2])
        oe = self.grid_embed(out_flat.clamp(0,10)).mean(dim=[1,2])
        return self.demo_encoder(torch.cat([ie, oe], -1))

    def forward(self, nf, nn_c, task_emb):
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        h = self.node_embed(nf)
        msg = (h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h = h+self.gnn1(torch.cat([h,msg.expand_as(h)],-1)); h = self.norm1(h)*mf
        msg = (h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h = h+self.gnn2(torch.cat([h,msg.expand_as(h)],-1)); h = self.norm2(h)*mf
        g = (h*mf).sum(1)/mf.sum(1).clamp(min=1)
        ctx = self.task_fuse(torch.cat([g, task_emb], -1))
        ptr_l = ((self.ptr_q(ctx).unsqueeze(1))*self.ptr_k(h)).sum(-1).masked_fill(~mask,-1e9)
        return self.op_head(ctx), self.c1_head(ctx), self.c2_head(ctx), ptr_l

class BaselineAgent(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.gnn1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden); self.norm2 = nn.LayerNorm(hidden)
        self.op_head = nn.Linear(hidden, N_OPS)
        self.c1_head = nn.Linear(hidden, N_COLORS)
        self.c2_head = nn.Linear(hidden, N_COLORS)
        self.ptr_q = nn.Linear(hidden, hidden); self.ptr_k = nn.Linear(hidden, hidden)

    def forward(self, nf, nn_c):
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        h = self.node_embed(nf)
        msg = (h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h = h+self.gnn1(torch.cat([h,msg.expand_as(h)],-1)); h = self.norm1(h)*mf
        msg = (h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h = h+self.gnn2(torch.cat([h,msg.expand_as(h)],-1)); h = self.norm2(h)*mf
        g = (h*mf).sum(1)/mf.sum(1).clamp(min=1)
        ptr_l = ((self.ptr_q(g).unsqueeze(1))*self.ptr_k(h)).sum(-1).masked_fill(~mask,-1e9)
        return self.op_head(g), self.c1_head(g), self.c2_head(g), ptr_l

def load_arc_tasks(data_dir, max_tasks=400):
    tasks = []
    for f in sorted(os.listdir(data_dir))[:max_tasks]:
        if f.endswith('.json'):
            with open(os.path.join(data_dir,f),'r',encoding='utf-8') as fp:
                tasks.append({'id':f.replace('.json',''), **json.load(fp)})
    return tasks

def main():
    print("="*60); print("Phase 22: Task Embedding (Gradient-Free Adaptation)"); print("="*60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR); print(f"Loaded {len(tasks)} tasks")

    # Organize by task (group demo pairs)
    task_data = []
    for task in tasks:
        pairs = task.get('train',[])
        samples = [prepare_sample(p['input'],p['output']) for p in pairs]
        samples = [s for s in samples if s is not None]
        if len(samples) >= 2: task_data.append(samples)

    split = int(len(task_data)*0.8)
    train_tasks, test_tasks = task_data[:split], task_data[split:]
    print(f"Tasks with 2+ demos: {len(task_data)}, Train: {split}, Test: {len(test_tasks)}")

    # Train task-conditioned model
    print("\n--- Training Task-Conditioned Agent ---")
    model = TaskConditionedAgent().to(DEVICE)
    baseline = BaselineAgent().to(DEVICE)
    print(f"TaskCond params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Baseline params: {sum(p.numel() for p in baseline.parameters()):,}")

    opt_m = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)

    for epoch in range(60):
        model.train(); baseline.train()
        el_m, el_b, nb = 0, 0, 0
        for samples in train_tasks:
            # Use first demo as context, rest as query
            demo = samples[0]
            for s in samples[1:]:
                task_emb = model.encode_demo(demo['inp_flat'].to(DEVICE), demo['out_flat'].to(DEVICE))
                ol,cl1,cl2,pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE), task_emb)
                loss_m = (F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
                          F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))
                opt_m.zero_grad(); loss_m.backward(); opt_m.step(); el_m += loss_m.item()

                ol2,cl12,cl22,pl2 = baseline(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
                loss_b = (F.cross_entropy(ol2,s['op'].to(DEVICE))+F.cross_entropy(cl12,s['c1'].to(DEVICE))+
                          F.cross_entropy(cl22,s['c2'].to(DEVICE))+F.cross_entropy(pl2,s['ptr'].to(DEVICE)))
                opt_b.zero_grad(); loss_b.backward(); opt_b.step(); el_b += loss_b.item(); nb += 1
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/60: cond={el_m/nb:.3f}, base={el_b/nb:.3f}")

    # Evaluate
    print("\n--- Evaluation ---")
    model.eval(); baseline.eval()
    cond_ok, base_ok, total = 0, 0, 0

    with torch.no_grad():
        for samples in test_tasks:
            if len(samples) < 2: continue
            demo = samples[0]
            task_emb = model.encode_demo(demo['inp_flat'].to(DEVICE), demo['out_flat'].to(DEVICE))
            for s in samples[1:]:
                total += 1
                ol,cl1,cl2,pl = model(s['nf'].to(DEVICE), s['nn'].to(DEVICE), task_emb)
                if (ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
                    cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):
                    cond_ok += 1
                ol2,cl12,cl22,pl2 = baseline(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
                if (ol2.argmax(1).item()==s['op'].item() and cl12.argmax(1).item()==s['c1'].item() and
                    cl22.argmax(1).item()==s['c2'].item() and pl2.argmax(1).item()==s['ptr'].item()):
                    base_ok += 1

    cond_rate = cond_ok/max(total,1); base_rate = base_ok/max(total,1)
    print(f"Baseline (no task info): {base_rate:.1%}")
    print(f"Task-Conditioned:        {cond_rate:.1%}")
    print(f"Improvement:             {cond_rate-base_rate:+.1%}")

    elapsed = time.time()-t0
    out = {'baseline': base_rate, 'task_conditioned': cond_rate,
           'improvement': cond_rate-base_rate, 'n_test': total, 'elapsed': elapsed,
           'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(os.path.join(RESULTS_DIR,'phase22_task_embedding.json'),'w',encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(['Baseline\n(no task info)','Task-Conditioned\n(demo embedding)'],
                   [base_rate, cond_rate], color=['#FF9800','#4CAF50'])
    ax.set_ylim(0,1); ax.set_ylabel('Full Match')
    ax.set_title('Phase 22: Gradient-Free Task Adaptation')
    for b,v in zip(bars,[base_rate,cond_rate]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.1%}', ha='center', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,'phase22_task_embedding.png'), dpi=150); plt.close()
    print(f"\nElapsed: {elapsed:.1f}s"); return out

if __name__ == '__main__': main()
