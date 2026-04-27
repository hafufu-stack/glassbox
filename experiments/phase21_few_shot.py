"""
Phase 21: Few-Shot Task Adaptation (MAML-style)
==================================================
ARC's core challenge: given 2-3 demo pairs, infer the rule.

Strategy: At test time, fine-tune the GlassBox Agent's weights
on the DEMO examples for a few gradient steps, then predict.

This is Model-Agnostic Meta-Learning (MAML) / Reptile:
- The model's initial weights are a "good starting point"
- A few gradient steps on demos specialize it for each task

Compare:
  A) Fixed model (no adaptation) -- baseline
  B) 5-step adaptation on demos
  C) 20-step adaptation on demos
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

MAX_OBJECTS = 20
NODE_FEAT_DIM = 16
N_OPS = 8
N_COLORS = 10
CONCEPT_DIM = 7

# ============================================================
# Shared utilities
# ============================================================
def extract_objects(grid):
    arr = np.array(grid)
    h, w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    bg = int(np.bincount(arr.flatten()).argmax())
    objects = []
    for r in range(h):
        for c in range(w):
            if not visited[r, c] and arr[r, c] != bg:
                color = int(arr[r, c])
                pixels = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    pixels.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                objects.append({
                    'color': color, 'pixels': pixels, 'area': len(pixels),
                    'center': (np.mean(rows), np.mean(cols)),
                    'bbox': (min(rows), min(cols), max(rows), max(cols)),
                })
    return objects, bg

def object_to_features(obj, h, w):
    feats = [0.0] * 10
    feats[min(obj['color'], 9)] = 1.0
    cr, cc = obj['center']
    feats.extend([cr/max(h,1), cc/max(w,1), obj['area']/max(h*w,1)])
    r0, c0, r1, c1 = obj['bbox']
    feats.extend([(r1-r0+1)/max(h,1), (c1-c0+1)/max(w,1), (c1-c0+1)/max(r1-r0+1,1)])
    return feats[:NODE_FEAT_DIM]

def extract_concepts(grid):
    arr = np.array(grid)
    h, w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    bg = int(np.bincount(arr.flatten()).argmax())
    n_obj = 0
    for r in range(h):
        for c in range(w):
            if not visited[r, c] and arr[r, c] != bg:
                n_obj += 1
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == arr[r, c]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
    h_sym = float(np.array_equal(arr, np.flipud(arr)))
    v_sym = float(np.array_equal(arr, np.fliplr(arr)))
    n_colors = len(set(arr.flatten().tolist()))
    return np.array([n_obj/10, h_sym, v_sym, n_colors/10, max(h,w)/30,
                     float(np.sum(arr==bg))/max(arr.size,1), bg/10], dtype=np.float32)

def extract_op_label(inp, out):
    inp_arr, out_arr = np.array(inp), np.array(out)
    objects, bg = extract_objects(inp)
    if inp_arr.shape == out_arr.shape and np.array_equal(inp_arr, out_arr):
        return 1, 0, 0, 0
    if out_arr.size > 0 and len(np.unique(out_arr)) == 1:
        return 2, 0, int(np.unique(out_arr)[0]), 0
    if inp_arr.shape == out_arr.shape:
        diff = inp_arr != out_arr
        if diff.any():
            old_c, new_c = set(inp_arr[diff].tolist()), set(out_arr[diff].tolist())
            if len(old_c) == 1 and len(new_c) == 1:
                oc = int(list(old_c)[0])
                ptr = 0
                for j, obj in enumerate(objects):
                    if obj['color'] == oc: ptr = j; break
                return 5, min(ptr, MAX_OBJECTS-1), oc, int(list(new_c)[0])
        if np.array_equal(np.flipud(inp_arr), out_arr): return 6, 0, 0, 0
        if np.array_equal(np.fliplr(inp_arr), out_arr): return 7, 0, 0, 0
    return 3, 0, 0, 0

def prepare_sample(inp, out):
    """Prepare a single input-output pair as tensors."""
    objects, bg = extract_objects(inp)
    h, w = len(inp), len(inp[0])
    if not objects:
        return None
    nf = np.zeros((MAX_OBJECTS, NODE_FEAT_DIM), dtype=np.float32)
    n = min(len(objects), MAX_OBJECTS)
    for j in range(n):
        f = object_to_features(objects[j], h, w)
        nf[j, :len(f)] = f[:NODE_FEAT_DIM]
    op_id, ptr_idx, c1, c2 = extract_op_label(inp, out)
    return {
        'nf': torch.tensor(nf, dtype=torch.float32).unsqueeze(0),
        'nn': torch.tensor([n], dtype=torch.long),
        'con': torch.tensor(extract_concepts(inp), dtype=torch.float32).unsqueeze(0),
        'op': torch.tensor([op_id], dtype=torch.long),
        'c1': torch.tensor([c1], dtype=torch.long),
        'c2': torch.tensor([c2], dtype=torch.long),
        'ptr': torch.tensor([min(ptr_idx, max(n-1, 0))], dtype=torch.long),
    }

# ============================================================
# GlassBox Agent (same architecture)
# ============================================================
class GlassBoxAgent(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.gnn1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.concept_proj = nn.Linear(CONCEPT_DIM, hidden)
        self.rel_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.trn_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.router = nn.Sequential(nn.Linear(hidden+CONCEPT_DIM, 16), nn.ReLU(), nn.Linear(16, 2))
        self.op_head = nn.Linear(hidden, N_OPS)
        self.c1_head = nn.Linear(hidden, N_COLORS)
        self.c2_head = nn.Linear(hidden, N_COLORS)
        self.ptr_q = nn.Linear(hidden, hidden)
        self.ptr_k = nn.Linear(hidden, hidden)

    def forward(self, nf, nn_c, con):
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        h = self.node_embed(nf)
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.gnn1(torch.cat([h, msg.expand_as(h)], -1)); h = self.norm1(h) * mf
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.gnn2(torch.cat([h, msg.expand_as(h)], -1)); h = self.norm2(h) * mf
        g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)
        g = g + self.concept_proj(con)
        rw = F.softmax(self.router(torch.cat([g, con], -1)), -1)
        routed = rw[:,0:1]*self.rel_head(g) + rw[:,1:2]*self.trn_head(g)
        ctx = g + routed
        ptr_l = ((self.ptr_q(ctx).unsqueeze(1)) * self.ptr_k(h)).sum(-1).masked_fill(~mask, -1e9)
        return self.op_head(ctx), self.c1_head(ctx), self.c2_head(ctx), ptr_l

# ============================================================
def load_arc_tasks(data_dir, max_tasks=400):
    tasks = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[:max_tasks]
    for fname in files:
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            task = json.load(f)
        tasks.append({'id': fname.replace('.json', ''), **task})
    return tasks

def compute_loss(model, sample):
    """Compute loss for a single prepared sample."""
    ol, cl1, cl2, pl = model(
        sample['nf'].to(DEVICE), sample['nn'].to(DEVICE), sample['con'].to(DEVICE))
    return (F.cross_entropy(ol, sample['op'].to(DEVICE)) +
            F.cross_entropy(cl1, sample['c1'].to(DEVICE)) +
            F.cross_entropy(cl2, sample['c2'].to(DEVICE)) +
            F.cross_entropy(pl, sample['ptr'].to(DEVICE)))

def evaluate_model(model, sample):
    """Evaluate model on a single prepared sample. Returns dict of correct flags."""
    with torch.no_grad():
        ol, cl1, cl2, pl = model(
            sample['nf'].to(DEVICE), sample['nn'].to(DEVICE), sample['con'].to(DEVICE))
    op_ok = ol.argmax(1).item() == sample['op'].item()
    c1_ok = cl1.argmax(1).item() == sample['c1'].item()
    c2_ok = cl2.argmax(1).item() == sample['c2'].item()
    ptr_ok = pl.argmax(1).item() == sample['ptr'].item()
    full_ok = op_ok and c1_ok and c2_ok and ptr_ok
    return {'op': op_ok, 'c1': c1_ok, 'c2': c2_ok, 'ptr': ptr_ok, 'full': full_ok}

def main():
    print("=" * 60)
    print("Phase 21: Few-Shot Task Adaptation (MAML-style)")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare ALL data for meta-training
    all_nf, all_nn, all_con = [], [], []
    all_ops, all_ptrs, all_c1s, all_c2s = [], [], [], []

    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            s = prepare_sample(inp, out)
            if s is None: continue
            all_nf.append(s['nf'].squeeze(0))
            all_nn.append(s['nn'].squeeze(0))
            all_con.append(s['con'].squeeze(0))
            all_ops.append(s['op'].squeeze(0))
            all_c1s.append(s['c1'].squeeze(0))
            all_c2s.append(s['c2'].squeeze(0))
            all_ptrs.append(s['ptr'].squeeze(0))

    nf_t = torch.stack(all_nf)
    nn_t = torch.stack(all_nn)
    con_t = torch.stack(all_con)
    op_t = torch.stack(all_ops)
    c1_t = torch.stack(all_c1s)
    c2_t = torch.stack(all_c2s)
    ptr_t = torch.stack(all_ptrs)

    N = len(nf_t)
    BATCH = 32

    # Meta-train: train on all tasks collectively
    print("\n--- Meta-Training ---")
    model = GlassBoxAgent().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(80):
        model.train()
        perm = torch.randperm(N)
        el, nb = 0, 0
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            ol, cl1, cl2, pl = model(nf_t[idx].to(DEVICE), nn_t[idx].to(DEVICE), con_t[idx].to(DEVICE))
            loss = (F.cross_entropy(ol, op_t[idx].to(DEVICE)) + F.cross_entropy(cl1, c1_t[idx].to(DEVICE)) +
                    F.cross_entropy(cl2, c2_t[idx].to(DEVICE)) + F.cross_entropy(pl, ptr_t[idx].to(DEVICE)))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); el += loss.item(); nb += 1
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/80: loss={el/nb:.4f}")

    # Test: per-task adaptation
    print("\n--- Few-Shot Adaptation ---")
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]
    adaptation_steps_list = [0, 5, 10, 20]
    results_by_steps = {}

    for n_steps in adaptation_steps_list:
        full_ok_total, total = 0, 0

        for task in test_tasks:
            demos = task.get('train', [])
            test_pairs = task.get('test', [])
            if not test_pairs: continue

            # Prepare demo samples
            demo_samples = [prepare_sample(p['input'], p['output']) for p in demos]
            demo_samples = [s for s in demo_samples if s is not None]

            # Prepare test samples
            test_samples = [prepare_sample(p['input'], p['output']) for p in test_pairs]
            test_samples = [s for s in test_samples if s is not None]
            if not test_samples: continue

            if n_steps == 0:
                # No adaptation: use base model
                adapted_model = model
            else:
                # Clone and adapt
                adapted_model = copy.deepcopy(model)
                adapt_opt = torch.optim.SGD(adapted_model.parameters(), lr=1e-2)

                if demo_samples:
                    adapted_model.train()
                    for step in range(n_steps):
                        total_loss = torch.tensor(0.0, device=DEVICE)
                        for s in demo_samples:
                            total_loss = total_loss + compute_loss(adapted_model, s)
                        total_loss = total_loss / len(demo_samples)
                        adapt_opt.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
                        adapt_opt.step()

            # Evaluate on test
            adapted_model.eval()
            for s in test_samples:
                result = evaluate_model(adapted_model, s)
                total += 1
                if result['full']:
                    full_ok_total += 1

        rate = full_ok_total / max(total, 1)
        results_by_steps[n_steps] = rate
        print(f"  Steps={n_steps:>2d}: Full Match = {rate:.1%} ({full_ok_total}/{total})")

    elapsed = time.time() - t0

    out = {
        'results_by_steps': {str(k): v for k, v in results_by_steps.items()},
        'model_params': params,
        'n_test_tasks': len(test_tasks),
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase21_few_shot.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    steps_list = sorted(results_by_steps.keys())
    rates = [results_by_steps[s] for s in steps_list]
    bars = ax.bar([f'{s} steps' for s in steps_list], rates,
                   color=['#FF9800', '#2196F3', '#4CAF50', '#9C27B0'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Full Match Rate')
    ax.set_xlabel('Adaptation Steps on Demo Examples')
    ax.set_title('Phase 21: Few-Shot Task Adaptation (MAML-style)')
    for b, v in zip(bars, rates):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.1%}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase21_few_shot.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
