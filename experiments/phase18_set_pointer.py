"""
Phase 18: Broadcast Set-Pointer
=================================
Current Pointer: argmax -> select ONE object.
ARC hard tasks: "recolor ALL red objects" -> need to select MULTIPLE objects.

Upgrade: Set-Pointer.
Instead of argmax(attention), use threshold(attention > tau) to select
a SET of objects, then broadcast the operation to all of them.

This enables for-each / for-all style abstract reasoning.
"""

import os, sys, json, time
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

# Labels: detect multi-object ops (for-each)
def extract_op_label_multi(inp, out):
    """Extract op label + set of target objects."""
    inp_arr, out_arr = np.array(inp), np.array(out)
    objects, bg = extract_objects(inp)

    if inp_arr.shape == out_arr.shape and np.array_equal(inp_arr, out_arr):
        return 1, [], 0, 0  # IDENTITY

    if inp_arr.shape == out_arr.shape:
        diff = inp_arr != out_arr
        if diff.any():
            old_c = set(inp_arr[diff].tolist())
            new_c = set(out_arr[diff].tolist())
            if len(old_c) == 1 and len(new_c) == 1:
                oc = int(list(old_c)[0])
                # Find ALL objects with this color
                targets = [j for j, obj in enumerate(objects) if obj['color'] == oc]
                return 5, targets, oc, int(list(new_c)[0])

        if np.array_equal(np.flipud(inp_arr), out_arr):
            return 6, list(range(len(objects))), 0, 0
        if np.array_equal(np.fliplr(inp_arr), out_arr):
            return 7, list(range(len(objects))), 0, 0

    if out_arr.size > 0 and len(np.unique(out_arr)) == 1:
        return 2, list(range(len(objects))), int(np.unique(out_arr)[0]), 0

    return 3, [0], 0, 0

# ============================================================
# Models: Single-Pointer vs Set-Pointer
# ============================================================
class SinglePointer(nn.Module):
    """Baseline: argmax pointer (Phase 8 style)."""
    def __init__(self, hidden=64):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.mp1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden)
        self.mp2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.op_head = nn.Linear(hidden, N_OPS)
        self.c1_head = nn.Linear(hidden, N_COLORS)
        self.c2_head = nn.Linear(hidden, N_COLORS)
        self.ptr_q = nn.Linear(hidden, hidden)
        self.ptr_k = nn.Linear(hidden, hidden)

    def forward(self, nf, nn_c):
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        h = self.node_embed(nf)
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.mp1(torch.cat([h, msg.expand_as(h)], -1)); h = self.norm1(h) * mf
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.mp2(torch.cat([h, msg.expand_as(h)], -1)); h = self.norm2(h) * mf
        g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)
        ptr_l = ((self.ptr_q(g).unsqueeze(1)) * self.ptr_k(h)).sum(-1).masked_fill(~mask, -1e9)
        return self.op_head(g), self.c1_head(g), self.c2_head(g), ptr_l

class SetPointer(nn.Module):
    """Upgrade: sigmoid threshold pointer for multi-selection."""
    def __init__(self, hidden=64):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEAT_DIM, hidden)
        self.mp1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden)
        self.mp2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.op_head = nn.Linear(hidden, N_OPS)
        self.c1_head = nn.Linear(hidden, N_COLORS)
        self.c2_head = nn.Linear(hidden, N_COLORS)
        # Set pointer: binary prediction per node (include or not)
        self.set_ptr = nn.Linear(hidden * 2, 1)  # graph_ctx + node -> include?

    def forward(self, nf, nn_c):
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_c.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        h = self.node_embed(nf)
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.mp1(torch.cat([h, msg.expand_as(h)], -1)); h = self.norm1(h) * mf
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.mp2(torch.cat([h, msg.expand_as(h)], -1)); h = self.norm2(h) * mf
        g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)

        # Set pointer: for each node, predict "include in set?"
        g_exp = g.unsqueeze(1).expand(-1, MAX_OBJECTS, -1)
        set_logits = self.set_ptr(torch.cat([g_exp, h], -1)).squeeze(-1)  # (B, MAX_OBJ)
        set_logits = set_logits.masked_fill(~mask, -1e9)

        return self.op_head(g), self.c1_head(g), self.c2_head(g), set_logits

# ============================================================
def load_arc_tasks(data_dir, max_tasks=400):
    tasks = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[:max_tasks]
    for fname in files:
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            task = json.load(f)
        tasks.append({'id': fname.replace('.json', ''), **task})
    return tasks

def main():
    print("=" * 60)
    print("Phase 18: Broadcast Set-Pointer")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare data with multi-target labels
    all_nf, all_nn = [], []
    all_ops, all_c1s, all_c2s = [], [], []
    all_single_ptr = []  # argmax target
    all_set_ptr = []     # binary mask of targets

    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            objects, bg = extract_objects(inp)
            h, w = len(inp), len(inp[0])
            if not objects: continue

            nf = np.zeros((MAX_OBJECTS, NODE_FEAT_DIM), dtype=np.float32)
            n = min(len(objects), MAX_OBJECTS)
            for j in range(n):
                f = object_to_features(objects[j], h, w)
                nf[j, :len(f)] = f[:NODE_FEAT_DIM]

            op_id, targets, c1, c2 = extract_op_label_multi(inp, out)

            # Single pointer: first target
            single_ptr = min(targets[0], n-1) if targets else 0

            # Set pointer: binary mask
            set_mask = np.zeros(MAX_OBJECTS, dtype=np.float32)
            for t in targets:
                if t < MAX_OBJECTS:
                    set_mask[t] = 1.0

            all_nf.append(nf); all_nn.append(n)
            all_ops.append(op_id); all_c1s.append(c1); all_c2s.append(c2)
            all_single_ptr.append(single_ptr)
            all_set_ptr.append(set_mask)

    nf_t = torch.tensor(np.array(all_nf), dtype=torch.float32)
    nn_t = torch.tensor(all_nn, dtype=torch.long)
    op_t = torch.tensor(all_ops, dtype=torch.long)
    c1_t = torch.tensor(all_c1s, dtype=torch.long)
    c2_t = torch.tensor(all_c2s, dtype=torch.long)
    sptr_t = torch.tensor(all_single_ptr, dtype=torch.long)
    mptr_t = torch.tensor(np.array(all_set_ptr), dtype=torch.float32)

    N = len(nf_t)
    split = int(N * 0.8)
    BATCH = 32
    EPOCHS = 60

    # Count multi-target samples
    multi_count = sum(1 for m in all_set_ptr if m.sum() > 1)
    print(f"Total: {N}, Train: {split}, Test: {N-split}")
    print(f"Multi-target samples: {multi_count}/{N} ({multi_count/N:.1%})")

    results = {}

    # ===== Model A: Single Pointer =====
    print("\n--- Training: Single Pointer ---")
    model_a = SinglePointer().to(DEVICE)
    params_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Params: {params_a:,}")
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model_a.train()
        perm = torch.randperm(split)
        el, nb = 0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            ol, cl1, cl2, pl = model_a(nf_t[idx].to(DEVICE), nn_t[idx].to(DEVICE))
            loss = (F.cross_entropy(ol, op_t[idx].to(DEVICE)) + F.cross_entropy(cl1, c1_t[idx].to(DEVICE)) +
                    F.cross_entropy(cl2, c2_t[idx].to(DEVICE)) + F.cross_entropy(pl, sptr_t[idx].to(DEVICE)))
            opt_a.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
            opt_a.step(); el += loss.item(); nb += 1
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: loss={el/nb:.4f}")

    # ===== Model B: Set Pointer =====
    print("\n--- Training: Set Pointer ---")
    model_b = SetPointer().to(DEVICE)
    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"  Params: {params_b:,}")
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model_b.train()
        perm = torch.randperm(split)
        el, nb = 0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            ol, cl1, cl2, sl = model_b(nf_t[idx].to(DEVICE), nn_t[idx].to(DEVICE))
            loss = (F.cross_entropy(ol, op_t[idx].to(DEVICE)) + F.cross_entropy(cl1, c1_t[idx].to(DEVICE)) +
                    F.cross_entropy(cl2, c2_t[idx].to(DEVICE)) +
                    F.binary_cross_entropy_with_logits(sl, mptr_t[idx].to(DEVICE)))
            opt_b.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
            opt_b.step(); el += loss.item(); nb += 1
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: loss={el/nb:.4f}")

    # Evaluate
    model_a.eval(); model_b.eval()
    single_op_ok, single_full_ok = 0, 0
    set_op_ok, set_full_ok, set_select_ok = 0, 0, 0
    total = 0

    with torch.no_grad():
        for i in range(split, N):
            nf = nf_t[i:i+1].to(DEVICE)
            nn_b = nn_t[i:i+1].to(DEVICE)
            total += 1

            # Single pointer
            ol_a, cl1_a, cl2_a, pl_a = model_a(nf, nn_b)
            op_a = ol_a.argmax(1).item()
            if op_a == op_t[i].item(): single_op_ok += 1
            if (op_a == op_t[i].item() and cl1_a.argmax(1).item() == c1_t[i].item() and
                cl2_a.argmax(1).item() == c2_t[i].item() and pl_a.argmax(1).item() == sptr_t[i].item()):
                single_full_ok += 1

            # Set pointer
            ol_b, cl1_b, cl2_b, sl_b = model_b(nf, nn_b)
            op_b = ol_b.argmax(1).item()
            if op_b == op_t[i].item(): set_op_ok += 1

            # Set selection accuracy
            pred_set = (torch.sigmoid(sl_b[0]) > 0.5).cpu().numpy()
            true_set = mptr_t[i].numpy() > 0.5
            if np.array_equal(pred_set, true_set):
                set_select_ok += 1
            if (op_b == op_t[i].item() and cl1_b.argmax(1).item() == c1_t[i].item() and
                cl2_b.argmax(1).item() == c2_t[i].item() and np.array_equal(pred_set, true_set)):
                set_full_ok += 1

    s_op = single_op_ok / max(total, 1)
    s_full = single_full_ok / max(total, 1)
    b_op = set_op_ok / max(total, 1)
    b_sel = set_select_ok / max(total, 1)
    b_full = set_full_ok / max(total, 1)

    print(f"\n--- Results ---")
    print(f"Single Pointer: op={s_op:.1%}, full={s_full:.1%}")
    print(f"Set Pointer:    op={b_op:.1%}, set_select={b_sel:.1%}, full={b_full:.1%}")
    print(f"Full match improvement: {b_full - s_full:+.1%}")

    elapsed = time.time() - t0
    out = {
        'single_op_acc': s_op, 'single_full': s_full,
        'set_op_acc': b_op, 'set_select_acc': b_sel, 'set_full': b_full,
        'improvement': b_full - s_full,
        'multi_target_ratio': multi_count / N,
        'single_params': params_a, 'set_params': params_b,
        'n_test': total, 'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase18_set_pointer.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics = ['Op Accuracy', 'Full Match']
    single_vals = [s_op, s_full]
    set_vals = [b_op, b_full]
    x = np.arange(len(metrics))
    w = 0.35
    axes[0].bar(x-w/2, single_vals, w, label='Single Pointer', color='#FF9800')
    axes[0].bar(x+w/2, set_vals, w, label='Set Pointer', color='#4CAF50')
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Single vs Set Pointer'); axes[0].legend()

    axes[1].bar(['Set Selection\nAccuracy'], [b_sel], color='#2196F3')
    axes[1].set_ylim(0, 1); axes[1].set_title('Set-Pointer Selection Accuracy')
    axes[1].text(0, b_sel+0.02, f'{b_sel:.1%}', ha='center', fontsize=14)

    plt.suptitle('Phase 18: Broadcast Set-Pointer', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase18_set_pointer.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
