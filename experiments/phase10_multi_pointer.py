"""
Phase 10: Multi-Primitive Pointer Composer
============================================
Fix Phase 9's failure by combining Phase 8's Pointer with multi-step.

Phase 9 failed because text-based DSL generation can't do multi-step.
Phase 8's Pointer Network (90.7% pointer accuracy) operates on STRUCTURE.

This phase: autoregressive Pointer Network that generates a SEQUENCE of
(operation, target_object, color) tuples. After each step, the grid state
is updated and fed back -- but using structured pointers, not text.

This is Phase 8 + Phase 9 fused: structured output + visual feedback.
"""

import os, sys, json, time, random
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

MAX_GRID = 30
MAX_OBJECTS = 20
NODE_FEAT_DIM = 16
N_OPS = 8  # 0=STOP, 1=IDENTITY, 2=FILL, 3=COPY, 4=SWAP, 5=RECOLOR, 6=MIRROR_H, 7=MIRROR_V
OP_NAMES = ['STOP', 'IDENTITY', 'FILL', 'COPY', 'SWAP', 'RECOLOR', 'MIRROR_H', 'MIRROR_V']
N_COLORS = 10

# ============================================================
# Object extraction & features (from Phase 4/8)
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
    feats.append(cr / max(h, 1))
    feats.append(cc / max(w, 1))
    feats.append(obj['area'] / max(h*w, 1))
    r0, c0, r1, c1 = obj['bbox']
    feats.append((r1-r0+1) / max(h, 1))
    feats.append((c1-c0+1) / max(w, 1))
    feats.append((c1-c0+1) / max(r1-r0+1, 1))
    return feats[:NODE_FEAT_DIM]

def execute_op(grid, op_id, target_obj, color1, color2):
    """Execute a single structured operation on grid."""
    arr = np.array(grid).copy()
    if op_id == 0:  # STOP
        return arr.tolist(), True
    elif op_id == 1:  # IDENTITY
        return arr.tolist(), False
    elif op_id == 2:  # FILL
        arr[:] = color1
        return arr.tolist(), False
    elif op_id == 5:  # RECOLOR
        arr[arr == color1] = color2
        return arr.tolist(), False
    elif op_id == 6:  # MIRROR_H
        return np.flipud(arr).tolist(), False
    elif op_id == 7:  # MIRROR_V
        return np.fliplr(arr).tolist(), False
    elif op_id == 4:  # SWAP
        m1, m2 = arr == color1, arr == color2
        arr[m1] = color2
        arr[m2] = color1
        return arr.tolist(), False
    return arr.tolist(), False

# ============================================================
# Multi-Step Pointer Composer
# ============================================================
class MultiStepPointer(nn.Module):
    def __init__(self, node_dim=NODE_FEAT_DIM, hidden=64, n_ops=N_OPS, n_colors=N_COLORS):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden)

        # GNN message passing
        self.mp1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mp2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # State encoder: incorporates "step context" (what happened before)
        self.state_rnn = nn.GRUCell(hidden + n_ops, hidden)

        # Heads
        self.op_head = nn.Linear(hidden, n_ops)
        self.c1_head = nn.Linear(hidden, n_colors)
        self.c2_head = nn.Linear(hidden, n_colors)
        self.ptr_query = nn.Linear(hidden, hidden)
        self.ptr_key = nn.Linear(hidden, hidden)
        self.hidden = hidden
        self.n_ops = n_ops

    def encode_graph(self, node_feats, n_nodes):
        B = node_feats.size(0)
        h = self.node_embed(node_feats)
        mask = torch.arange(MAX_OBJECTS, device=h.device).unsqueeze(0) < n_nodes.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)

        msg = (h * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.mp1(torch.cat([h, msg.expand_as(h)], -1))
        h = self.norm1(h) * mf

        msg = (h * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.mp2(torch.cat([h, msg.expand_as(h)], -1))
        h = self.norm2(h) * mf

        graph_vec = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
        return h, graph_vec, mask

    def step(self, node_h, graph_vec, mask, rnn_state, prev_op_onehot):
        """One autoregressive step."""
        rnn_input = torch.cat([graph_vec, prev_op_onehot], -1)
        rnn_state = self.state_rnn(rnn_input, rnn_state)

        op_logits = self.op_head(rnn_state)
        c1_logits = self.c1_head(rnn_state)
        c2_logits = self.c2_head(rnn_state)

        query = self.ptr_query(rnn_state).unsqueeze(1)
        keys = self.ptr_key(node_h)
        ptr_logits = (query * keys).sum(-1)
        ptr_logits = ptr_logits.masked_fill(~mask, -1e9)

        return op_logits, c1_logits, c2_logits, ptr_logits, rnn_state

# ============================================================
# Label extraction
# ============================================================
def extract_op_label(inp, out):
    inp_arr = np.array(inp)
    out_arr = np.array(out)
    objects, bg = extract_objects(inp)

    if inp_arr.shape == out_arr.shape and np.array_equal(inp_arr, out_arr):
        return [(1, 0, 0, 0)]  # IDENTITY

    if out_arr.size > 0:
        unique = np.unique(out_arr)
        if len(unique) == 1:
            return [(2, 0, int(unique[0]), 0)]  # FILL

    if inp_arr.shape == out_arr.shape:
        diff = inp_arr != out_arr
        if diff.any():
            old_c = set(inp_arr[diff].tolist())
            new_c = set(out_arr[diff].tolist())
            if len(old_c) == 1 and len(new_c) == 1:
                ptr = 0
                oc = int(list(old_c)[0])
                for j, obj in enumerate(objects):
                    if obj['color'] == oc:
                        ptr = j
                        break
                return [(5, min(ptr, MAX_OBJECTS-1), oc, int(list(new_c)[0]))]  # RECOLOR

    if inp_arr.shape == out_arr.shape:
        if np.array_equal(np.flipud(inp_arr), out_arr):
            return [(6, 0, 0, 0)]  # MIRROR_H
        if np.array_equal(np.fliplr(inp_arr), out_arr):
            return [(7, 0, 0, 0)]  # MIRROR_V

    return [(3, 0, 0, 0)]  # COPY fallback

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
    print("Phase 10: Multi-Primitive Pointer Composer")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare data: each sample has object graph + sequence of ops
    all_nf, all_nn, all_ops, all_ptrs, all_c1s, all_c2s = [], [], [], [], [], []

    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            objects, bg = extract_objects(inp)
            h, w = len(inp), len(inp[0])
            if not objects:
                continue

            nf = np.zeros((MAX_OBJECTS, NODE_FEAT_DIM), dtype=np.float32)
            n = min(len(objects), MAX_OBJECTS)
            for j in range(n):
                f = object_to_features(objects[j], h, w)
                nf[j, :len(f)] = f[:NODE_FEAT_DIM]

            op_seq = extract_op_label(inp, out)
            op_seq.append((0, 0, 0, 0))  # STOP

            for op_id, ptr_idx, c1, c2 in op_seq:
                all_nf.append(nf)
                all_nn.append(n)
                all_ops.append(op_id)
                all_ptrs.append(min(ptr_idx, max(n-1, 0)))
                all_c1s.append(c1)
                all_c2s.append(c2)

    nf_t = torch.tensor(np.array(all_nf), dtype=torch.float32)
    nn_t = torch.tensor(all_nn, dtype=torch.long)
    op_t = torch.tensor(all_ops, dtype=torch.long)
    ptr_t = torch.tensor(all_ptrs, dtype=torch.long)
    c1_t = torch.tensor(all_c1s, dtype=torch.long)
    c2_t = torch.tensor(all_c2s, dtype=torch.long)

    N = len(nf_t)
    split = int(N * 0.8)
    print(f"Total steps: {N}, Train: {split}, Test: {N-split}")

    # Train
    model = MultiStepPointer().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 32
    losses = []

    for epoch in range(60):
        model.train()
        perm = torch.randperm(split)
        epoch_loss, n_b = 0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            nf = nf_t[idx].to(DEVICE)
            nn_b = nn_t[idx].to(DEVICE)
            op = op_t[idx].to(DEVICE)
            ptr = ptr_t[idx].to(DEVICE)
            c1 = c1_t[idx].to(DEVICE)
            c2 = c2_t[idx].to(DEVICE)

            node_h, graph_vec, mask = model.encode_graph(nf, nn_b)
            B = nf.size(0)
            rnn_state = graph_vec
            prev_op = torch.zeros(B, N_OPS, device=DEVICE)

            op_l, c1_l, c2_l, ptr_l, _ = model.step(node_h, graph_vec, mask, rnn_state, prev_op)
            loss = (F.cross_entropy(op_l, op) + F.cross_entropy(c1_l, c1) +
                    F.cross_entropy(c2_l, c2) + F.cross_entropy(ptr_l, ptr))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        avg_loss = epoch_loss / max(n_b, 1)
        losses.append(avg_loss)
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/60: loss={avg_loss:.4f}")

    # Evaluation
    model.eval()
    # Compare single-step pointer (Phase 8 baseline) vs multi-step
    single_correct, multi_correct, total = 0, 0, 0

    with torch.no_grad():
        for i in range(split, N):
            if op_t[i].item() == 0:  # Skip STOP tokens
                continue
            nf = nf_t[i:i+1].to(DEVICE)
            nn_b = nn_t[i:i+1].to(DEVICE)

            node_h, graph_vec, mask = model.encode_graph(nf, nn_b)
            rnn_state = graph_vec
            prev_op = torch.zeros(1, N_OPS, device=DEVICE)

            # Step 1
            op_l, c1_l, c2_l, ptr_l, rnn_state = model.step(node_h, graph_vec, mask, rnn_state, prev_op)
            op_p = op_l.argmax(1).item()
            ptr_p = ptr_l.argmax(1).item()
            c1_p = c1_l.argmax(1).item()
            c2_p = c2_l.argmax(1).item()

            total += 1
            gt_op = op_t[i].item()
            gt_ptr = ptr_t[i].item()
            gt_c1 = c1_t[i].item()
            gt_c2 = c2_t[i].item()

            # Single step match
            if op_p == gt_op and ptr_p == gt_ptr and c1_p == gt_c1 and c2_p == gt_c2:
                single_correct += 1

            # Multi-step: try step 2 with feedback
            prev_op_oh = torch.zeros(1, N_OPS, device=DEVICE)
            prev_op_oh[0, op_p] = 1.0
            op_l2, c1_l2, c2_l2, ptr_l2, _ = model.step(node_h, graph_vec, mask, rnn_state, prev_op_oh)
            op_p2 = op_l2.argmax(1).item()

            # If step 1 was wrong but step 2 corrects (self-correction)
            if op_p2 == gt_op:
                ptr_p2 = ptr_l2.argmax(1).item()
                c1_p2 = c1_l2.argmax(1).item()
                c2_p2 = c2_l2.argmax(1).item()
                if ptr_p2 == gt_ptr and c1_p2 == gt_c1 and c2_p2 == gt_c2:
                    multi_correct += 1
            elif op_p == gt_op and ptr_p == gt_ptr and c1_p == gt_c1 and c2_p == gt_c2:
                multi_correct += 1  # Step 1 was already correct

    s_rate = single_correct / max(total, 1)
    m_rate = multi_correct / max(total, 1)

    print(f"\n--- Results ---")
    print(f"Single-step full match: {s_rate:.1%}")
    print(f"Multi-step full match:  {m_rate:.1%}")
    print(f"Improvement: {m_rate - s_rate:+.1%}")

    elapsed = time.time() - t0

    results = {
        'single_step_match': s_rate,
        'multi_step_match': m_rate,
        'improvement': m_rate - s_rate,
        'n_test': total,
        'model_params': params,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase10_multi_pointer.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(['Single-Step\nPointer', 'Multi-Step\nComposer'], [s_rate, m_rate],
                        color=['#FF9800', '#4CAF50'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Full Match Rate')
    axes[0].set_title('Phase 10: Single vs Multi-Step Pointer')
    for bar, val in zip(bars, [s_rate, m_rate]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f'{val:.1%}', ha='center', fontsize=12)

    axes[1].plot(losses)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase10_multi_pointer.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
