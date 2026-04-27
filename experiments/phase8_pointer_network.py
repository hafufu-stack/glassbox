"""
Phase 8: Attention-as-Pointer Network
=======================================
Fuse Phase 4 (Object Graph) + Phase 6 (Attention Grounding).

Instead of generating coordinates as text tokens (hallucination-prone),
the model:
1. Predicts operation type (RECOLOR, FILL, MIRROR, etc.) via classification
2. Predicts parameters (colors) via classification
3. Selects TARGET OBJECT by pointing at object graph nodes via attention

The "pointer" is literally the cross-attention weight over object nodes.
No text generation of coordinates = zero coordinate hallucination.
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

MAX_OBJECTS = 20
NODE_FEAT_DIM = 16
N_OPS = 7  # IDENTITY, FILL, COPY, SWAP, RECOLOR, MIRROR_H, MIRROR_V
OP_NAMES = ['IDENTITY', 'FILL', 'COPY', 'SWAP', 'RECOLOR', 'MIRROR_H', 'MIRROR_V']
N_COLORS = 10

# ============================================================
# Object extraction
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

# ============================================================
# DSL label extraction
# ============================================================
def extract_dsl_label(inp, out):
    """Extract (op_id, color1, color2, target_obj_idx) from input-output pair."""
    inp_arr = np.array(inp)
    out_arr = np.array(out)
    objects, bg = extract_objects(inp)

    # IDENTITY
    if inp_arr.shape == out_arr.shape and np.array_equal(inp_arr, out_arr):
        return 0, 0, 0, 0  # IDENTITY

    # FILL
    if out_arr.size > 0:
        unique = np.unique(out_arr)
        if len(unique) == 1:
            return 1, int(unique[0]), 0, 0  # FILL color

    # RECOLOR
    if inp_arr.shape == out_arr.shape:
        diff = inp_arr != out_arr
        if diff.any():
            old_c = set(inp_arr[diff].tolist())
            new_c = set(out_arr[diff].tolist())
            if len(old_c) == 1 and len(new_c) == 1:
                old_color = int(list(old_c)[0])
                new_color = int(list(new_c)[0])
                # Find which object has old_color
                target_idx = 0
                for j, obj in enumerate(objects):
                    if obj['color'] == old_color:
                        target_idx = j
                        break
                return 4, old_color, new_color, min(target_idx, MAX_OBJECTS-1)

    # MIRROR_H
    if inp_arr.shape == out_arr.shape and np.array_equal(np.flipud(inp_arr), out_arr):
        return 5, 0, 0, 0

    # MIRROR_V
    if inp_arr.shape == out_arr.shape and np.array_equal(np.fliplr(inp_arr), out_arr):
        return 6, 0, 0, 0

    return 2, 0, 0, 0  # COPY fallback

# ============================================================
# Pointer Network Model
# ============================================================
class PointerNetwork(nn.Module):
    def __init__(self, node_dim=NODE_FEAT_DIM, hidden=64, n_ops=N_OPS, n_colors=N_COLORS):
        super().__init__()
        # Object encoder (GNN-like)
        self.node_embed = nn.Linear(node_dim, hidden)
        self.mp1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mp2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # Graph pool -> global context
        self.graph_pool = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

        # Prediction heads
        self.op_head = nn.Linear(hidden, n_ops)        # Which operation
        self.color1_head = nn.Linear(hidden, n_colors)  # Source color
        self.color2_head = nn.Linear(hidden, n_colors)  # Target color

        # Pointer: attention over object nodes for target selection
        self.pointer_query = nn.Linear(hidden, hidden)
        self.pointer_key = nn.Linear(hidden, hidden)

        self.hidden = hidden

    def forward(self, node_feats, n_nodes):
        """
        node_feats: (B, MAX_OBJ, NODE_FEAT_DIM)
        n_nodes: (B,)
        Returns: op_logits, color1_logits, color2_logits, pointer_logits
        """
        B = node_feats.size(0)
        h = self.node_embed(node_feats)  # (B, MAX_OBJ, hidden)

        mask = torch.arange(MAX_OBJECTS, device=h.device).unsqueeze(0) < n_nodes.unsqueeze(1)
        mask_f = mask.float().unsqueeze(-1)

        # Message passing 1
        msg = (h * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        h = h + self.mp1(torch.cat([h, msg.expand_as(h)], dim=-1))
        h = self.norm1(h) * mask_f

        # Message passing 2
        msg = (h * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        h = h + self.mp2(torch.cat([h, msg.expand_as(h)], dim=-1))
        h = self.norm2(h) * mask_f

        # Global context
        graph_vec = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        ctx = self.graph_pool(graph_vec)  # (B, hidden)

        # Predictions
        op_logits = self.op_head(ctx)
        c1_logits = self.color1_head(ctx)
        c2_logits = self.color2_head(ctx)

        # Pointer: which object to target
        query = self.pointer_query(ctx).unsqueeze(1)  # (B, 1, hidden)
        keys = self.pointer_key(h)                      # (B, MAX_OBJ, hidden)
        pointer_logits = (query * keys).sum(dim=-1)     # (B, MAX_OBJ)
        pointer_logits = pointer_logits.masked_fill(~mask, -1e9)

        return op_logits, c1_logits, c2_logits, pointer_logits

# ============================================================
# Data preparation
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
    print("Phase 8: Attention-as-Pointer Network")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare dataset
    all_node_feats = []
    all_n_nodes = []
    all_op_labels = []
    all_c1_labels = []
    all_c2_labels = []
    all_ptr_labels = []

    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            objects, bg = extract_objects(inp)
            h, w = len(inp), len(inp[0])

            if not objects:
                continue

            # Node features
            nf = np.zeros((MAX_OBJECTS, NODE_FEAT_DIM), dtype=np.float32)
            n = min(len(objects), MAX_OBJECTS)
            for j in range(n):
                feats = object_to_features(objects[j], h, w)
                nf[j, :len(feats)] = feats[:NODE_FEAT_DIM]

            # Labels
            op_id, c1, c2, ptr_idx = extract_dsl_label(inp, out)

            all_node_feats.append(nf)
            all_n_nodes.append(n)
            all_op_labels.append(op_id)
            all_c1_labels.append(c1)
            all_c2_labels.append(c2)
            all_ptr_labels.append(min(ptr_idx, n-1))

    nf_t = torch.tensor(np.array(all_node_feats), dtype=torch.float32)
    nn_t = torch.tensor(all_n_nodes, dtype=torch.long)
    op_t = torch.tensor(all_op_labels, dtype=torch.long)
    c1_t = torch.tensor(all_c1_labels, dtype=torch.long)
    c2_t = torch.tensor(all_c2_labels, dtype=torch.long)
    ptr_t = torch.tensor(all_ptr_labels, dtype=torch.long)

    N = len(nf_t)
    split = int(N * 0.8)
    print(f"Total: {N}, Train: {split}, Test: {N-split}")

    # Train
    model = PointerNetwork().to(DEVICE)
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
            c1 = c1_t[idx].to(DEVICE)
            c2 = c2_t[idx].to(DEVICE)
            ptr = ptr_t[idx].to(DEVICE)

            op_l, c1_l, c2_l, ptr_l = model(nf, nn_b)
            loss = (F.cross_entropy(op_l, op) +
                    F.cross_entropy(c1_l, c1) +
                    F.cross_entropy(c2_l, c2) +
                    F.cross_entropy(ptr_l, ptr))
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
    op_correct, c1_correct, c2_correct, ptr_correct, total = 0, 0, 0, 0, 0
    full_correct = 0

    with torch.no_grad():
        for i in range(split, N):
            nf = nf_t[i:i+1].to(DEVICE)
            nn_b = nn_t[i:i+1].to(DEVICE)
            op_l, c1_l, c2_l, ptr_l = model(nf, nn_b)

            op_p = op_l.argmax(1).item()
            c1_p = c1_l.argmax(1).item()
            c2_p = c2_l.argmax(1).item()
            ptr_p = ptr_l.argmax(1).item()

            total += 1
            if op_p == op_t[i].item(): op_correct += 1
            if c1_p == c1_t[i].item(): c1_correct += 1
            if c2_p == c2_t[i].item(): c2_correct += 1
            if ptr_p == ptr_t[i].item(): ptr_correct += 1
            if (op_p == op_t[i].item() and c1_p == c1_t[i].item() and
                c2_p == c2_t[i].item() and ptr_p == ptr_t[i].item()):
                full_correct += 1

    op_acc = op_correct / max(total, 1)
    c1_acc = c1_correct / max(total, 1)
    c2_acc = c2_correct / max(total, 1)
    ptr_acc = ptr_correct / max(total, 1)
    full_acc = full_correct / max(total, 1)

    print(f"\n--- Results ---")
    print(f"Operation accuracy: {op_acc:.1%}")
    print(f"Color1 accuracy:    {c1_acc:.1%}")
    print(f"Color2 accuracy:    {c2_acc:.1%}")
    print(f"Pointer accuracy:   {ptr_acc:.1%}")
    print(f"Full match (all correct): {full_acc:.1%}")

    elapsed = time.time() - t0

    results = {
        'op_accuracy': op_acc,
        'color1_accuracy': c1_acc,
        'color2_accuracy': c2_acc,
        'pointer_accuracy': ptr_acc,
        'full_match': full_acc,
        'model_params': params,
        'n_test': total,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase8_pointer_network.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    metrics = ['Operation', 'Color1', 'Color2', 'Pointer', 'Full Match']
    values = [op_acc, c1_acc, c2_acc, ptr_acc, full_acc]
    colors = ['#2196F3', '#FF9800', '#FF9800', '#9C27B0', '#4CAF50']
    bars = axes[0].bar(metrics, values, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Phase 8: Pointer Network Component Accuracy')
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f'{val:.1%}', ha='center', fontsize=10)

    axes[1].plot(losses)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase8_pointer_network.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
