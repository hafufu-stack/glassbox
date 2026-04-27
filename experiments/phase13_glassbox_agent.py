"""
Phase 13: The GlassBox Agent (Unified Architecture)
=====================================================
Fuse the 4 best components from Phase 1-12 into ONE model:

  Phase 4: Object Graph Extraction (pixel -> object nodes)
  Phase 2: Dynamic Router (select best neural module per task)
  Phase 8: Pointer Network (attention = variable, no hallucination)
  Phase 12: Concept Directions (steering vectors as features)

Pipeline:
  Input Grid -> Extract Objects -> GNN Encode -> Dynamic Route ->
  Pointer Decode (op + target + color) -> Output

This is the first "full stack" GlassBox model.
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
N_OPS = 8
OP_NAMES = ['STOP', 'IDENTITY', 'FILL', 'COPY', 'SWAP', 'RECOLOR', 'MIRROR_H', 'MIRROR_V']
N_COLORS = 10
CONCEPT_DIM = 7
MAX_GRID = 30

# ============================================================
# Shared utilities (from earlier phases)
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
                color = arr[r, c]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == color:
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

# ============================================================
# GlassBox Agent: Unified Architecture
# ============================================================
class GlassBoxAgent(nn.Module):
    """Unified: Object GNN + Dynamic Router + Pointer + Concept Features."""
    def __init__(self, node_dim=NODE_FEAT_DIM, concept_dim=CONCEPT_DIM,
                 hidden=64, n_ops=N_OPS, n_colors=N_COLORS):
        super().__init__()
        self.hidden = hidden

        # === Stage 1: Object Graph Encoder (Phase 4) ===
        self.node_embed = nn.Linear(node_dim, hidden)
        self.gnn_layers = nn.ModuleList()
        for _ in range(3):
            self.gnn_layers.append(nn.ModuleDict({
                'msg': nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                'upd': nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                'norm': nn.LayerNorm(hidden),
            }))

        # === Stage 2: Concept Injection (Phase 12) ===
        self.concept_proj = nn.Linear(concept_dim, hidden)

        # === Stage 3: Dynamic Router (Phase 2) ===
        # Three specialist processing heads
        self.relation_head = nn.Sequential(   # For relational reasoning
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.symmetry_head = nn.Sequential(   # For symmetry detection
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.transform_head = nn.Sequential(  # For color/fill transforms
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.router = nn.Sequential(
            nn.Linear(hidden + concept_dim, 32), nn.ReLU(), nn.Linear(32, 3))

        # === Stage 4: Pointer Decoder (Phase 8) ===
        self.op_head = nn.Linear(hidden, n_ops)
        self.c1_head = nn.Linear(hidden, n_colors)
        self.c2_head = nn.Linear(hidden, n_colors)
        self.ptr_query = nn.Linear(hidden, hidden)
        self.ptr_key = nn.Linear(hidden, hidden)

    def forward(self, node_feats, n_nodes, concepts):
        B = node_feats.size(0)
        mask = torch.arange(MAX_OBJECTS, device=node_feats.device).unsqueeze(0) < n_nodes.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)

        # Stage 1: GNN encode
        h = self.node_embed(node_feats)
        for layer in self.gnn_layers:
            msg_agg = (h * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
            msgs = layer['msg'](torch.cat([h, msg_agg.expand_as(h)], -1))
            h = h + layer['upd'](torch.cat([h, msgs], -1))
            h = layer['norm'](h) * mf

        graph_vec = (h * mf).sum(1) / mf.sum(1).clamp(min=1)  # (B, hidden)

        # Stage 2: Concept injection
        concept_vec = self.concept_proj(concepts)  # (B, hidden)
        graph_vec = graph_vec + concept_vec  # Additive steering

        # Stage 3: Dynamic routing
        route_input = torch.cat([graph_vec, concepts], -1)
        route_weights = F.softmax(self.router(route_input), dim=-1)  # (B, 3)

        routed = (route_weights[:, 0:1] * self.relation_head(graph_vec) +
                  route_weights[:, 1:2] * self.symmetry_head(graph_vec) +
                  route_weights[:, 2:3] * self.transform_head(graph_vec))

        ctx = graph_vec + routed  # Residual

        # Stage 4: Pointer decode
        op_logits = self.op_head(ctx)
        c1_logits = self.c1_head(ctx)
        c2_logits = self.c2_head(ctx)

        query = self.ptr_query(ctx).unsqueeze(1)
        keys = self.ptr_key(h)
        ptr_logits = (query * keys).sum(-1)
        ptr_logits = ptr_logits.masked_fill(~mask, -1e9)

        return op_logits, c1_logits, c2_logits, ptr_logits, route_weights

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
    print("Phase 13: The GlassBox Agent (Unified Architecture)")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare data
    all_nf, all_nn, all_concepts = [], [], []
    all_ops, all_ptrs, all_c1s, all_c2s = [], [], [], []

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

            op_id, ptr_idx, c1, c2 = extract_op_label(inp, out)

            all_nf.append(nf)
            all_nn.append(n)
            all_concepts.append(extract_concepts(inp))
            all_ops.append(op_id)
            all_ptrs.append(min(ptr_idx, max(n-1, 0)))
            all_c1s.append(c1)
            all_c2s.append(c2)

    nf_t = torch.tensor(np.array(all_nf), dtype=torch.float32)
    nn_t = torch.tensor(all_nn, dtype=torch.long)
    con_t = torch.tensor(np.array(all_concepts), dtype=torch.float32)
    op_t = torch.tensor(all_ops, dtype=torch.long)
    ptr_t = torch.tensor(all_ptrs, dtype=torch.long)
    c1_t = torch.tensor(all_c1s, dtype=torch.long)
    c2_t = torch.tensor(all_c2s, dtype=torch.long)

    N = len(nf_t)
    split = int(N * 0.8)
    print(f"Total: {N}, Train: {split}, Test: {N-split}")

    # Train
    model = GlassBoxAgent().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"GlassBox Agent parameters: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 32
    losses = []

    for epoch in range(80):
        model.train()
        perm = torch.randperm(split)
        epoch_loss, n_b = 0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            op_l, c1_l, c2_l, ptr_l, _ = model(
                nf_t[idx].to(DEVICE), nn_t[idx].to(DEVICE), con_t[idx].to(DEVICE))
            loss = (F.cross_entropy(op_l, op_t[idx].to(DEVICE)) +
                    F.cross_entropy(c1_l, c1_t[idx].to(DEVICE)) +
                    F.cross_entropy(c2_l, c2_t[idx].to(DEVICE)) +
                    F.cross_entropy(ptr_l, ptr_t[idx].to(DEVICE)))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        losses.append(epoch_loss / max(n_b, 1))
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/80: loss={losses[-1]:.4f}")

    # Evaluate
    model.eval()
    op_ok, c1_ok, c2_ok, ptr_ok, full_ok, total = 0, 0, 0, 0, 0, 0
    route_accum = np.zeros(3)

    with torch.no_grad():
        for i in range(split, N):
            op_l, c1_l, c2_l, ptr_l, rw = model(
                nf_t[i:i+1].to(DEVICE), nn_t[i:i+1].to(DEVICE), con_t[i:i+1].to(DEVICE))
            op_p = op_l.argmax(1).item()
            c1_p = c1_l.argmax(1).item()
            c2_p = c2_l.argmax(1).item()
            ptr_p = ptr_l.argmax(1).item()
            total += 1
            if op_p == op_t[i].item(): op_ok += 1
            if c1_p == c1_t[i].item(): c1_ok += 1
            if c2_p == c2_t[i].item(): c2_ok += 1
            if ptr_p == ptr_t[i].item(): ptr_ok += 1
            if (op_p == op_t[i].item() and c1_p == c1_t[i].item() and
                c2_p == c2_t[i].item() and ptr_p == ptr_t[i].item()):
                full_ok += 1
            route_accum += rw[0].cpu().numpy()

    op_acc = op_ok / max(total, 1)
    c1_acc = c1_ok / max(total, 1)
    c2_acc = c2_ok / max(total, 1)
    ptr_acc = ptr_ok / max(total, 1)
    full_acc = full_ok / max(total, 1)
    route_avg = route_accum / max(total, 1)
    route_names = ['Relation', 'Symmetry', 'Transform']

    print(f"\n--- Results ---")
    print(f"Operation: {op_acc:.1%}")
    print(f"Color1:    {c1_acc:.1%}")
    print(f"Color2:    {c2_acc:.1%}")
    print(f"Pointer:   {ptr_acc:.1%}")
    print(f"Full Match: {full_acc:.1%}")
    print(f"Route weights: {dict(zip(route_names, [f'{w:.2f}' for w in route_avg]))}")

    elapsed = time.time() - t0

    results = {
        'op_accuracy': op_acc, 'color1_accuracy': c1_acc,
        'color2_accuracy': c2_acc, 'pointer_accuracy': ptr_acc,
        'full_match': full_acc,
        'route_weights': dict(zip(route_names, route_avg.tolist())),
        'model_params': params, 'n_test': total,
        'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase13_glassbox_agent.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ['Op', 'Color1', 'Color2', 'Pointer', 'Full']
    vals = [op_acc, c1_acc, c2_acc, ptr_acc, full_acc]
    colors = ['#2196F3', '#FF9800', '#FF9800', '#9C27B0', '#4CAF50']
    bars = axes[0].bar(metrics, vals, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('GlassBox Agent: Component Accuracy')
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.0%}', ha='center')

    axes[1].bar(route_names, route_avg, color=['#2196F3', '#9C27B0', '#FF9800'])
    axes[1].set_title('Dynamic Route Distribution')
    axes[1].set_ylabel('Average Weight')

    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss'); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 13: The GlassBox Agent (Unified Architecture)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase13_glassbox_agent.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
