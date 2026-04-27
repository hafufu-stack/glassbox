"""
Phase 4: Object-Centric Graph Extraction
==========================================
Fix the n_objects blindspot (28.7% in Phase 3).

Instead of feeding raw pixels to a conv net, we:
1. Extract connected components (objects) from the grid
2. Build each object as a graph node with features (color, position, size, shape)
3. Run a GNN on the object graph
4. Re-probe with linear probes to test if n_objects recognition improves

This directly addresses Phase 2's finding that the router prefers GNN (76%)
and Phase 3's finding that object counting is the model's worst concept.
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
NODE_FEAT_DIM = 16  # Features per object node

# ============================================================
# Connected Component Extraction
# ============================================================
def extract_objects(grid):
    """Extract connected components from grid as objects.
    
    Returns list of dicts with:
      - color: int
      - pixels: list of (r,c) tuples
      - bbox: (r_min, c_min, r_max, c_max)
      - area: int
      - center: (r_center, c_center)
    """
    arr = np.array(grid)
    h, w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    bg_color = int(np.bincount(arr.flatten()).argmax())
    
    objects = []
    for r in range(h):
        for c in range(w):
            if not visited[r, c] and arr[r, c] != bg_color:
                # BFS flood fill
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
                obj = {
                    'color': color,
                    'pixels': pixels,
                    'bbox': (min(rows), min(cols), max(rows), max(cols)),
                    'area': len(pixels),
                    'center': (np.mean(rows), np.mean(cols)),
                }
                objects.append(obj)
    
    return objects, bg_color

def object_to_features(obj, grid_h, grid_w):
    """Convert object dict to normalized feature vector."""
    feats = []
    # Color (one-hot, 10 colors)
    color_oh = [0.0] * 10
    color_oh[min(obj['color'], 9)] = 1.0
    
    # Normalized position
    cr, cc = obj['center']
    feats.append(cr / max(grid_h, 1))
    feats.append(cc / max(grid_w, 1))
    
    # Normalized area
    feats.append(obj['area'] / max(grid_h * grid_w, 1))
    
    # Aspect ratio of bounding box
    r0, c0, r1, c1 = obj['bbox']
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    feats.append(bh / max(grid_h, 1))
    feats.append(bw / max(grid_w, 1))
    feats.append(bw / max(bh, 1))  # aspect ratio
    
    return color_oh + feats  # 10 + 6 = 16 dims

# ============================================================
# Object Graph GNN
# ============================================================
class ObjectGNN(nn.Module):
    def __init__(self, node_dim=NODE_FEAT_DIM, hidden=64, n_layers=3, n_colors=11):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mp_layers.append(nn.ModuleDict({
                'edge_fn': nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                'node_fn': nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                'norm': nn.LayerNorm(hidden),
            }))
        
        # Graph-level readout for grid prediction
        self.graph_readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, MAX_GRID * MAX_GRID * n_colors),
        )
        
        self.hidden = hidden
        self.n_colors = n_colors
    
    def forward(self, node_features, n_nodes, capture_hidden=False):
        """
        node_features: (B, MAX_OBJECTS, NODE_FEAT_DIM) padded
        n_nodes: (B,) actual number of nodes per sample
        """
        B = node_features.size(0)
        h = self.node_embed(node_features)  # (B, MAX_OBJ, hidden)
        
        hidden_states = {}
        if capture_hidden:
            hidden_states['embed'] = h.detach()
        
        for i, layer in enumerate(self.mp_layers):
            # Create node mask
            mask = torch.arange(MAX_OBJECTS, device=h.device).unsqueeze(0) < n_nodes.unsqueeze(1)
            mask_float = mask.float().unsqueeze(-1)
            
            # Mean aggregation (all-to-all message passing)
            msg_sum = (h * mask_float).sum(dim=1, keepdim=True)
            msg_count = mask_float.sum(dim=1, keepdim=True).clamp(min=1)
            msg_mean = msg_sum / msg_count
            msg_mean = msg_mean.expand_as(h)
            
            # Edge function
            edge_input = torch.cat([h, msg_mean], dim=-1)
            edge_msg = layer['edge_fn'](edge_input)
            
            # Node update
            node_input = torch.cat([h, edge_msg], dim=-1)
            h = h + layer['node_fn'](node_input)
            h = layer['norm'](h)
            h = h * mask_float
            
            if capture_hidden:
                hidden_states[f'mp_{i}'] = h.detach()
        
        # Graph readout: mean pool over valid nodes
        mask_float = (torch.arange(MAX_OBJECTS, device=h.device).unsqueeze(0) < n_nodes.unsqueeze(1)).float().unsqueeze(-1)
        graph_vec = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        
        if capture_hidden:
            hidden_states['graph_pool'] = graph_vec.detach()
            self._hidden_states = hidden_states
        
        out = self.graph_readout(graph_vec)
        return out.view(B, MAX_GRID, MAX_GRID, self.n_colors).permute(0, 3, 1, 2)

# ============================================================
# Linear Probe (same as Phase 3)
# ============================================================
class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.linear(x)

# ============================================================
# Concept labels (same as Phase 3)
# ============================================================
def count_objects_label(grid):
    objs, _ = extract_objects(grid)
    return min(len(objs), 9)

def has_symmetry(grid):
    arr = np.array(grid)
    h_sym = np.array_equal(arr, np.flipud(arr))
    v_sym = np.array_equal(arr, np.fliplr(arr))
    if h_sym and v_sym: return 3
    elif h_sym: return 1
    elif v_sym: return 2
    return 0

def background_color(grid):
    arr = np.array(grid).flatten()
    return int(np.bincount(arr, minlength=10).argmax())

def n_colors_label(grid):
    return min(len(set(c for row in grid for c in row)), 10)

def grid_size_cat(grid):
    mx = max(len(grid), len(grid[0]))
    if mx <= 5: return 0
    elif mx <= 15: return 1
    return 2

CONCEPTS = {
    'n_objects': (count_objects_label, 10),
    'symmetry': (has_symmetry, 4),
    'bg_color': (background_color, 10),
    'n_colors': (n_colors_label, 11),
    'grid_size': (grid_size_cat, 3),
}

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

def pad_grid(grid, max_h=MAX_GRID, max_w=MAX_GRID, pad_val=10):
    h, w = len(grid), len(grid[0])
    padded = np.full((max_h, max_w), pad_val, dtype=np.int64)
    padded[:h, :w] = np.array(grid)
    return padded

def prepare_graph_data(grid):
    """Convert raw grid to object graph representation."""
    objects, bg = extract_objects(grid)
    h, w = len(grid), len(grid[0])
    
    node_feats = np.zeros((MAX_OBJECTS, NODE_FEAT_DIM), dtype=np.float32)
    n_nodes = min(len(objects), MAX_OBJECTS)
    
    for i in range(n_nodes):
        feats = object_to_features(objects[i], h, w)
        node_feats[i, :len(feats)] = feats[:NODE_FEAT_DIM]
    
    return node_feats, n_nodes

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 4: Object-Centric Graph Extraction")
    print("=" * 60)
    t0 = time.time()
    
    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")
    
    # Prepare data
    all_node_feats = []
    all_n_nodes = []
    all_outputs = []
    concept_labels = {name: [] for name in CONCEPTS}
    
    for task in tasks:
        for pair in task.get('train', []):
            inp = pair['input']
            out = pair['output']
            
            nf, nn_count = prepare_graph_data(inp)
            all_node_feats.append(nf)
            all_n_nodes.append(nn_count)
            all_outputs.append(pad_grid(out))
            
            for name, (fn, _) in CONCEPTS.items():
                concept_labels[name].append(fn(inp))
    
    node_feats_t = torch.tensor(np.array(all_node_feats), dtype=torch.float32)
    n_nodes_t = torch.tensor(all_n_nodes, dtype=torch.long)
    outputs_t = torch.tensor(np.array(all_outputs), dtype=torch.long)
    concept_labels_t = {name: torch.tensor(vals, dtype=torch.long) for name, vals in concept_labels.items()}
    
    N = len(node_feats_t)
    split = int(N * 0.8)
    print(f"Total: {N}, Train: {split}, Test: {N-split}")
    
    # Object count distribution
    obj_counts = all_n_nodes
    avg_objects = np.mean(obj_counts)
    print(f"Average objects per grid: {avg_objects:.1f}")
    
    # Train Object GNN
    print("\n--- Training Object GNN ---")
    model = ObjectGNN(node_dim=NODE_FEAT_DIM, hidden=64, n_layers=3).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Object GNN parameters: {params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 32
    losses = []
    
    for epoch in range(40):
        model.train()
        perm = torch.randperm(split)
        epoch_loss = 0
        n_b = 0
        
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            nf = node_feats_t[idx].to(DEVICE)
            nn_c = n_nodes_t[idx].to(DEVICE)
            y = outputs_t[idx].to(DEVICE)
            
            logits = model(nf, nn_c)
            loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_b += 1
        
        avg_loss = epoch_loss / max(n_b, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/40: loss={avg_loss:.4f}")
    
    # Extract hidden states for probing
    print("\n--- Extracting Graph Hidden States ---")
    model.eval()
    all_hidden = {}
    
    with torch.no_grad():
        for i in range(0, N, BATCH):
            nf = node_feats_t[i:i+BATCH].to(DEVICE)
            nn_c = n_nodes_t[i:i+BATCH].to(DEVICE)
            model(nf, nn_c, capture_hidden=True)
            
            for name, h in model._hidden_states.items():
                if h.dim() == 3:
                    # Node-level: mean pool over nodes
                    pooled = h.mean(dim=1)
                else:
                    pooled = h
                if name not in all_hidden:
                    all_hidden[name] = []
                all_hidden[name].append(pooled.cpu())
    
    for name in all_hidden:
        all_hidden[name] = torch.cat(all_hidden[name], dim=0)
    
    layer_names = list(all_hidden.keys())
    print(f"Layers to probe: {layer_names}")
    
    # Linear probing
    print("\n--- Linear Probing (Object GNN) ---")
    probe_results = {}
    
    for concept_name, (_, n_classes) in CONCEPTS.items():
        probe_results[concept_name] = {}
        labels = concept_labels_t[concept_name]
        
        for layer_name in layer_names:
            hidden = all_hidden[layer_name]
            in_dim = hidden.size(1)
            
            h_train, h_test = hidden[:split], hidden[split:]
            l_train, l_test = labels[:split], labels[split:]
            
            probe = LinearProbe(in_dim, n_classes).to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
            
            for ep in range(50):
                probe.train()
                logits = probe(h_train.to(DEVICE))
                loss = F.cross_entropy(logits, l_train.to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            probe.eval()
            with torch.no_grad():
                logits = probe(h_test.to(DEVICE))
                preds = logits.argmax(dim=1)
                acc = (preds == l_test.to(DEVICE)).float().mean().item()
            
            probe_results[concept_name][layer_name] = acc
        
        best_layer = max(probe_results[concept_name], key=probe_results[concept_name].get)
        best_acc = probe_results[concept_name][best_layer]
        print(f"  {concept_name}: best={best_acc:.1%} @ {best_layer}")
    
    # Compare with Phase 3 results
    phase3_path = os.path.join(RESULTS_DIR, 'phase3_concept_probing.json')
    phase3_results = {}
    if os.path.exists(phase3_path):
        with open(phase3_path, 'r', encoding='utf-8') as f:
            p3 = json.load(f)
        for concept in CONCEPTS:
            if concept in p3.get('probe_results', {}):
                phase3_results[concept] = max(p3['probe_results'][concept].values())
    
    elapsed = time.time() - t0
    
    # Save results
    out = {
        'probe_results': probe_results,
        'phase3_comparison': phase3_results,
        'layer_names': layer_names,
        'model_params': params,
        'avg_objects_per_grid': float(avg_objects),
        'n_train': split,
        'n_test': N - split,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase4_object_graph.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    # Plot: comparison with Phase 3
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Bar comparison
    concepts = list(CONCEPTS.keys())
    p4_best = [max(probe_results[c].values()) for c in concepts]
    p3_best = [phase3_results.get(c, 0) for c in concepts]
    
    x = np.arange(len(concepts))
    w = 0.35
    axes[0].bar(x - w/2, p3_best, w, label='Phase 3 (Conv)', color='#FF9800', alpha=0.8)
    axes[0].bar(x + w/2, p4_best, w, label='Phase 4 (Object GNN)', color='#4CAF50', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(concepts, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('Probe Accuracy')
    axes[0].set_title('Phase 3 (Conv) vs Phase 4 (Object GNN)')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # Improvement delta
    deltas = [p4 - p3 for p4, p3 in zip(p4_best, p3_best)]
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
    axes[1].bar(concepts, deltas, color=colors)
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_ylabel('Delta (Phase 4 - Phase 3)')
    axes[1].set_title('Improvement from Object-Centric Representation')
    axes[1].tick_params(axis='x', rotation=30)
    for i, (c, d) in enumerate(zip(concepts, deltas)):
        axes[1].text(i, d + 0.01 * np.sign(d), f'{d:+.1%}', ha='center', fontsize=9)
    
    # Training loss
    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Object GNN Training Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_object_graph.png'), dpi=150)
    plt.close()
    
    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
