"""
Phase 3: Mechanistic Concept Probing
======================================
X-ray the AI's brain: what concepts does it internally represent?

Train a simple model on ARC tasks, then freeze it and attach
Linear Probes to each hidden layer. Each probe tries to predict
human-interpretable concepts from the hidden state alone:

  - n_objects: how many distinct colored objects are in the grid?
  - has_symmetry: is the grid horizontally or vertically symmetric?
  - background_color: what is the most common color?
  - grid_size_category: small (≤5), medium (6-15), large (>15)?
  - n_colors: how many distinct colors are present?

If a probe achieves high accuracy, the model has "learned" that concept
internally. If not, that concept is invisible to the model.

Metrics:
  - probe_accuracy: per-concept, per-layer linear probe accuracy
  - concept_localization: which layers encode which concepts
"""

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
N_COLORS = 11

# ============================================================
# Target model to probe (multi-layer ConvNet for ARC)
# ============================================================
class TargetModel(nn.Module):
    """A multi-layer ConvNet trained on ARC. We'll probe each layer."""
    def __init__(self, channels=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(N_COLORS, channels)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ))
        
        self.readout = nn.Conv2d(channels, N_COLORS, 1)
        self._hidden_states = {}
    
    def forward(self, x, capture_hidden=False):
        """x: (B, H, W) integer grid."""
        h = self.embed(x).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        if capture_hidden:
            self._hidden_states = {}
            self._hidden_states['embed'] = h.detach()
        
        for i, layer in enumerate(self.layers):
            h = h + layer(h)  # Residual
            if capture_hidden:
                self._hidden_states[f'layer_{i}'] = h.detach()
        
        return self.readout(h)

# ============================================================
# Concept label extraction
# ============================================================
def count_objects(grid):
    """Count distinct contiguous colored regions (simple flood fill)."""
    arr = np.array(grid)
    h, w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    bg = int(np.bincount(arr.flatten()).argmax())
    count = 0
    
    for r in range(h):
        for c in range(w):
            if not visited[r, c] and arr[r, c] != bg:
                # BFS flood fill
                count += 1
                stack = [(r, c)]
                color = arr[r, c]
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < h and 0 <= cc < w and not visited[cr, cc] and arr[cr, cc] == color:
                        visited[cr, cc] = True
                        stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
    return min(count, 9)  # Cap at 9 for classification

def has_symmetry(grid):
    """Check horizontal and vertical symmetry."""
    arr = np.array(grid)
    h_sym = np.array_equal(arr, np.flipud(arr))
    v_sym = np.array_equal(arr, np.fliplr(arr))
    if h_sym and v_sym:
        return 3  # Both
    elif h_sym:
        return 1
    elif v_sym:
        return 2
    return 0  # None

def background_color(grid):
    """Most common color."""
    arr = np.array(grid).flatten()
    return int(np.bincount(arr, minlength=10).argmax())

def n_colors(grid):
    """Number of distinct colors."""
    return min(len(set(c for row in grid for c in row)), 10)

def grid_size_cat(grid):
    """0=small(≤5), 1=medium(6-15), 2=large(>15)."""
    h, w = len(grid), len(grid[0])
    mx = max(h, w)
    if mx <= 5: return 0
    elif mx <= 15: return 1
    return 2

CONCEPT_FNS = {
    'n_objects': (count_objects, 10),       # 10 classes (0-9)
    'symmetry': (has_symmetry, 4),          # 4 classes
    'bg_color': (background_color, 10),     # 10 classes
    'n_colors': (n_colors, 11),             # 11 classes
    'grid_size': (grid_size_cat, 3),        # 3 classes
}

# ============================================================
# Linear Probe
# ============================================================
class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
    
    def forward(self, x):
        return self.linear(x)

# ============================================================
# Data loading
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

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 3: Mechanistic Concept Probing")
    print("=" * 60)
    t0 = time.time()
    
    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")
    
    # Step 1: Prepare data with concept labels
    grids = []
    outputs = []
    concepts = {name: [] for name in CONCEPT_FNS}
    
    for task in tasks:
        for pair in task.get('train', []):
            inp = pair['input']
            out = pair['output']
            grids.append(pad_grid(inp))
            outputs.append(pad_grid(out))
            for name, (fn, _) in CONCEPT_FNS.items():
                concepts[name].append(fn(inp))
    
    grids = torch.tensor(np.array(grids), dtype=torch.long)
    outputs = torch.tensor(np.array(outputs), dtype=torch.long)
    concept_labels = {name: torch.tensor(vals, dtype=torch.long) for name, vals in concepts.items()}
    
    N = len(grids)
    split = int(N * 0.8)
    print(f"Total samples: {N}, Train: {split}, Test: {N-split}")
    
    # Step 2: Train the target model
    print("\n--- Training Target Model ---")
    model = TargetModel(channels=64, n_layers=4).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Target model parameters: {params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 32
    
    for epoch in range(30):
        model.train()
        perm = torch.randperm(split)
        epoch_loss = 0
        n_b = 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            x = grids[idx].to(DEVICE)
            y = outputs[idx].to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30: loss={epoch_loss/n_b:.4f}")
    
    # Step 3: Extract hidden states
    print("\n--- Extracting Hidden States ---")
    model.eval()
    all_hidden = {}
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, N, BATCH):
            x = grids[i:i+BATCH].to(DEVICE)
            model(x, capture_hidden=True)
            for name, h in model._hidden_states.items():
                # Global average pool to get (B, C) vector
                pooled = h.mean(dim=[2, 3])  # (B, C)
                if name not in all_hidden:
                    all_hidden[name] = []
                all_hidden[name].append(pooled.cpu())
    
    for name in all_hidden:
        all_hidden[name] = torch.cat(all_hidden[name], dim=0)
    
    layer_names = list(all_hidden.keys())
    print(f"Layers to probe: {layer_names}")
    
    # Step 4: Train linear probes
    print("\n--- Training Linear Probes ---")
    probe_results = {}  # {concept: {layer: accuracy}}
    
    for concept_name, (_, n_classes) in CONCEPT_FNS.items():
        probe_results[concept_name] = {}
        labels = concept_labels[concept_name]
        
        for layer_name in layer_names:
            hidden = all_hidden[layer_name]
            in_dim = hidden.size(1)
            
            # Train/test split
            h_train, h_test = hidden[:split], hidden[split:]
            l_train, l_test = labels[:split], labels[split:]
            
            # Train probe
            probe = LinearProbe(in_dim, n_classes).to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
            
            for ep in range(50):
                probe.train()
                logits = probe(h_train.to(DEVICE))
                loss = F.cross_entropy(logits, l_train.to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            # Evaluate
            probe.eval()
            with torch.no_grad():
                logits = probe(h_test.to(DEVICE))
                preds = logits.argmax(dim=1)
                acc = (preds == l_test.to(DEVICE)).float().mean().item()
            
            probe_results[concept_name][layer_name] = acc
        
        # Print results for this concept
        best_layer = max(probe_results[concept_name], key=probe_results[concept_name].get)
        best_acc = probe_results[concept_name][best_layer]
        print(f"  {concept_name}: best={best_acc:.1%} @ {best_layer}")
    
    elapsed = time.time() - t0
    
    # Save results
    out = {
        'probe_results': probe_results,
        'layer_names': layer_names,
        'concept_names': list(CONCEPT_FNS.keys()),
        'target_model_params': params,
        'n_train': split,
        'n_test': N - split,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase3_concept_probing.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    # Plot: heatmap of concept × layer accuracies
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    concept_names = list(CONCEPT_FNS.keys())
    data_matrix = np.array([[probe_results[c][l] for l in layer_names] for c in concept_names])
    
    im = axes[0].imshow(data_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(layer_names)))
    axes[0].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticks(range(len(concept_names)))
    axes[0].set_yticklabels(concept_names)
    axes[0].set_title('Concept Probe Accuracy (Layer × Concept)')
    plt.colorbar(im, ax=axes[0])
    
    # Add text annotations
    for i in range(len(concept_names)):
        for j in range(len(layer_names)):
            axes[0].text(j, i, f'{data_matrix[i,j]:.0%}', ha='center', va='center',
                        color='black' if data_matrix[i,j] < 0.7 else 'white', fontsize=9)
    
    # Best layer per concept bar chart
    best_accs = [max(probe_results[c].values()) for c in concept_names]
    best_layers = [max(probe_results[c], key=probe_results[c].get) for c in concept_names]
    
    bars = axes[1].barh(concept_names, best_accs, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Best Probe Accuracy')
    axes[1].set_title('Best Concept Decodability')
    for bar, acc, layer in zip(bars, best_accs, best_layers):
        axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{acc:.0%} ({layer})', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase3_concept_probing.png'), dpi=150)
    plt.close()
    
    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
