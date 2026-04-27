"""
Phase 12: Causal Concept Surgery
===================================
The spiritual successor to SNN-Genesis "Aha! Steering Vector".

Phase 3 showed linear probes can decode concepts from hidden states.
Phase 7 showed attention intervention changes outputs causally.

This phase: find the LINEAR DIRECTION in hidden space that encodes
a concept (e.g., "symmetry"), then ADD or SUBTRACT it.

Experiment:
1. Train a model on ARC
2. Extract hidden states, fit linear probes
3. The probe weight vector IS the concept direction
4. At inference: add symmetry vector -> does model start predicting
   symmetric outputs? Subtract it -> does it stop?

This is Representation Engineering / Activation Steering applied
to visual reasoning. If it works, we can literally INSTALL or
REMOVE concepts from the model's brain.
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
# Target model (same architecture as Phase 3)
# ============================================================
class TargetModel(nn.Module):
    def __init__(self, channels=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(N_COLORS, channels)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()))
        self.readout = nn.Conv2d(channels, N_COLORS, 1)

    def forward(self, x):
        h = self.embed(x).permute(0, 3, 1, 2)
        for layer in self.layers:
            h = h + layer(h)
        return self.readout(h)

    def forward_with_intervention(self, x, layer_idx, direction, alpha):
        """Forward with concept steering at a specific layer."""
        h = self.embed(x).permute(0, 3, 1, 2)
        for i, layer in enumerate(self.layers):
            h = h + layer(h)
            if i == layer_idx:
                # Steer: add alpha * direction to hidden state
                # direction: (C,) -> broadcast to (B, C, H, W)
                h = h + alpha * direction.reshape(1, -1, 1, 1)
        return self.readout(h)

# ============================================================
# Concept extraction
# ============================================================
def has_symmetry(grid):
    arr = np.array(grid)
    h_sym = np.array_equal(arr, np.flipud(arr))
    v_sym = np.array_equal(arr, np.fliplr(arr))
    if h_sym and v_sym: return 3
    elif h_sym: return 1
    elif v_sym: return 2
    return 0

def background_color(grid):
    return int(np.bincount(np.array(grid).flatten(), minlength=10).argmax())

# ============================================================
def pad_grid(grid, pad_val=10):
    h, w = len(grid), len(grid[0])
    padded = np.full((MAX_GRID, MAX_GRID), pad_val, dtype=np.int64)
    padded[:h, :w] = np.array(grid)
    return padded

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
    print("Phase 12: Causal Concept Surgery")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare
    grids, outputs, sym_labels, bg_labels = [], [], [], []
    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            grids.append(pad_grid(inp))
            outputs.append(pad_grid(out))
            sym_labels.append(has_symmetry(inp))
            bg_labels.append(background_color(inp))

    grids_t = torch.tensor(np.array(grids), dtype=torch.long)
    outputs_t = torch.tensor(np.array(outputs), dtype=torch.long)
    sym_t = torch.tensor(sym_labels, dtype=torch.long)
    bg_t = torch.tensor(bg_labels, dtype=torch.long)

    N = len(grids_t)
    split = int(N * 0.8)
    BATCH = 32

    # Step 1: Train target model
    print("\n--- Training Target Model ---")
    model = TargetModel().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        model.train()
        perm = torch.randperm(split)
        epoch_loss, n_b = 0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            x = grids_t[idx].to(DEVICE)
            y = outputs_t[idx].to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30: loss={epoch_loss/n_b:.4f}")

    # Step 2: Extract hidden states at each layer
    print("\n--- Extracting Hidden States ---")
    model.eval()
    layer_hiddens = {i: [] for i in range(4)}

    with torch.no_grad():
        for i in range(0, N, BATCH):
            x = grids_t[i:i+BATCH].to(DEVICE)
            h = model.embed(x).permute(0, 3, 1, 2)
            for li, layer in enumerate(model.layers):
                h = h + layer(h)
                pooled = h.mean(dim=[2, 3])
                layer_hiddens[li].append(pooled.cpu())

    for li in layer_hiddens:
        layer_hiddens[li] = torch.cat(layer_hiddens[li], 0)

    # Step 3: Find concept directions via linear probes
    print("\n--- Finding Concept Directions ---")
    concept_directions = {}  # {concept: {layer: direction_vector}}

    for concept_name, labels in [('symmetry', sym_t), ('bg_color', bg_t)]:
        concept_directions[concept_name] = {}
        n_classes = labels.max().item() + 1

        for li in range(4):
            hidden = layer_hiddens[li]
            in_dim = hidden.size(1)

            probe = nn.Linear(in_dim, n_classes).to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-2)

            h_train = hidden[:split].to(DEVICE)
            l_train = labels[:split].to(DEVICE)

            for ep in range(50):
                logits = probe(h_train)
                loss = F.cross_entropy(logits, l_train)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # The concept direction = probe weight vector
            # For binary-ish concepts, use the difference between class weights
            W = probe.weight.data.cpu()  # (n_classes, in_dim)
            if n_classes == 2:
                direction = (W[1] - W[0])  # Direction from class 0 to class 1
            else:
                # Use the principal direction: class with highest variance
                direction = W[W.abs().sum(dim=1).argmax()]

            direction = direction / direction.norm()
            concept_directions[concept_name][li] = direction

            # Probe accuracy
            with torch.no_grad():
                h_test = hidden[split:].to(DEVICE)
                l_test = labels[split:].to(DEVICE)
                acc = (probe(h_test).argmax(1) == l_test).float().mean().item()
            print(f"  {concept_name} @ layer_{li}: probe_acc={acc:.1%}")

    # Step 4: Causal surgery experiment
    print("\n--- Causal Surgery ---")
    surgery_results = {}

    for concept_name in ['symmetry', 'bg_color']:
        surgery_results[concept_name] = {}

        for li in range(4):
            direction = concept_directions[concept_name][li].to(DEVICE)

            baseline_pa, add_pa, sub_pa = 0, 0, 0
            total_pixels = 0

            with torch.no_grad():
                for i in range(split, N):
                    x = grids_t[i:i+1].to(DEVICE)
                    y = outputs_t[i:i+1].to(DEVICE)

                    # Baseline
                    logits_base = model(x)
                    base_correct = (logits_base.argmax(1) == y).sum().item()

                    # Add concept (+3.0 strength)
                    logits_add = model.forward_with_intervention(x, li, direction, alpha=3.0)
                    add_correct = (logits_add.argmax(1) == y).sum().item()

                    # Subtract concept (-3.0 strength)
                    logits_sub = model.forward_with_intervention(x, li, direction, alpha=-3.0)
                    sub_correct = (logits_sub.argmax(1) == y).sum().item()

                    pixels = y.numel()
                    baseline_pa += base_correct
                    add_pa += add_correct
                    sub_pa += sub_correct
                    total_pixels += pixels

            baseline_pa /= max(total_pixels, 1)
            add_pa /= max(total_pixels, 1)
            sub_pa /= max(total_pixels, 1)

            surgery_results[concept_name][f'layer_{li}'] = {
                'baseline': baseline_pa,
                'add': add_pa,
                'subtract': sub_pa,
                'add_delta': add_pa - baseline_pa,
                'sub_delta': sub_pa - baseline_pa,
            }

        # Print best layer
        best_li = max(range(4), key=lambda li: abs(
            surgery_results[concept_name][f'layer_{li}']['add_delta'] -
            surgery_results[concept_name][f'layer_{li}']['sub_delta']))
        r = surgery_results[concept_name][f'layer_{best_li}']
        print(f"  {concept_name} @ layer_{best_li}:")
        print(f"    Baseline: {r['baseline']:.1%}")
        print(f"    +concept: {r['add']:.1%} ({r['add_delta']:+.1%})")
        print(f"    -concept: {r['subtract']:.1%} ({r['sub_delta']:+.1%})")

    elapsed = time.time() - t0

    # Save
    out = {
        'surgery_results': surgery_results,
        'model_params': params,
        'n_test': N - split,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase12_concept_surgery.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, concept_name in enumerate(['symmetry', 'bg_color']):
        layers = [f'layer_{i}' for i in range(4)]
        baselines = [surgery_results[concept_name][l]['baseline'] for l in layers]
        adds = [surgery_results[concept_name][l]['add'] for l in layers]
        subs = [surgery_results[concept_name][l]['subtract'] for l in layers]

        x = np.arange(4)
        w = 0.25
        axes[ax_idx].bar(x - w, baselines, w, label='Baseline', color='#9E9E9E')
        axes[ax_idx].bar(x, adds, w, label='+Concept', color='#4CAF50')
        axes[ax_idx].bar(x + w, subs, w, label='-Concept', color='#F44336')
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([f'Layer {i}' for i in range(4)])
        axes[ax_idx].set_ylabel('Pixel Accuracy')
        axes[ax_idx].set_title(f'Concept Surgery: {concept_name}')
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].set_ylim(0, 1)

    plt.suptitle('Phase 12: Causal Concept Surgery (Activation Steering)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase12_concept_surgery.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
