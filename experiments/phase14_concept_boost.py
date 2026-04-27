"""
Phase 14: Concept Boost (Steering at Inference Time)
======================================================
Phase 12 proved we can ADD/SUBTRACT concept directions causally.

Now: use this as a PERFORMANCE BOOSTER at test time.
For each test sample:
1. Detect which concepts are present (symmetry? many objects?)
2. Amplify the corresponding concept direction in hidden space
3. Measure if this "concept boost" improves accuracy

This is Representation Engineering as a practical tool, not just analysis.
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

MAX_GRID = 30
N_COLORS = 11

def extract_objects(grid):
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
    return n_obj, bg

def has_symmetry(grid):
    arr = np.array(grid)
    return np.array_equal(arr, np.flipud(arr)) or np.array_equal(arr, np.fliplr(arr))

class SteerableModel(nn.Module):
    def __init__(self, channels=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(N_COLORS, channels)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels), nn.ReLU()))
        self.readout = nn.Conv2d(channels, N_COLORS, 1)
        self.channels = channels

    def forward(self, x, steering=None):
        """steering: list of (layer_idx, direction, alpha) tuples."""
        h = self.embed(x).permute(0, 3, 1, 2)
        for i, layer in enumerate(self.layers):
            h = h + layer(h)
            if steering:
                for li, direction, alpha in steering:
                    if i == li:
                        h = h + alpha * direction.reshape(1, -1, 1, 1)
        return self.readout(h)

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
    print("Phase 14: Concept Boost (Steering at Inference)")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    grids, outputs, sym_flags, obj_counts = [], [], [], []
    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            grids.append(pad_grid(inp))
            outputs.append(pad_grid(out))
            sym_flags.append(1 if has_symmetry(inp) else 0)
            n_obj, _ = extract_objects(inp)
            obj_counts.append(n_obj)

    grids_t = torch.tensor(np.array(grids), dtype=torch.long)
    outputs_t = torch.tensor(np.array(outputs), dtype=torch.long)
    sym_t = torch.tensor(sym_flags, dtype=torch.long)

    N = len(grids_t)
    split = int(N * 0.8)
    BATCH = 32

    # Train model
    print("\n--- Training Model ---")
    model = SteerableModel().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        model.train()
        perm = torch.randperm(split)
        epoch_loss, n_b = 0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            logits = model(grids_t[idx].to(DEVICE))
            loss = F.cross_entropy(logits, outputs_t[idx].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30: loss={epoch_loss/n_b:.4f}")

    # Find concept directions via probes
    print("\n--- Finding Concept Directions ---")
    model.eval()
    layer_hiddens = {i: [] for i in range(4)}

    with torch.no_grad():
        for i in range(0, N, BATCH):
            x = grids_t[i:i+BATCH].to(DEVICE)
            h = model.embed(x).permute(0, 3, 1, 2)
            for li, layer in enumerate(model.layers):
                h = h + layer(h)
                layer_hiddens[li].append(h.mean(dim=[2,3]).cpu())

    for li in layer_hiddens:
        layer_hiddens[li] = torch.cat(layer_hiddens[li], 0)

    # Find symmetry direction at best layer
    best_li, best_acc = 0, 0
    best_direction = None
    for li in range(4):
        hidden = layer_hiddens[li]
        probe = nn.Linear(hidden.size(1), 2).to(DEVICE)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        for _ in range(50):
            logits = probe(hidden[:split].to(DEVICE))
            loss = F.cross_entropy(logits, sym_t[:split].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (probe(hidden[split:].to(DEVICE)).argmax(1) == sym_t[split:].to(DEVICE)).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_li = li
            W = probe.weight.data.cpu()
            best_direction = (W[1] - W[0])
            best_direction = best_direction / best_direction.norm()
        print(f"  Symmetry probe @ layer_{li}: {acc:.1%}")

    print(f"  Best: layer_{best_li} ({best_acc:.1%})")
    direction = best_direction.to(DEVICE)

    # Test: baseline vs concept-boosted
    print("\n--- Concept Boost Experiment ---")
    alphas = [0.0, 1.0, 2.0, 3.0, 5.0, -3.0]
    results_by_alpha = {}

    for alpha in alphas:
        correct, total_px = 0, 0
        sym_correct, sym_total = 0, 0
        nosym_correct, nosym_total = 0, 0

        with torch.no_grad():
            for i in range(split, N):
                x = grids_t[i:i+1].to(DEVICE)
                y = outputs_t[i:i+1].to(DEVICE)

                if alpha == 0.0:
                    logits = model(x)
                else:
                    # Only boost samples that HAVE the concept
                    is_sym = sym_flags[i]
                    actual_alpha = alpha if is_sym else alpha * 0.1  # Gentle for non-sym
                    steering = [(best_li, direction, actual_alpha)]
                    logits = model(x, steering=steering)

                c = (logits.argmax(1) == y).sum().item()
                correct += c
                total_px += y.numel()

                if sym_flags[i]:
                    sym_correct += c
                    sym_total += y.numel()
                else:
                    nosym_correct += c
                    nosym_total += y.numel()

        pa = correct / max(total_px, 1)
        sym_pa = sym_correct / max(sym_total, 1) if sym_total > 0 else 0
        nosym_pa = nosym_correct / max(nosym_total, 1) if nosym_total > 0 else 0

        results_by_alpha[alpha] = {'pa': pa, 'sym_pa': sym_pa, 'nosym_pa': nosym_pa}
        label = "Baseline" if alpha == 0 else f"alpha={alpha}"
        delta = pa - results_by_alpha[0.0]['pa'] if alpha != 0 else 0
        print(f"  {label}: PA={pa:.1%} (sym={sym_pa:.1%}, nosym={nosym_pa:.1%}) delta={delta:+.2%}")

    elapsed = time.time() - t0

    out = {
        'results_by_alpha': {str(k): v for k, v in results_by_alpha.items()},
        'best_probe_layer': best_li,
        'best_probe_accuracy': best_acc,
        'model_params': params,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase14_concept_boost.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    alpha_vals = sorted(results_by_alpha.keys())
    pas = [results_by_alpha[a]['pa'] for a in alpha_vals]
    sym_pas = [results_by_alpha[a]['sym_pa'] for a in alpha_vals]

    axes[0].plot(alpha_vals, pas, 'o-', color='#4CAF50', label='Overall PA', linewidth=2)
    axes[0].plot(alpha_vals, sym_pas, 's--', color='#2196F3', label='Symmetric subset PA')
    axes[0].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Steering Alpha')
    axes[0].set_ylabel('Pixel Accuracy')
    axes[0].set_title('Concept Boost: PA vs Steering Strength')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    baseline = results_by_alpha[0.0]['pa']
    deltas = [results_by_alpha[a]['pa'] - baseline for a in alpha_vals if a != 0]
    alpha_no0 = [a for a in alpha_vals if a != 0]
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
    axes[1].bar([f'a={a}' for a in alpha_no0], deltas, color=colors)
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_ylabel('Delta PA from Baseline')
    axes[1].set_title('Improvement by Steering Strength')
    for i, d in enumerate(deltas):
        axes[1].text(i, d + 0.001 * np.sign(d), f'{d:+.2%}', ha='center', fontsize=9)

    plt.suptitle('Phase 14: Concept Boost (Activation Steering at Inference)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase14_concept_boost.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
