"""
Phase 17: Sparse Concept Discovery via SAE
=============================================
Phase 12 used HUMAN-DEFINED concepts (symmetry, bg_color).
But what concepts has the model learned ON ITS OWN?

Method: Sparse Autoencoder (SAE) decomposes dense hidden vectors
into sparse, interpretable features without any labels.

If feature #42 only activates on grids with "enclosed rectangles"
and feature #107 only activates on "diagonal patterns",
then the model has autonomously discovered geometric concepts.

This is Anthropic's approach to mechanistic interpretability,
applied to visual reasoning for the first time.
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

# ============================================================
# Target model to analyze
# ============================================================
class TargetModel(nn.Module):
    def __init__(self, channels=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(N_COLORS, channels)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels), nn.ReLU()))
        self.readout = nn.Conv2d(channels, N_COLORS, 1)
    def forward(self, x):
        h = self.embed(x).permute(0, 3, 1, 2)
        for layer in self.layers:
            h = h + layer(h)
        return self.readout(h)

# ============================================================
# Sparse Autoencoder
# ============================================================
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, n_features=256, sparsity_coeff=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim)
        self.sparsity_coeff = sparsity_coeff
        self.n_features = n_features

    def forward(self, x):
        # Encode to sparse features
        z = F.relu(self.encoder(x))  # (B, n_features) - sparse activations
        # Decode back
        x_hat = self.decoder(z)
        # Losses
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = z.abs().mean()  # L1 penalty for sparsity
        total_loss = recon_loss + self.sparsity_coeff * sparsity_loss
        return z, x_hat, total_loss, recon_loss.item(), sparsity_loss.item()

# ============================================================
# Concept labeling functions (for validation only)
# ============================================================
def has_symmetry(grid):
    arr = np.array(grid)
    return np.array_equal(arr, np.flipud(arr)) or np.array_equal(arr, np.fliplr(arr))

def count_objects(grid):
    arr = np.array(grid)
    h, w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    bg = int(np.bincount(arr.flatten()).argmax())
    n = 0
    for r in range(h):
        for c in range(w):
            if not visited[r, c] and arr[r, c] != bg:
                n += 1
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == arr[r, c]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
    return n

def has_diagonal(grid):
    arr = np.array(grid)
    h, w = arr.shape
    bg = int(np.bincount(arr.flatten()).argmax())
    for r in range(h-1):
        for c in range(w-1):
            if arr[r,c] != bg and arr[r+1,c+1] == arr[r,c] and arr[r,c+1] == bg:
                return True
    return False

def has_border(grid):
    arr = np.array(grid)
    h, w = arr.shape
    if h < 3 or w < 3: return False
    border = np.concatenate([arr[0,:], arr[-1,:], arr[1:-1,0], arr[1:-1,-1]])
    bg = int(np.bincount(arr.flatten()).argmax())
    border_color = set(border.tolist()) - {bg}
    return len(border_color) == 1 and len(border_color) > 0

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
    print("Phase 17: Sparse Concept Discovery via SAE")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    grids, outputs = [], []
    concept_labels = []  # For validation
    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            grids.append(pad_grid(inp))
            outputs.append(pad_grid(out))
            concept_labels.append({
                'symmetry': has_symmetry(inp),
                'n_objects': count_objects(inp),
                'diagonal': has_diagonal(inp),
                'border': has_border(inp),
                'n_colors': len(set(np.array(inp).flatten().tolist())),
            })

    grids_t = torch.tensor(np.array(grids), dtype=torch.long)
    outputs_t = torch.tensor(np.array(outputs), dtype=torch.long)
    N = len(grids_t)
    BATCH = 32

    # Train target model
    print("\n--- Training Target Model ---")
    model = TargetModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            logits = model(grids_t[idx].to(DEVICE))
            loss = F.cross_entropy(logits, outputs_t[idx].to(DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30")

    # Extract hidden states (layer 2 - middle of network)
    print("\n--- Extracting Hidden States ---")
    model.eval()
    all_hiddens = []
    with torch.no_grad():
        for i in range(0, N, BATCH):
            x = grids_t[i:i+BATCH].to(DEVICE)
            h = model.embed(x).permute(0, 3, 1, 2)
            for li, layer in enumerate(model.layers):
                h = h + layer(h)
                if li == 2:  # Extract from layer 2
                    pooled = h.mean(dim=[2, 3])
                    all_hiddens.append(pooled.cpu())
    hiddens = torch.cat(all_hiddens, 0)  # (N, 64)
    print(f"  Hidden shape: {hiddens.shape}")

    # Train SAE
    print("\n--- Training Sparse Autoencoder ---")
    N_FEATURES = 128
    sae = SparseAutoencoder(hiddens.size(1), N_FEATURES, sparsity_coeff=5e-3).to(DEVICE)
    sae_opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

    sae_losses = []
    for epoch in range(200):
        perm = torch.randperm(N)
        epoch_loss = 0
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            z, x_hat, loss, rl, sl = sae(hiddens[idx].to(DEVICE))
            sae_opt.zero_grad(); loss.backward(); sae_opt.step()
            epoch_loss += loss.item()
        sae_losses.append(epoch_loss)
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/200: loss={epoch_loss:.4f}")

    # Analyze discovered features
    print("\n--- Analyzing Discovered Features ---")
    sae.eval()
    with torch.no_grad():
        all_z, _, _, _, _ = sae(hiddens.to(DEVICE))
        all_z = all_z.cpu().numpy()  # (N, N_FEATURES)

    # Sparsity statistics
    active_per_sample = (all_z > 0.01).sum(axis=1)
    avg_active = active_per_sample.mean()
    total_ever_active = (all_z.max(axis=0) > 0.01).sum()
    print(f"  Average active features per sample: {avg_active:.1f} / {N_FEATURES}")
    print(f"  Total features ever activated: {total_ever_active} / {N_FEATURES}")

    # Correlate features with known concepts
    print("\n--- Feature-Concept Correlations ---")
    sym_labels = np.array([c['symmetry'] for c in concept_labels], dtype=float)
    nobj_labels = np.array([c['n_objects'] for c in concept_labels], dtype=float)
    diag_labels = np.array([c['diagonal'] for c in concept_labels], dtype=float)
    border_labels = np.array([c['border'] for c in concept_labels], dtype=float)

    best_correlations = {}
    for concept_name, labels in [('symmetry', sym_labels), ('n_objects', nobj_labels),
                                   ('diagonal', diag_labels), ('border', border_labels)]:
        best_feat, best_corr = -1, 0
        for fi in range(N_FEATURES):
            if all_z[:, fi].std() < 1e-6: continue
            corr = np.corrcoef(all_z[:, fi], labels)[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_feat = fi
        best_correlations[concept_name] = {'feature': best_feat, 'correlation': best_corr}
        print(f"  {concept_name}: best feature #{best_feat}, r={best_corr:.3f}")

    # Find "novel" features (high variance, low correlation with known concepts)
    feature_variances = all_z.var(axis=0)
    known_features = set(v['feature'] for v in best_correlations.values())
    novel_features = []
    for fi in np.argsort(-feature_variances):
        if fi not in known_features and feature_variances[fi] > 0.01:
            novel_features.append(int(fi))
            if len(novel_features) >= 5:
                break

    print(f"\n  Top novel features (unknown concepts): {novel_features}")
    for fi in novel_features[:3]:
        top_samples = np.argsort(-all_z[:, fi])[:5]
        print(f"    Feature #{fi}: top activating samples = {top_samples.tolist()}")

    elapsed = time.time() - t0

    out = {
        'n_features': N_FEATURES,
        'avg_active_per_sample': float(avg_active),
        'total_ever_active': int(total_ever_active),
        'best_correlations': {k: {'feature': v['feature'], 'r': round(v['correlation'], 4)}
                              for k, v in best_correlations.items()},
        'novel_features': novel_features,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase17_sae_discovery.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Feature activation heatmap (top 30 features for first 50 samples)
    top_feats = np.argsort(-feature_variances)[:30]
    axes[0].imshow(all_z[:50, top_feats].T, aspect='auto', cmap='hot')
    axes[0].set_xlabel('Sample'); axes[0].set_ylabel('Feature')
    axes[0].set_title('SAE Feature Activations (top 30)')

    # Sparsity histogram
    axes[1].hist(active_per_sample, bins=20, color='#2196F3', edgecolor='white')
    axes[1].axvline(avg_active, color='red', linestyle='--', label=f'Mean={avg_active:.1f}')
    axes[1].set_xlabel('Active Features per Sample')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Sparsity Distribution')
    axes[1].legend()

    # Concept correlations
    concepts = list(best_correlations.keys())
    corrs = [abs(best_correlations[c]['correlation']) for c in concepts]
    bars = axes[2].bar(concepts, corrs, color=['#4CAF50' if c > 0.3 else '#FF9800' for c in corrs])
    axes[2].set_ylabel('|Correlation|')
    axes[2].set_title('Feature-Concept Alignment')
    axes[2].set_ylim(0, 1)
    for b, v in zip(bars, corrs):
        axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.2f}', ha='center')

    plt.suptitle('Phase 17: Sparse Concept Discovery (SAE)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase17_sae_discovery.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
