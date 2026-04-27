"""
Phase 11: Concept-Guided Routing
==================================
Cross-breed Phase 3 (Concept Probing) x Phase 2 (Meta-Router).

Phase 3 showed we CAN decode concepts: symmetry(92%), bg_color(82%).
Phase 2 showed dynamic routing beats static (94% vs 76% PA).

This phase: explicitly inject decoded concept vectors INTO the router.
Instead of routing blindly on raw features, the router KNOWS:
  "This task has symmetry -> prioritize Mirror module"
  "This task has 5 objects -> prioritize GNN module"

Compare: blind router vs concept-guided router.
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
from collections import deque

# ============================================================
# Concept extraction (from Phase 3)
# ============================================================
def extract_concepts(grid):
    """Extract human-interpretable concept features."""
    arr = np.array(grid)
    h, w = arr.shape

    # n_objects
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

    # Symmetry
    h_sym = float(np.array_equal(arr, np.flipud(arr)))
    v_sym = float(np.array_equal(arr, np.fliplr(arr)))

    # Colors
    n_colors = len(set(arr.flatten().tolist()))

    # Grid size
    size_ratio = max(h, w) / 30.0

    # Background ratio
    bg_ratio = float(np.sum(arr == bg)) / max(arr.size, 1)

    return np.array([
        n_obj / 10.0,    # normalized object count
        h_sym,            # horizontal symmetry
        v_sym,            # vertical symmetry
        n_colors / 10.0,  # normalized color count
        size_ratio,       # grid size
        bg_ratio,         # background ratio
        float(bg) / 10.0, # background color
    ], dtype=np.float32)

CONCEPT_DIM = 7

# ============================================================
# Neural modules (from Phase 2)
# ============================================================
class LocalConvModule(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())
    def forward(self, x): return self.net(x)

class GlobalAttnModule(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(ch, 4, batch_first=True)
        self.norm = nn.LayerNorm(ch)
    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.reshape(B, C, -1).permute(0, 2, 1)
        out, _ = self.attn(flat, flat, flat)
        out = self.norm(out)
        return out.permute(0, 2, 1).reshape(B, C, H, W)

class MirrorModule(nn.Module):
    """Specialized module for symmetry-related transforms."""
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())
    def forward(self, x):
        flipped = torch.flip(x, [-1])  # Horizontal flip
        combined = torch.cat([x, flipped], dim=1)
        return self.net(combined)

# ============================================================
# Routers
# ============================================================
class BlindRouter(nn.Module):
    """Phase 2 style: routes based on raw grid statistics."""
    def __init__(self, ch=32, n_modules=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Linear(ch, 32), nn.ReLU(), nn.Linear(32, n_modules))
    def forward(self, x):
        pooled = self.pool(x).squeeze(-1).squeeze(-1)
        return F.softmax(self.router(pooled), dim=-1)

class ConceptGuidedRouter(nn.Module):
    """New: routes based on raw grid + explicit concept features."""
    def __init__(self, ch=32, concept_dim=CONCEPT_DIM, n_modules=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Linear(ch + concept_dim, 32), nn.ReLU(), nn.Linear(32, n_modules))
    def forward(self, x, concepts):
        pooled = self.pool(x).squeeze(-1).squeeze(-1)
        combined = torch.cat([pooled, concepts], dim=-1)
        return F.softmax(self.router(combined), dim=-1)

# ============================================================
# Full model
# ============================================================
class ConceptRouterModel(nn.Module):
    def __init__(self, ch=32, use_concepts=False):
        super().__init__()
        self.embed = nn.Embedding(N_COLORS, ch)
        self.modules_list = nn.ModuleList([
            LocalConvModule(ch), GlobalAttnModule(ch), MirrorModule(ch)])
        if use_concepts:
            self.router = ConceptGuidedRouter(ch, CONCEPT_DIM, 3)
        else:
            self.router = BlindRouter(ch, 3)
        self.readout = nn.Conv2d(ch, N_COLORS, 1)
        self.use_concepts = use_concepts

    def forward(self, x, concepts=None):
        h = self.embed(x).permute(0, 3, 1, 2)
        if self.use_concepts:
            weights = self.router(h, concepts)
        else:
            weights = self.router(h)

        out = torch.zeros_like(h)
        for i, mod in enumerate(self.modules_list):
            out = out + weights[:, i:i+1, None, None] * mod(h)
        return self.readout(out), weights

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
    print("Phase 11: Concept-Guided Routing")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare data
    grids, outputs, concepts = [], [], []
    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            grids.append(pad_grid(inp))
            outputs.append(pad_grid(out))
            concepts.append(extract_concepts(inp))

    grids_t = torch.tensor(np.array(grids), dtype=torch.long)
    outputs_t = torch.tensor(np.array(outputs), dtype=torch.long)
    concepts_t = torch.tensor(np.array(concepts), dtype=torch.float32)

    N = len(grids_t)
    split = int(N * 0.8)
    print(f"Total: {N}, Train: {split}, Test: {N-split}")

    BATCH = 32
    EPOCHS = 40
    results_all = {}

    for mode_name, use_concepts in [('blind', False), ('concept_guided', True)]:
        print(f"\n--- Training: {mode_name} router ---")
        model = ConceptRouterModel(ch=32, use_concepts=use_concepts).to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(split)
            epoch_loss, n_b = 0, 0
            for i in range(0, split, BATCH):
                idx = perm[i:i+BATCH]
                x = grids_t[idx].to(DEVICE)
                y = outputs_t[idx].to(DEVICE)
                c = concepts_t[idx].to(DEVICE) if use_concepts else None
                logits, _ = model(x, c)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_b += 1
            avg_loss = epoch_loss / max(n_b, 1)
            losses.append(avg_loss)
            if (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}")

        # Evaluate
        model.eval()
        correct_pixels, total_pixels = 0, 0
        module_usage = np.zeros(3)

        with torch.no_grad():
            for i in range(split, N):
                x = grids_t[i:i+1].to(DEVICE)
                y = outputs_t[i:i+1].to(DEVICE)
                c = concepts_t[i:i+1].to(DEVICE) if use_concepts else None
                logits, weights = model(x, c)
                preds = logits.argmax(1)
                correct_pixels += (preds == y).sum().item()
                total_pixels += y.numel()
                module_usage += weights[0].cpu().numpy()

        pa = correct_pixels / max(total_pixels, 1)
        n_test = N - split
        module_usage /= max(n_test, 1)
        module_names = ['LocalConv', 'GlobalAttn', 'Mirror']

        print(f"  PA: {pa:.1%}")
        print(f"  Module weights: {dict(zip(module_names, [f'{w:.2f}' for w in module_usage]))}")

        results_all[mode_name] = {
            'pa': pa,
            'module_usage': dict(zip(module_names, module_usage.tolist())),
            'params': params,
            'losses': losses,
        }

    elapsed = time.time() - t0

    blind_pa = results_all['blind']['pa']
    guided_pa = results_all['concept_guided']['pa']
    improvement = guided_pa - blind_pa

    print(f"\n--- Comparison ---")
    print(f"Blind Router PA:           {blind_pa:.1%}")
    print(f"Concept-Guided Router PA:  {guided_pa:.1%}")
    print(f"Improvement:               {improvement:+.1%}")

    out = {
        'blind_pa': blind_pa,
        'concept_guided_pa': guided_pa,
        'improvement': improvement,
        'blind_usage': results_all['blind']['module_usage'],
        'guided_usage': results_all['concept_guided']['module_usage'],
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase11_concept_routing.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bars = axes[0].bar(['Blind\nRouter', 'Concept-Guided\nRouter'], [blind_pa, guided_pa],
                        color=['#FF9800', '#4CAF50'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Pixel Accuracy')
    axes[0].set_title('Phase 11: Blind vs Concept-Guided Routing')
    for bar, val in zip(bars, [blind_pa, guided_pa]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f'{val:.1%}', ha='center', fontsize=12)

    # Module usage comparison
    module_names = ['LocalConv', 'GlobalAttn', 'Mirror']
    x_pos = np.arange(3)
    w = 0.35
    blind_u = [results_all['blind']['module_usage'][m] for m in module_names]
    guided_u = [results_all['concept_guided']['module_usage'][m] for m in module_names]
    axes[1].bar(x_pos - w/2, blind_u, w, label='Blind', color='#FF9800', alpha=0.8)
    axes[1].bar(x_pos + w/2, guided_u, w, label='Concept-Guided', color='#4CAF50', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(module_names)
    axes[1].set_ylabel('Average Weight')
    axes[1].set_title('Module Usage Distribution')
    axes[1].legend()

    # Training losses
    axes[2].plot(results_all['blind']['losses'], label='Blind', color='#FF9800')
    axes[2].plot(results_all['concept_guided']['losses'], label='Concept-Guided', color='#4CAF50')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase11_concept_routing.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
