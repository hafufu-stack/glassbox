"""
Phase 2: Test-Time Architecture Assembly
==========================================
Dynamic neural module composition per task.

Given a pool of heterogeneous neural modules:
  - LocalConv: 3x3 convolution (local patterns)
  - GlobalAttn: Self-attention (global relationships)
  - GNNRelation: Graph neural network (object relations)

A Meta-Router analyzes each ARC task's demo pairs and selects
which modules to chain, in what order, to solve it.

The key insight: different ARC tasks need different "brain structures."
Color-swap tasks need LocalConv. Symmetry tasks need GlobalAttn.
Object-movement tasks need GNNRelation.

Metrics:
  - pa_fixed: Pixel accuracy with fixed architecture (baseline)
  - pa_dynamic: Pixel accuracy with dynamic assembly
  - module_usage: Which modules get selected for which tasks
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
N_COLORS = 11  # 0-9 + padding

# ============================================================
# Neural Modules
# ============================================================
class LocalConvModule(nn.Module):
    """Local 3x3 convolution — good for color changes, edge detection."""
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return x + self.net(x)  # Residual

class GlobalAttnModule(nn.Module):
    """Global self-attention — good for symmetry, long-range deps."""
    def __init__(self, channels=64):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(channels, channels*2), nn.ReLU(), nn.Linear(channels*2, channels))
    
    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.norm(tokens))
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)

class GNNRelationModule(nn.Module):
    """Graph-based relation reasoning — good for object interactions."""
    def __init__(self, channels=64):
        super().__init__()
        self.edge_fn = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        self.node_fn = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        nodes = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Simple mean aggregation (approximate message passing)
        global_msg = nodes.mean(dim=1, keepdim=True).expand_as(nodes)
        edge_features = self.edge_fn(torch.cat([nodes, global_msg], dim=-1))
        updated = self.node_fn(torch.cat([nodes, edge_features], dim=-1))
        
        return (nodes + updated).reshape(B, H, W, C).permute(0, 3, 1, 2)

# ============================================================
# Meta-Router: selects modules based on task features
# ============================================================
class MetaRouter(nn.Module):
    """Analyzes task demos and outputs module selection weights."""
    def __init__(self, n_modules=3, feature_dim=32):
        super().__init__()
        # Task feature extractor: simple stats from demo pairs
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Route weights for each of 2 assembly slots
        self.slot1 = nn.Linear(32, n_modules)
        self.slot2 = nn.Linear(32, n_modules)
    
    def forward(self, task_features):
        """Returns (B, 2, n_modules) routing weights."""
        h = self.feature_net(task_features)
        w1 = F.softmax(self.slot1(h), dim=-1)
        w2 = F.softmax(self.slot2(h), dim=-1)
        return torch.stack([w1, w2], dim=1)

# ============================================================
# Assembled Network
# ============================================================
class AssembledNCA(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.embed = nn.Embedding(N_COLORS, channels)
        
        # Module pool
        self.modules_pool = nn.ModuleList([
            LocalConvModule(channels),
            GlobalAttnModule(channels),
            GNNRelationModule(channels),
        ])
        self.module_names = ['LocalConv', 'GlobalAttn', 'GNNRelation']
        
        # Meta-router
        self.router = MetaRouter(n_modules=3)
        
        # Output head
        self.readout = nn.Conv2d(channels, N_COLORS, 1)
    
    def extract_task_features(self, inp_grid, out_grid):
        """Extract simple statistical features from demo pair."""
        features = []
        inp = inp_grid.float()
        out = out_grid.float()
        
        # Size features
        features.append(inp.shape[-2] / MAX_GRID)
        features.append(inp.shape[-1] / MAX_GRID)
        
        # Color count features
        for g in [inp, out]:
            unique = torch.unique(g)
            features.append(len(unique) / N_COLORS)
        
        # Change features
        if inp.shape == out.shape:
            diff = (inp != out).float()
            features.append(diff.mean().item())
            features.append(diff.sum().item() / max(inp.numel(), 1))
        else:
            features.append(1.0)
            features.append(1.0)
        
        # Symmetry features
        if inp.shape[-2] > 1:
            h_sym = (inp == torch.flip(inp, [-2])).float().mean().item()
        else:
            h_sym = 0.0
        features.append(h_sym)
        
        if inp.shape[-1] > 1:
            v_sym = (inp == torch.flip(inp, [-1])).float().mean().item()
        else:
            v_sym = 0.0
        features.append(v_sym)
        
        # Pad to feature_dim
        while len(features) < 32:
            features.append(0.0)
        
        return torch.tensor(features[:32], dtype=torch.float32)
    
    def forward(self, x, task_features, use_router=True):
        """x: (B, H, W) integer grid. task_features: (B, 32)."""
        B, H, W = x.shape
        h = self.embed(x).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        if use_router:
            # Dynamic assembly
            weights = self.router(task_features)  # (B, 2, 3)
            
            for slot in range(2):
                w = weights[:, slot, :]  # (B, 3)
                outputs = [mod(h) for mod in self.modules_pool]
                # Weighted combination
                h_new = torch.zeros_like(h)
                for j, out in enumerate(outputs):
                    h_new = h_new + w[:, j].view(B, 1, 1, 1) * out
                h = h_new
        else:
            # Fixed: just use LocalConv twice (baseline)
            h = self.modules_pool[0](h)
            h = self.modules_pool[0](h)
        
        return self.readout(h)  # (B, N_COLORS, H, W)

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
    """Pad grid to fixed size."""
    h, w = len(grid), len(grid[0])
    padded = np.full((max_h, max_w), pad_val, dtype=np.int64)
    padded[:h, :w] = np.array(grid)
    return padded

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 2: Test-Time Architecture Assembly")
    print("=" * 60)
    t0 = time.time()
    
    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")
    
    # Prepare train/test split
    random.seed(42)
    random.shuffle(tasks)
    n_train = min(int(len(tasks) * 0.8), 320)
    train_tasks = tasks[:n_train]
    test_tasks = tasks[n_train:n_train+50]
    
    # Build model
    model = AssembledNCA(channels=64).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    EPOCHS = 50
    losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_samples = 0
        
        random.shuffle(train_tasks)
        for task in train_tasks[:100]:  # Subsample per epoch
            pairs = task.get('train', [])
            if not pairs:
                continue
            
            for pair in pairs:
                inp = pad_grid(pair['input'])
                out = pad_grid(pair['output'])
                
                inp_t = torch.tensor(inp, dtype=torch.long).unsqueeze(0).to(DEVICE)
                out_t = torch.tensor(out, dtype=torch.long).unsqueeze(0).to(DEVICE)
                
                # Extract task features from this pair
                tf = model.extract_task_features(inp_t[0], out_t[0]).unsqueeze(0).to(DEVICE)
                
                logits = model(inp_t, tf, use_router=True)
                loss = F.cross_entropy(logits, out_t)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_samples += 1
        
        avg_loss = epoch_loss / max(n_samples, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    results_dynamic = {'correct': 0, 'total': 0, 'pa_sum': 0}
    results_fixed = {'correct': 0, 'total': 0, 'pa_sum': 0}
    module_usage = {'LocalConv': 0, 'GlobalAttn': 0, 'GNNRelation': 0}
    
    with torch.no_grad():
        for task in test_tasks:
            for pair in task.get('test', task.get('train', []))[:1]:
                inp = pad_grid(pair['input'])
                out = pad_grid(pair['output'])
                
                inp_t = torch.tensor(inp, dtype=torch.long).unsqueeze(0).to(DEVICE)
                out_t = torch.tensor(out, dtype=torch.long).unsqueeze(0).to(DEVICE)
                tf = model.extract_task_features(inp_t[0], out_t[0]).unsqueeze(0).to(DEVICE)
                
                # Dynamic
                logits_d = model(inp_t, tf, use_router=True)
                pred_d = logits_d.argmax(dim=1)
                pa_d = (pred_d == out_t).float().mean().item()
                results_dynamic['pa_sum'] += pa_d
                results_dynamic['total'] += 1
                if torch.equal(pred_d, out_t):
                    results_dynamic['correct'] += 1
                
                # Track module usage
                weights = model.router(tf)  # (1, 2, 3)
                for slot in range(2):
                    top = weights[0, slot].argmax().item()
                    module_usage[model.module_names[top]] += 1
                
                # Fixed
                logits_f = model(inp_t, tf, use_router=False)
                pred_f = logits_f.argmax(dim=1)
                pa_f = (pred_f == out_t).float().mean().item()
                results_fixed['pa_sum'] += pa_f
                results_fixed['total'] += 1
                if torch.equal(pred_f, out_t):
                    results_fixed['correct'] += 1
    
    pa_dynamic = results_dynamic['pa_sum'] / max(results_dynamic['total'], 1)
    pa_fixed = results_fixed['pa_sum'] / max(results_fixed['total'], 1)
    em_dynamic = results_dynamic['correct'] / max(results_dynamic['total'], 1)
    em_fixed = results_fixed['correct'] / max(results_fixed['total'], 1)
    
    print(f"\n--- Results ---")
    print(f"Dynamic Assembly: PA={pa_dynamic:.1%}, EM={em_dynamic:.1%}")
    print(f"Fixed (LocalConv): PA={pa_fixed:.1%}, EM={em_fixed:.1%}")
    print(f"Module usage: {module_usage}")
    
    elapsed = time.time() - t0
    
    # Save results
    out_results = {
        'dynamic': {'pa': pa_dynamic, 'em': em_dynamic},
        'fixed': {'pa': pa_fixed, 'em': em_fixed},
        'module_usage': module_usage,
        'model_params': params,
        'n_test': results_dynamic['total'],
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase2_architecture_assembly.json'), 'w', encoding='utf-8') as f:
        json.dump(out_results, f, indent=2, ensure_ascii=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    bars = axes[1].bar(['Fixed\n(LocalConv)', 'Dynamic\n(Assembly)'], [pa_fixed, pa_dynamic],
                        color=['#FF9800', '#4CAF50'])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Pixel Accuracy')
    axes[1].set_title('Fixed vs Dynamic Architecture')
    for bar, val in zip(bars, [pa_fixed, pa_dynamic]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.1%}', ha='center', fontsize=12)
    
    names = list(module_usage.keys())
    counts = list(module_usage.values())
    axes[2].bar(names, counts, color=['#2196F3', '#9C27B0', '#F44336'])
    axes[2].set_ylabel('Selection Count')
    axes[2].set_title('Module Usage by Router')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_architecture_assembly.png'), dpi=150)
    plt.close()
    
    print(f"\nElapsed: {elapsed:.1f}s")
    return out_results

if __name__ == '__main__':
    main()
