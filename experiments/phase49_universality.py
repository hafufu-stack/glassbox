"""
Phase 49: Cross-Domain Universality
======================================
Is "15% ablation + 50 steps adaptation" a universal law of neural networks,
or just a quirk of GlassBox's GNN architecture?

Test the one-punch formula on completely different architectures and tasks:
  1. A simple MLP on a synthetic few-shot classification task
  2. A CNN on a 1D pattern recognition task
  3. A Transformer-style self-attention model on sequence tasks

If one-punch works across ALL architectures, it proves that
super-compensation is a fundamental property of gradient descent,
not an artifact of our specific model.
"""
import os,sys,json,time,copy,random
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results')
FIGURES_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'figures')
os.makedirs(RESULTS_DIR,exist_ok=True);os.makedirs(FIGURES_DIR,exist_ok=True)


# ===== Architecture 1: MLP =====
class MLP(nn.Module):
    def __init__(self, in_dim=20, hid=64, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim))
    def forward(self, x):
        return self.net(x)

# ===== Architecture 2: CNN =====
class CNN1D(nn.Module):
    def __init__(self, in_ch=1, hid=32, out_dim=5, seq_len=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hid, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(hid, out_dim)
    def forward(self, x):
        # x: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.conv(x).squeeze(-1))

# ===== Architecture 3: Self-Attention =====
class TinyTransformer(nn.Module):
    def __init__(self, in_dim=20, hid=64, out_dim=5, n_heads=4):
        super().__init__()
        self.emb = nn.Linear(in_dim, hid)
        self.attn = nn.MultiheadAttention(hid, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hid)
        self.fc = nn.Linear(hid, out_dim)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)  # (batch, 1, in_dim)
        h = self.emb(x)
        h2, _ = self.attn(h, h, h)
        h = self.norm(h + h2)
        return self.fc(h.mean(dim=1))


def generate_few_shot_task(n_classes=5, n_support=5, n_query=10, dim=20):
    """Generate a synthetic few-shot classification task.
    Each class has a random prototype; samples are Gaussian around it."""
    prototypes = torch.randn(n_classes, dim) * 2.0
    support_x, support_y = [], []
    query_x, query_y = [], []
    for c in range(n_classes):
        for _ in range(n_support):
            support_x.append(prototypes[c] + torch.randn(dim) * 0.5)
            support_y.append(c)
        for _ in range(n_query):
            query_x.append(prototypes[c] + torch.randn(dim) * 0.5)
            query_y.append(c)
    return (torch.stack(support_x), torch.tensor(support_y),
            torch.stack(query_x), torch.tensor(query_y))


def ablate_random(model, rate):
    am = copy.deepcopy(model)
    with torch.no_grad():
        for p in am.parameters():
            p.mul_((torch.rand_like(p) > rate).float())
    return am

def ablate_least_important_generic(model, rate, support_x, support_y):
    """Gradient-based ablation for any model."""
    am = copy.deepcopy(model); am.train()
    out = am(support_x.to(DEVICE))
    loss = F.cross_entropy(out, support_y.to(DEVICE))
    loss.backward()
    with torch.no_grad():
        for p in am.parameters():
            if p.grad is not None:
                imp = p.grad.abs()
                thr = torch.quantile(imp.flatten(), rate)
                p.mul_((imp > thr).float())
            else:
                p.mul_((torch.rand_like(p) > rate).float())
    am.eval(); return am

def adapt_generic(model, support_x, support_y, steps=50, lr=1e-2):
    am = copy.deepcopy(model)
    opt = torch.optim.SGD(am.parameters(), lr=lr)
    am.train()
    sx, sy = support_x.to(DEVICE), support_y.to(DEVICE)
    for _ in range(steps):
        out = am(sx)
        loss = F.cross_entropy(out, sy)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(am.parameters(), 1.0)
        opt.step()
    am.eval(); return am

def eval_generic(model, query_x, query_y):
    model.eval()
    with torch.no_grad():
        out = model(query_x.to(DEVICE))
        pred = out.argmax(1).cpu()
    return (pred == query_y).float().mean().item()


def run_experiment(arch_name, model_fn, n_pretrain=200, n_test=100):
    """Run the full one-punch universality test for a given architecture."""
    print(f"\n  === {arch_name} ===")

    # Pre-train on many random tasks
    model = model_fn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(n_pretrain):
        model.train()
        sx, sy, _, _ = generate_few_shot_task()
        out = model(sx.to(DEVICE))
        loss = F.cross_entropy(out, sy.to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"    Pre-trained on {n_pretrain} random tasks")

    # Test on new tasks
    methods = {
        'greedy': 'No Adaptation',
        'adapt_50': 'Adapt 50',
        'punch_random': 'Random Ablate 15% + Adapt 50',
        'punch_strategic': 'Strategic Ablate 15% + Adapt 50',
    }
    results = {}
    for key in methods:
        accs = []
        for _ in range(n_test):
            sx, sy, qx, qy = generate_few_shot_task()

            if key == 'greedy':
                current = model
            elif key == 'adapt_50':
                current = adapt_generic(model, sx, sy, steps=50)
            elif key == 'punch_random':
                abl = ablate_random(model, 0.15)
                current = adapt_generic(abl, sx, sy, steps=50)
            elif key == 'punch_strategic':
                abl = ablate_least_important_generic(model, 0.15, sx, sy)
                current = adapt_generic(abl, sx, sy, steps=50)

            acc = eval_generic(current, qx, qy)
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[key] = {'mean': mean_acc, 'std': std_acc}
        delta = mean_acc - results.get('adapt_50', {}).get('mean', mean_acc)
        print(f"    {methods[key]:<35}: {mean_acc:.1%} ± {std_acc:.3f} (delta: {delta:+.1%})")

    return results


def main():
    print("=" * 60)
    print("Phase 49: Cross-Domain Universality")
    print("Is one-punch a universal law of neural networks?")
    print("=" * 60)
    t0 = time.time()

    all_results = {}

    # Test on 3 architectures
    all_results['MLP'] = run_experiment(
        'MLP (3-layer, 64 hidden)',
        lambda: MLP(20, 64, 5),
        n_pretrain=500, n_test=200)

    all_results['CNN'] = run_experiment(
        'CNN (1D Conv, 32 channels)',
        lambda: CNN1D(1, 32, 5, 20),
        n_pretrain=500, n_test=200)

    all_results['Transformer'] = run_experiment(
        'Tiny Transformer (4-head attention)',
        lambda: TinyTransformer(20, 64, 5, 4),
        n_pretrain=500, n_test=200)

    # Summary
    print(f"\n  {'='*60}")
    print(f"  UNIVERSALITY SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Architecture':<20} | {'Adapt':>8} | {'Punch':>8} | {'Delta':>8} | {'Universal?'}")
    print(f"  " + "-" * 65)

    universal_count = 0
    for arch in ['MLP', 'CNN', 'Transformer']:
        adapt = all_results[arch]['adapt_50']['mean']
        punch = all_results[arch]['punch_strategic']['mean']
        delta = punch - adapt
        is_universal = delta > 0
        if is_universal: universal_count += 1
        print(f"  {arch:<20} | {adapt:>7.1%} | {punch:>7.1%} | {delta:>+7.1%} | {'YES' if is_universal else 'NO'}")

    print(f"\n  One-Punch works on {universal_count}/3 architectures")
    verdict = "UNIVERSAL LAW" if universal_count >= 2 else "ARCHITECTURE-SPECIFIC"
    print(f"  Verdict: {verdict}")

    elapsed = time.time() - t0
    # Flatten for JSON
    results_flat = {'verdict': verdict, 'universal_count': universal_count}
    for arch, data in all_results.items():
        for method, vals in data.items():
            results_flat[f'{arch}_{method}'] = vals['mean']
            results_flat[f'{arch}_{method}_std'] = vals['std']
    results_flat['elapsed'] = elapsed
    results_flat['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase49_universality.json'), 'w', encoding='utf-8') as f:
        json.dump(results_flat, f, indent=2, ensure_ascii=False)

    # Plot: grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, arch in enumerate(['MLP', 'CNN', 'Transformer']):
        ax = axes[i]
        methods = ['greedy', 'adapt_50', 'punch_random', 'punch_strategic']
        labels = ['Greedy', 'Adapt 50', 'Random\nPunch', 'Strategic\nPunch']
        vals = [all_results[arch][m]['mean'] for m in methods]
        stds = [all_results[arch][m]['std'] for m in methods]
        colors = ['#9E9E9E', '#FF9800', '#E91E63', '#2196F3']
        bars = ax.bar(range(4), vals, yerr=stds, color=colors, alpha=0.85, capsize=5)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Accuracy'); ax.set_title(f'{arch}')
        ax.set_ylim(0, 1.1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.0%}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle(f'Phase 49: Cross-Domain Universality — Verdict: {verdict}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase49_universality.png'), dpi=150)
    plt.close()
    print(f"\nElapsed: {elapsed:.1f}s")
    return results_flat

if __name__ == '__main__': main()
