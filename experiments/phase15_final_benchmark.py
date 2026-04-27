"""
Phase 15: Final Benchmark (Three-Way Comparison)
===================================================
The grand finale: compare all approaches head-to-head.

  Baseline:     Phase 1 Transformer (text DSL, 1.4M params)
  Pointer:      Phase 8 Pointer Network (structured, 40K params)
  GlassBox:     Phase 13 Unified Agent (object graph + router + pointer + concepts)

Same test set, same metrics. Which architecture best solves ARC?
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
MAX_OBJECTS = 20
NODE_FEAT_DIM = 16
N_OPS = 8
N_COLORS = 10
CONCEPT_DIM = 7
N_COLORS_EMB = 11

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.phase1_neural_program_synthesis import (
    DSL_OPS, TOK2ID, ID2TOK, VOCAB_SIZE,
    encode_grid_pair, tokenize_program, detokenize, try_dsl_programs,
    ProgramSynthesizer, load_arc_tasks
)

# ============================================================
# Shared object/concept extraction
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
# Pointer Network (Phase 8 style)
# ============================================================
class PointerNet(nn.Module):
    def __init__(self, node_dim=NODE_FEAT_DIM, hidden=64):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden)
        self.mp1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mp2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.pool = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.op_head = nn.Linear(hidden, N_OPS)
        self.c1_head = nn.Linear(hidden, N_COLORS)
        self.c2_head = nn.Linear(hidden, N_COLORS)
        self.ptr_q = nn.Linear(hidden, hidden)
        self.ptr_k = nn.Linear(hidden, hidden)

    def forward(self, nf, nn_count):
        B = nf.size(0)
        h = self.node_embed(nf)
        mask = torch.arange(MAX_OBJECTS, device=h.device).unsqueeze(0) < nn_count.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.mp1(torch.cat([h, msg.expand_as(h)], -1)); h = self.norm1(h) * mf
        msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
        h = h + self.mp2(torch.cat([h, msg.expand_as(h)], -1)); h = self.norm2(h) * mf
        g = self.pool((h*mf).sum(1) / mf.sum(1).clamp(min=1))
        ptr_l = ((self.ptr_q(g).unsqueeze(1)) * self.ptr_k(h)).sum(-1).masked_fill(~mask, -1e9)
        return self.op_head(g), self.c1_head(g), self.c2_head(g), ptr_l

# ============================================================
# GlassBox Agent (Phase 13 style)
# ============================================================
class GlassBoxAgent(nn.Module):
    def __init__(self, node_dim=NODE_FEAT_DIM, concept_dim=CONCEPT_DIM, hidden=64):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden)
        self.gnn1 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn3 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.concept_proj = nn.Linear(concept_dim, hidden)
        self.rel_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.sym_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.trn_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.router = nn.Sequential(nn.Linear(hidden+concept_dim, 32), nn.ReLU(), nn.Linear(32, 3))
        self.op_head = nn.Linear(hidden, N_OPS)
        self.c1_head = nn.Linear(hidden, N_COLORS)
        self.c2_head = nn.Linear(hidden, N_COLORS)
        self.ptr_q = nn.Linear(hidden, hidden)
        self.ptr_k = nn.Linear(hidden, hidden)

    def forward(self, nf, nn_count, concepts):
        B = nf.size(0)
        mask = torch.arange(MAX_OBJECTS, device=nf.device).unsqueeze(0) < nn_count.unsqueeze(1)
        mf = mask.float().unsqueeze(-1)
        h = self.node_embed(nf)
        for gnn, norm in zip([self.gnn1, self.gnn2, self.gnn3], self.norms):
            msg = (h*mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
            h = h + gnn(torch.cat([h, msg.expand_as(h)], -1)); h = norm(h) * mf
        g = (h*mf).sum(1) / mf.sum(1).clamp(min=1)
        g = g + self.concept_proj(concepts)
        rw = F.softmax(self.router(torch.cat([g, concepts], -1)), -1)
        routed = rw[:,0:1]*self.rel_head(g) + rw[:,1:2]*self.sym_head(g) + rw[:,2:3]*self.trn_head(g)
        ctx = g + routed
        ptr_l = ((self.ptr_q(ctx).unsqueeze(1)) * self.ptr_k(h)).sum(-1).masked_fill(~mask, -1e9)
        return self.op_head(ctx), self.c1_head(ctx), self.c2_head(ctx), ptr_l

# ============================================================
def main():
    print("=" * 60)
    print("Phase 15: Final Benchmark (Three-Way Comparison)")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare shared data
    # For text-based model (Phase 1)
    text_grids, text_progs, raw_pairs = [], [], []
    # For pointer/agent models
    all_nf, all_nn, all_con = [], [], []
    all_ops, all_ptrs, all_c1s, all_c2s = [], [], [], []

    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            prog = try_dsl_programs(inp, out)
            objects, bg = extract_objects(inp)
            h, w = len(inp), len(inp[0])

            if prog is not None:
                text_grids.append(encode_grid_pair(inp, out))
                text_progs.append(tokenize_program(prog))
                raw_pairs.append((inp, out, prog))

            if objects:
                nf = np.zeros((MAX_OBJECTS, NODE_FEAT_DIM), dtype=np.float32)
                n = min(len(objects), MAX_OBJECTS)
                for j in range(n):
                    f = object_to_features(objects[j], h, w)
                    nf[j, :len(f)] = f[:NODE_FEAT_DIM]
                op_id, ptr_idx, c1, c2 = extract_op_label(inp, out)
                all_nf.append(nf)
                all_nn.append(n)
                all_con.append(extract_concepts(inp))
                all_ops.append(op_id)
                all_ptrs.append(min(ptr_idx, max(n-1, 0)))
                all_c1s.append(c1)
                all_c2s.append(c2)

    # Tensors
    tg = torch.tensor(text_grids, dtype=torch.long)
    tp = torch.tensor(text_progs, dtype=torch.long)
    nf_t = torch.tensor(np.array(all_nf), dtype=torch.float32)
    nn_t = torch.tensor(all_nn, dtype=torch.long)
    con_t = torch.tensor(np.array(all_con), dtype=torch.float32)
    op_t = torch.tensor(all_ops, dtype=torch.long)
    ptr_t = torch.tensor(all_ptrs, dtype=torch.long)
    c1_t = torch.tensor(all_c1s, dtype=torch.long)
    c2_t = torch.tensor(all_c2s, dtype=torch.long)

    N_text = len(tg)
    N_struct = len(nf_t)
    split_t = int(N_text * 0.8)
    split_s = int(N_struct * 0.8)
    BATCH = 32
    EPOCHS = 60

    results = {}

    # ===== Model A: Baseline Transformer (Phase 1) =====
    print("\n--- Training Model A: Baseline Transformer ---")
    model_a = ProgramSynthesizer().to(DEVICE)
    params_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Parameters: {params_a:,}")
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model_a.train()
        perm = torch.randperm(split_t)
        el, nb = 0, 0
        for i in range(0, split_t, BATCH):
            idx = perm[i:i+BATCH]
            g, p = tg[idx].to(DEVICE), tp[idx].to(DEVICE)
            mem = model_a.encode(g)
            logits = model_a.decode(mem, p[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), p[:,1:].reshape(-1), ignore_index=TOK2ID['<PAD>'])
            opt_a.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
            opt_a.step(); el += loss.item(); nb += 1
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: loss={el/nb:.4f}")

    model_a.eval()
    em_a = 0
    with torch.no_grad():
        for i in range(split_t, N_text):
            gen = model_a.generate(tg[i:i+1].to(DEVICE))
            pred = detokenize(gen[0].cpu().tolist())
            true = detokenize(tp[i].tolist())
            if pred.strip() == true.strip(): em_a += 1
    n_test_a = N_text - split_t
    em_rate_a = em_a / max(n_test_a, 1)
    results['Baseline Transformer'] = {'em': em_rate_a, 'params': params_a, 'n_test': n_test_a}
    print(f"  EM: {em_rate_a:.1%} ({em_a}/{n_test_a})")

    # ===== Model B: Pointer Network (Phase 8) =====
    print("\n--- Training Model B: Pointer Network ---")
    model_b = PointerNet().to(DEVICE)
    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"  Parameters: {params_b:,}")
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model_b.train()
        perm = torch.randperm(split_s)
        el, nb = 0, 0
        for i in range(0, split_s, BATCH):
            idx = perm[i:i+BATCH]
            ol, cl1, cl2, pl = model_b(nf_t[idx].to(DEVICE), nn_t[idx].to(DEVICE))
            loss = (F.cross_entropy(ol, op_t[idx].to(DEVICE)) + F.cross_entropy(cl1, c1_t[idx].to(DEVICE)) +
                    F.cross_entropy(cl2, c2_t[idx].to(DEVICE)) + F.cross_entropy(pl, ptr_t[idx].to(DEVICE)))
            opt_b.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
            opt_b.step(); el += loss.item(); nb += 1
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: loss={el/nb:.4f}")

    model_b.eval()
    full_b = 0
    with torch.no_grad():
        for i in range(split_s, N_struct):
            ol, cl1, cl2, pl = model_b(nf_t[i:i+1].to(DEVICE), nn_t[i:i+1].to(DEVICE))
            if (ol.argmax(1).item() == op_t[i].item() and cl1.argmax(1).item() == c1_t[i].item() and
                cl2.argmax(1).item() == c2_t[i].item() and pl.argmax(1).item() == ptr_t[i].item()):
                full_b += 1
    n_test_b = N_struct - split_s
    em_rate_b = full_b / max(n_test_b, 1)
    results['Pointer Network'] = {'em': em_rate_b, 'params': params_b, 'n_test': n_test_b}
    print(f"  Full Match: {em_rate_b:.1%} ({full_b}/{n_test_b})")

    # ===== Model C: GlassBox Agent (Phase 13) =====
    print("\n--- Training Model C: GlassBox Agent ---")
    model_c = GlassBoxAgent().to(DEVICE)
    params_c = sum(p.numel() for p in model_c.parameters())
    print(f"  Parameters: {params_c:,}")
    opt_c = torch.optim.Adam(model_c.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model_c.train()
        perm = torch.randperm(split_s)
        el, nb = 0, 0
        for i in range(0, split_s, BATCH):
            idx = perm[i:i+BATCH]
            ol, cl1, cl2, pl = model_c(nf_t[idx].to(DEVICE), nn_t[idx].to(DEVICE), con_t[idx].to(DEVICE))
            loss = (F.cross_entropy(ol, op_t[idx].to(DEVICE)) + F.cross_entropy(cl1, c1_t[idx].to(DEVICE)) +
                    F.cross_entropy(cl2, c2_t[idx].to(DEVICE)) + F.cross_entropy(pl, ptr_t[idx].to(DEVICE)))
            opt_c.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
            opt_c.step(); el += loss.item(); nb += 1
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: loss={el/nb:.4f}")

    model_c.eval()
    full_c = 0
    with torch.no_grad():
        for i in range(split_s, N_struct):
            ol, cl1, cl2, pl = model_c(nf_t[i:i+1].to(DEVICE), nn_t[i:i+1].to(DEVICE), con_t[i:i+1].to(DEVICE))
            if (ol.argmax(1).item() == op_t[i].item() and cl1.argmax(1).item() == c1_t[i].item() and
                cl2.argmax(1).item() == c2_t[i].item() and pl.argmax(1).item() == ptr_t[i].item()):
                full_c += 1
    em_rate_c = full_c / max(n_test_b, 1)
    results['GlassBox Agent'] = {'em': em_rate_c, 'params': params_c, 'n_test': n_test_b}
    print(f"  Full Match: {em_rate_c:.1%} ({full_c}/{n_test_b})")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("FINAL BENCHMARK RESULTS")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name}: EM={r['em']:.1%} | Params={r['params']:,} | Efficiency={r['em']*1000/max(r['params'],1):.4f}")

    out = {'results': results, 'elapsed': elapsed, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(os.path.join(RESULTS_DIR, 'phase15_final_benchmark.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = list(results.keys())
    ems = [results[n]['em'] for n in names]
    params_list = [results[n]['params'] for n in names]
    efficiency = [e * 1000 / max(p, 1) for e, p in zip(ems, params_list)]

    colors = ['#FF9800', '#2196F3', '#4CAF50']
    bars = axes[0].bar(names, ems, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Exact Match / Full Match')
    axes[0].set_title('Accuracy Comparison')
    for b, v in zip(bars, ems):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.1%}', ha='center', fontsize=11)

    axes[1].bar(names, [p/1000 for p in params_list], color=colors)
    axes[1].set_ylabel('Parameters (K)')
    axes[1].set_title('Model Size')
    for i, p in enumerate(params_list):
        axes[1].text(i, p/1000+10, f'{p/1000:.0f}K', ha='center', fontsize=10)

    bars2 = axes[2].bar(names, efficiency, color=colors)
    axes[2].set_ylabel('Accuracy per 1K params')
    axes[2].set_title('Parameter Efficiency')
    for b, v in zip(bars2, efficiency):
        axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{v:.3f}', ha='center', fontsize=10)

    plt.suptitle('Phase 15: Final Three-Way Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase15_final_benchmark.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
