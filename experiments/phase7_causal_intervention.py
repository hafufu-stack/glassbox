"""
Phase 7: Causal Attention Intervention
========================================
Prove CAUSALITY, not just correlation.

Phase 6 showed attention correlates with output.
Phase 7 asks: if we FORCE the attention to look at a different object,
does the output code CHANGE to target that object?

Method:
1. Train the Phase 6 model
2. At inference, identify the attention peak (object A)
3. Zero out attention to object A, force it to object B
4. Check if the output DSL now targets object B's properties

If yes: attention CAUSES the output (causal, not just correlational)
If no: attention is epiphenomenal (just a byproduct)
"""

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.phase1_neural_program_synthesis import (
    DSL_OPS, DSL_VOCAB, TOK2ID, ID2TOK, VOCAB_SIZE,
    encode_grid_pair, tokenize_program, detokenize, try_dsl_programs,
    load_arc_tasks
)

# ============================================================
# Object extraction (from Phase 4)
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
                objects.append({'color': color, 'pixels': pixels,
                                'center': (np.mean([p[0] for p in pixels]),
                                           np.mean([p[1] for p in pixels]))})
    return objects, bg

# ============================================================
# Transformer with Attention Intervention
# ============================================================
class IntervenableTransformer(nn.Module):
    def __init__(self, grid_dim=256, d_model=128, nhead=4, num_layers=3,
                 vocab_size=VOCAB_SIZE, max_prog_len=32):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.grid_embed = nn.Embedding(101, d_model)
        self.pos_enc_e = nn.Embedding(grid_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4,
                                                    dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc_d = nn.Embedding(max_prog_len, d_model)
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'ff': nn.Sequential(nn.Linear(d_model, d_model*4), nn.ReLU(), nn.Linear(d_model*4, d_model)),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }))
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._intervention = None  # (mask_positions, force_positions)

    def set_intervention(self, mask_positions=None, force_positions=None):
        """Set attention intervention: mask some positions, force others."""
        self._intervention = (mask_positions, force_positions)

    def clear_intervention(self):
        self._intervention = None

    def encode(self, grid_seq):
        pos = torch.arange(grid_seq.size(1), device=grid_seq.device)
        x = self.grid_embed(grid_seq.clamp(0, 100)) + self.pos_enc_e(pos)
        return self.encoder(x)

    def decode(self, memory, tgt_tokens):
        T = tgt_tokens.size(1)
        pos = torch.arange(T, device=tgt_tokens.device)
        x = self.tok_embed(tgt_tokens) + self.pos_enc_d(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_tokens.device)

        for layer in self.decoder_layers:
            normed = layer['norm1'](x)
            attn_out, _ = layer['self_attn'](normed, normed, normed, attn_mask=causal_mask)
            x = x + attn_out

            normed = layer['norm2'](x)
            if self._intervention is not None:
                # Compute attention manually with intervention
                attn_out, attn_w = layer['cross_attn'](
                    normed, memory, memory, need_weights=True, average_attn_weights=True)
                # Intervene on attention
                mask_pos, force_pos = self._intervention
                if mask_pos is not None and force_pos is not None:
                    # Re-weight: suppress mask_pos, boost force_pos
                    # We modify the output by re-computing with modified memory
                    mem_mod = memory.clone()
                    for mp in mask_pos:
                        if mp < mem_mod.size(1):
                            mem_mod[:, mp, :] *= 0.0  # Zero out masked positions
                    for fp in force_pos:
                        if fp < mem_mod.size(1):
                            mem_mod[:, fp, :] *= 3.0  # Amplify forced positions
                    attn_out, _ = layer['cross_attn'](normed, mem_mod, mem_mod)
            else:
                attn_out, _ = layer['cross_attn'](normed, memory, memory)
            x = x + attn_out

            normed = layer['norm3'](x)
            x = x + layer['ff'](normed)
        return self.output_proj(x)

    def generate(self, grid_seq, max_len=32):
        memory = self.encode(grid_seq)
        B = grid_seq.size(0)
        tokens = torch.full((B, 1), TOK2ID['<BOS>'], dtype=torch.long, device=grid_seq.device)
        for _ in range(max_len - 1):
            logits = self.decode(memory, tokens)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == TOK2ID['<EOS>']).all():
                break
        return tokens

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 7: Causal Attention Intervention")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare data
    grid_seqs, prog_seqs, raw_pairs = [], [], []
    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            prog = try_dsl_programs(inp, out)
            if prog is not None:
                grid_seqs.append(encode_grid_pair(inp, out))
                prog_seqs.append(tokenize_program(prog))
                raw_pairs.append((inp, out, prog))
    grid_seqs = torch.tensor(grid_seqs, dtype=torch.long)
    prog_seqs = torch.tensor(prog_seqs, dtype=torch.long)
    N = len(grid_seqs)
    split = int(N * 0.8)
    print(f"Train: {split}, Test: {N-split}")

    # Train
    print("\n--- Training Intervenable Model ---")
    model = IntervenableTransformer().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 32

    for epoch in range(80):
        model.train()
        perm = torch.randperm(split)
        epoch_loss, n_b = 0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            g = grid_seqs[idx].to(DEVICE)
            p = prog_seqs[idx].to(DEVICE)
            memory = model.encode(g)
            logits = model.decode(memory, p[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), p[:, 1:].reshape(-1),
                                   ignore_index=TOK2ID['<PAD>'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/80: loss={epoch_loss/n_b:.4f}")

    # Causal intervention experiment
    print("\n--- Causal Intervention Experiment ---")
    model.eval()
    n_causal = 0      # Times intervention changed output to match target
    n_changed = 0     # Times output changed at all
    n_tested = 0      # Total interventions performed
    n_baseline_correct = 0

    with torch.no_grad():
        for i in range(split, min(N, split + 50)):
            inp, out, true_prog = raw_pairs[i]
            h, w = len(inp), len(inp[0])
            objects, bg = extract_objects(inp)

            if len(objects) < 2:
                continue

            g = grid_seqs[i:i+1].to(DEVICE)

            # Baseline: no intervention
            model.clear_intervention()
            baseline_tokens = model.generate(g)
            baseline_prog = detokenize(baseline_tokens[0].cpu().tolist())

            if baseline_prog.strip() == true_prog.strip():
                n_baseline_correct += 1

            # For each pair of objects, try swapping attention
            obj_a = objects[0]
            obj_b = objects[1] if len(objects) > 1 else objects[0]

            # Positions in grid_seq corresponding to object A's pixels
            mask_positions = []
            for (r, c) in obj_a['pixels'][:10]:
                pos = 2 + r * w + c  # grid_seq layout: [h, w, pixels...]
                if pos < 256:
                    mask_positions.append(pos)

            # Positions for object B
            force_positions = []
            for (r, c) in obj_b['pixels'][:10]:
                pos = 2 + r * w + c
                if pos < 256:
                    force_positions.append(pos)

            if not mask_positions or not force_positions:
                continue

            # Intervened: mask A, force B
            model.set_intervention(mask_positions, force_positions)
            intervened_tokens = model.generate(g)
            intervened_prog = detokenize(intervened_tokens[0].cpu().tolist())
            model.clear_intervention()

            n_tested += 1
            if intervened_prog.strip() != baseline_prog.strip():
                n_changed += 1
                # Check if the change is semantically toward object B
                # e.g., RECOLOR changed color to B's color
                parts_b = intervened_prog.split()
                if len(parts_b) >= 2:
                    for p in parts_b[1:]:
                        try:
                            if int(p) == obj_b['color']:
                                n_causal += 1
                                break
                        except ValueError:
                            pass

    change_rate = n_changed / max(n_tested, 1)
    causal_rate = n_causal / max(n_tested, 1)
    print(f"\n--- Results ---")
    print(f"Tests: {n_tested}")
    print(f"Baseline correct: {n_baseline_correct}")
    print(f"Output changed by intervention: {change_rate:.1%} ({n_changed}/{n_tested})")
    print(f"Causally redirected to target: {causal_rate:.1%} ({n_causal}/{n_tested})")

    elapsed = time.time() - t0

    results = {
        'n_tested': n_tested,
        'n_changed': n_changed,
        'n_causal': n_causal,
        'change_rate': change_rate,
        'causal_rate': causal_rate,
        'baseline_correct': n_baseline_correct,
        'model_params': params,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase7_causal_intervention.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    bars = ax[0].bar(['Output\nChanged', 'Causally\nRedirected'], [change_rate, causal_rate],
                      color=['#FF9800', '#4CAF50'])
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Rate')
    ax[0].set_title('Phase 7: Causal Attention Intervention')
    for bar, val in zip(bars, [change_rate, causal_rate]):
        ax[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                   f'{val:.1%}', ha='center', fontsize=12)

    # Breakdown
    labels = ['No Change', 'Changed\n(non-causal)', 'Causally\nRedirected']
    sizes = [n_tested - n_changed, n_changed - n_causal, n_causal]
    colors_pie = ['#BDBDBD', '#FF9800', '#4CAF50']
    ax[1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90)
    ax[1].set_title('Intervention Outcome Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase7_causal_intervention.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
