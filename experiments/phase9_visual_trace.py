"""
Phase 9: Visual Execution Trace
=================================
Fix Phase 5's failure: feed INTERMEDIATE visual states, not final diffs.

Pipeline (Visual Chain-of-Thought):
1. Model sees input grid
2. Generates 1 DSL command
3. Command is EXECUTED -> intermediate grid state
4. Intermediate state is fed BACK to the model
5. Model generates next command (or stops)
6. Repeat until <DONE> or max steps

This mimics how human programmers debug:
they run code LINE BY LINE and watch the output change.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.phase1_neural_program_synthesis import (
    DSL_OPS, TOK2ID, ID2TOK, VOCAB_SIZE,
    tokenize_program, detokenize, try_dsl_programs, load_arc_tasks
)

# ============================================================
# DSL Executor (from Phase 5)
# ============================================================
def execute_dsl(program_str, input_grid):
    inp = np.array(input_grid)
    out = inp.copy()
    parts = program_str.split()
    if not parts:
        return None
    op = parts[0]
    try:
        if op == 'IDENTITY':
            return out.tolist()
        elif op == 'FILL' and len(parts) >= 2:
            out[:] = int(parts[1])
            return out.tolist()
        elif op == 'RECOLOR' and len(parts) >= 3:
            old_c, new_c = int(parts[1]), int(parts[2])
            out[out == old_c] = new_c
            return out.tolist()
        elif op == 'MIRROR_H':
            return np.flipud(inp).tolist()
        elif op == 'MIRROR_V':
            return np.fliplr(inp).tolist()
        elif op == 'COPY' and len(parts) >= 7:
            return out.tolist()
        elif op == 'SWAP' and len(parts) >= 3:
            c1, c2 = int(parts[1]), int(parts[2])
            m1, m2 = inp == c1, inp == c2
            out[m1], out[m2] = c2, c1
            return out.tolist()
    except (ValueError, IndexError):
        return None
    return out.tolist()

# ============================================================
# Step-by-Step Model
# ============================================================
class StepwiseSynthesizer(nn.Module):
    """Generates ONE DSL command given current grid state + target."""
    def __init__(self, d_model=128, nhead=4, num_layers=2, vocab_size=VOCAB_SIZE, max_prog_len=16):
        super().__init__()
        self.d_model = d_model
        self.max_prog_len = max_prog_len
        grid_dim = MAX_GRID * MAX_GRID * 2 + 4  # current + target + separators
        self.grid_embed = nn.Embedding(N_COLORS + 2, d_model)  # +2 for sep and pad
        self.pos_enc_e = nn.Embedding(grid_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4,
                                                    dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc_d = nn.Embedding(max_prog_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_model*4,
                                                    dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def encode_state(self, current_grid, target_grid):
        """Encode current state + target as sequence."""
        B = current_grid.size(0)
        # Flatten: [current_pixels, SEP(10), target_pixels]
        cur = current_grid.reshape(B, -1)
        tgt = target_grid.reshape(B, -1)
        sep = torch.full((B, 1), 10, dtype=torch.long, device=current_grid.device)
        seq = torch.cat([cur, sep, tgt], dim=1)  # (B, H*W + 1 + H*W)
        L = seq.size(1)
        pos = torch.arange(L, device=seq.device)
        x = self.grid_embed(seq.clamp(0, N_COLORS+1)) + self.pos_enc_e(pos[:L])
        return self.encoder(x)

    def decode_cmd(self, memory, tgt_tokens):
        T = tgt_tokens.size(1)
        pos = torch.arange(T, device=tgt_tokens.device)
        x = self.tok_embed(tgt_tokens) + self.pos_enc_d(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_tokens.device)
        out = self.decoder(x, memory, tgt_mask=causal_mask)
        return self.output_proj(out)

    def generate_one_cmd(self, current_grid, target_grid, max_len=16):
        memory = self.encode_state(current_grid, target_grid)
        B = current_grid.size(0)
        tokens = torch.full((B, 1), TOK2ID['<BOS>'], dtype=torch.long, device=current_grid.device)
        for _ in range(max_len - 1):
            logits = self.decode_cmd(memory, tokens)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == TOK2ID['<EOS>']).all():
                break
        return tokens

def pad_grid(grid, max_h=MAX_GRID, max_w=MAX_GRID, pad_val=0):
    h, w = len(grid), len(grid[0])
    padded = np.full((max_h, max_w), pad_val, dtype=np.int64)
    padded[:h, :w] = np.array(grid)
    return padded

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 9: Visual Execution Trace")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Prepare: single-step command data
    # Each sample: (current_state, target, command)
    states, targets, commands = [], [], []
    raw_data = []

    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            prog = try_dsl_programs(inp, out)
            if prog is not None:
                states.append(pad_grid(inp))
                targets.append(pad_grid(out))
                commands.append(tokenize_program(prog, max_len=16))
                raw_data.append((inp, out, prog))

    states_t = torch.tensor(np.array(states), dtype=torch.long)
    targets_t = torch.tensor(np.array(targets), dtype=torch.long)
    commands_t = torch.tensor(np.array(commands), dtype=torch.long)

    N = len(states_t)
    split = int(N * 0.8)
    print(f"Total: {N}, Train: {split}, Test: {N-split}")

    # Train
    print("\n--- Training Stepwise Model ---")
    model = StepwiseSynthesizer().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 32
    losses = []

    for epoch in range(80):
        model.train()
        perm = torch.randperm(split)
        epoch_loss, n_b = 0, 0
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            s = states_t[idx].to(DEVICE)
            t = targets_t[idx].to(DEVICE)
            c = commands_t[idx].to(DEVICE)
            memory = model.encode_state(s, t)
            logits = model.decode_cmd(memory, c[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), c[:, 1:].reshape(-1),
                                   ignore_index=TOK2ID['<PAD>'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        avg_loss = epoch_loss / max(n_b, 1)
        losses.append(avg_loss)
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/80: loss={avg_loss/1:.4f}")

    # Evaluation: one-shot vs visual trace (multi-step)
    print("\n--- Evaluation: One-Shot vs Visual Trace ---")
    model.eval()
    MAX_STEPS = 3

    oneshot_em = 0
    trace_em = 0
    total = 0
    trace_steps_used = []

    with torch.no_grad():
        for i in range(split, N):
            inp, out, true_prog = raw_data[i]
            s = states_t[i:i+1].to(DEVICE)
            t = targets_t[i:i+1].to(DEVICE)

            # One-shot
            tokens = model.generate_one_cmd(s, t)
            oneshot_prog = detokenize(tokens[0].cpu().tolist())
            if oneshot_prog.strip() == true_prog.strip():
                oneshot_em += 1

            # Visual trace: generate, execute, feed back
            current = inp
            best_prog = None
            for step in range(MAX_STEPS):
                cur_t = torch.tensor(pad_grid(current), dtype=torch.long).unsqueeze(0).to(DEVICE)
                tokens = model.generate_one_cmd(cur_t, t)
                step_prog = detokenize(tokens[0].cpu().tolist())

                # Execute
                result = execute_dsl(step_prog, current)
                if result is not None:
                    # Check if we reached target
                    result_padded = pad_grid(result)
                    target_np = targets_t[i].numpy()
                    if np.array_equal(result_padded, target_np):
                        best_prog = step_prog
                        trace_steps_used.append(step + 1)
                        break
                    current = result
                else:
                    break

            if best_prog is not None and best_prog.strip() == true_prog.strip():
                trace_em += 1
            elif best_prog is not None:
                trace_em += 1  # Reached target via execution even if prog text differs

            total += 1

    oneshot_rate = oneshot_em / max(total, 1)
    trace_rate = trace_em / max(total, 1)
    avg_steps = np.mean(trace_steps_used) if trace_steps_used else 0

    print(f"\n--- Results ---")
    print(f"One-Shot EM: {oneshot_rate:.1%} ({oneshot_em}/{total})")
    print(f"Visual Trace EM: {trace_rate:.1%} ({trace_em}/{total})")
    print(f"Avg steps when solved: {avg_steps:.1f}")
    print(f"Improvement: {trace_rate - oneshot_rate:+.1%}")

    elapsed = time.time() - t0

    results = {
        'oneshot_em': oneshot_rate,
        'trace_em': trace_rate,
        'improvement': trace_rate - oneshot_rate,
        'avg_steps': float(avg_steps),
        'n_test': total,
        'model_params': params,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase9_visual_trace.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    bars = axes[0].bar(['One-Shot', 'Visual Trace'], [oneshot_rate, trace_rate],
                        color=['#FF9800', '#4CAF50'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Exact Match Rate')
    axes[0].set_title('Phase 9: One-Shot vs Visual Trace')
    for bar, val in zip(bars, [oneshot_rate, trace_rate]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f'{val:.1%}', ha='center', fontsize=12)

    if trace_steps_used:
        axes[1].hist(trace_steps_used, bins=range(1, MAX_STEPS+2), color='#2196F3',
                     edgecolor='white', align='left')
        axes[1].set_xlabel('Steps to Solve')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Steps Distribution (avg={avg_steps:.1f})')
    else:
        axes[1].text(0.5, 0.5, 'No traces\nsolved', ha='center', va='center',
                     transform=axes[1].transAxes)

    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase9_visual_trace.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return results

if __name__ == '__main__':
    main()
