"""
Phase 5: Execution-Guided Self-Debugging
==========================================
Boost Phase 1's 47.4% EM via test-time compute.

Pipeline:
1. Generate DSL program (Phase 1 model)
2. Execute it on training demos
3. Compare output with ground truth -> error mask
4. Feed error info back to model -> generate corrected program
5. Repeat up to 3 refinement steps

This is "Test-Time Compute" applied to program synthesis:
the model writes code, runs it, sees the bug, and fixes it.
"""

import os, sys, json, time, random, math, copy
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

# Import Phase 1 components
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.phase1_neural_program_synthesis import (
    DSL_OPS, DSL_VOCAB, TOK2ID, ID2TOK, VOCAB_SIZE,
    encode_grid_pair, tokenize_program, detokenize, try_dsl_programs,
    ProgramSynthesizer, load_arc_tasks
)

# ============================================================
# DSL Executor
# ============================================================
def execute_dsl(program_str, input_grid):
    """Execute a DSL program on an input grid. Returns output grid or None."""
    inp = np.array(input_grid)
    h, w = inp.shape
    out = inp.copy()
    
    parts = program_str.split()
    if not parts:
        return None
    
    op = parts[0]
    try:
        if op == 'IDENTITY':
            return out.tolist()
        elif op == 'FILL' and len(parts) >= 2:
            color = int(parts[1])
            out[:] = color
            return out.tolist()
        elif op == 'RECOLOR' and len(parts) >= 3:
            old_color = int(parts[1])
            new_color = int(parts[2])
            out[out == old_color] = new_color
            return out.tolist()
        elif op == 'MIRROR_H':
            return np.flipud(inp).tolist()
        elif op == 'MIRROR_V':
            return np.fliplr(inp).tolist()
        elif op == 'COPY' and len(parts) >= 7:
            # COPY r0 c0 h w dr dc
            return out.tolist()
        elif op == 'SWAP' and len(parts) >= 3:
            c1, c2 = int(parts[1]), int(parts[2])
            mask1 = inp == c1
            mask2 = inp == c2
            out[mask1] = c2
            out[mask2] = c1
            return out.tolist()
    except (ValueError, IndexError):
        return None
    
    return out.tolist()

def compute_error_mask(predicted_grid, target_grid):
    """Compute error mask between predicted and target grids."""
    pred = np.array(predicted_grid)
    target = np.array(target_grid)
    
    if pred.shape != target.shape:
        # Size mismatch - return full error
        return np.ones_like(target), 1.0
    
    errors = (pred != target).astype(np.float32)
    error_rate = errors.mean()
    return errors, error_rate

def encode_with_error(inp_grid, out_grid, error_mask, max_len=256):
    """Encode grid pair + error feedback as integer sequence."""
    flat_in = []
    h_in, w_in = len(inp_grid), len(inp_grid[0])
    flat_in.extend([h_in, w_in])
    for row in inp_grid:
        flat_in.extend(row)
    
    flat_in.append(99)  # separator
    
    flat_out = []
    h_out, w_out = len(out_grid), len(out_grid[0])
    flat_out.extend([h_out, w_out])
    for row in out_grid:
        flat_out.extend(row)
    flat_in.extend(flat_out)
    
    flat_in.append(98)  # error separator
    
    # Flatten error mask (quantized to 0/1 -> 50/51)
    err = np.array(error_mask).flatten()
    for e in err[:50]:  # Limit error info
        flat_in.append(50 + int(e))
    
    if len(flat_in) > max_len:
        flat_in = flat_in[:max_len]
    else:
        flat_in = flat_in + [0] * (max_len - len(flat_in))
    
    return flat_in

# ============================================================
# Refinement Model (extends Phase 1)
# ============================================================
class RefinementSynthesizer(ProgramSynthesizer):
    """Phase 1 model extended with error-conditioned refinement."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Additional error embedding for tokens 50-51 (error mask values)
        # Reuse existing grid_embed which handles 0-100

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 5: Execution-Guided Self-Debugging")
    print("=" * 60)
    t0 = time.time()
    
    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")
    
    # Prepare dataset (same as Phase 1)
    grid_seqs = []
    prog_seqs = []
    raw_pairs = []  # Keep raw grids for execution
    
    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            prog = try_dsl_programs(inp, out)
            if prog is not None:
                grid_seq = encode_grid_pair(inp, out)
                prog_tok = tokenize_program(prog)
                grid_seqs.append(grid_seq)
                prog_seqs.append(prog_tok)
                raw_pairs.append((inp, out, prog))
    
    grid_seqs = torch.tensor(grid_seqs, dtype=torch.long)
    prog_seqs = torch.tensor(prog_seqs, dtype=torch.long)
    
    N = len(grid_seqs)
    split = int(N * 0.8)
    train_g, test_g = grid_seqs[:split], grid_seqs[split:]
    train_p, test_p = prog_seqs[:split], prog_seqs[split:]
    test_raw = raw_pairs[split:]
    
    print(f"Train: {split}, Test: {N-split}")
    
    # Train base model (same as Phase 1 but fewer epochs for speed)
    print("\n--- Training Base Model ---")
    model = RefinementSynthesizer().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 32
    
    for epoch in range(60):
        model.train()
        perm = torch.randperm(len(train_g))
        epoch_loss = 0
        n_b = 0
        for i in range(0, len(train_g), BATCH):
            idx = perm[i:i+BATCH]
            g = train_g[idx].to(DEVICE)
            p = train_p[idx].to(DEVICE)
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
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/60: loss={epoch_loss/n_b:.4f}")
    
    # Train refinement model on error-conditioned inputs
    print("\n--- Training Refinement Head ---")
    # Generate training data for refinement: intentionally wrong programs + error feedback
    ref_grid_seqs = []
    ref_prog_seqs = []
    
    for i in range(min(split, 500)):
        inp, out, true_prog = raw_pairs[i]
        
        # Generate a wrong program by corruption
        corrupted = true_prog
        parts = true_prog.split()
        if len(parts) >= 2 and random.random() < 0.5:
            # Corrupt a parameter
            new_parts = parts.copy()
            idx_to_corrupt = random.randint(1, len(new_parts)-1)
            try:
                old_val = int(new_parts[idx_to_corrupt])
                new_parts[idx_to_corrupt] = str((old_val + random.randint(1, 5)) % 10)
                corrupted = ' '.join(new_parts)
            except ValueError:
                pass
        
        # Execute corrupted program
        exec_result = execute_dsl(corrupted, inp)
        if exec_result is not None:
            error_mask, error_rate = compute_error_mask(exec_result, out)
            if error_rate > 0:
                # Create error-conditioned input
                err_seq = encode_with_error(inp, out, error_mask)
                ref_grid_seqs.append(err_seq)
                ref_prog_seqs.append(tokenize_program(true_prog))
    
    if ref_grid_seqs:
        ref_g = torch.tensor(ref_grid_seqs, dtype=torch.long)
        ref_p = torch.tensor(ref_prog_seqs, dtype=torch.long)
        print(f"Refinement training pairs: {len(ref_g)}")
        
        # Fine-tune model on error-conditioned data
        for epoch in range(30):
            model.train()
            perm = torch.randperm(len(ref_g))
            for i in range(0, len(ref_g), BATCH):
                idx = perm[i:i+BATCH]
                g = ref_g[idx].to(DEVICE)
                p = ref_p[idx].to(DEVICE)
                memory = model.encode(g)
                logits = model.decode(memory, p[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), p[:, 1:].reshape(-1),
                                       ignore_index=TOK2ID['<PAD>'])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    
    # Evaluation with self-debugging loop
    print("\n--- Evaluation: Self-Debugging Loop ---")
    model.eval()
    MAX_REFINEMENTS = 3
    
    results_per_step = {step: {'em': 0, 'total': 0, 'syntax_valid': 0} 
                        for step in range(MAX_REFINEMENTS + 1)}
    
    with torch.no_grad():
        for i in range(len(test_raw)):
            inp, out, true_prog = test_raw[i]
            
            # Step 0: Initial generation
            g = test_g[i:i+1].to(DEVICE)
            generated = model.generate(g)
            pred_str = detokenize(generated[0].cpu().tolist())
            
            parts = pred_str.split()
            is_valid = len(parts) > 0 and parts[0] in DSL_OPS
            is_match = pred_str.strip() == true_prog.strip()
            
            results_per_step[0]['total'] += 1
            if is_valid:
                results_per_step[0]['syntax_valid'] += 1
            if is_match:
                results_per_step[0]['em'] += 1
            
            # Refinement steps
            current_prog = pred_str
            for step in range(1, MAX_REFINEMENTS + 1):
                if is_match:
                    # Already correct, carry forward
                    results_per_step[step]['total'] += 1
                    results_per_step[step]['syntax_valid'] += 1
                    results_per_step[step]['em'] += 1
                    continue
                
                # Execute current program
                exec_result = execute_dsl(current_prog, inp)
                if exec_result is None:
                    exec_result = [[0] * len(inp[0]) for _ in range(len(inp))]
                
                error_mask, _ = compute_error_mask(exec_result, out)
                
                # Create error-conditioned input
                err_seq = encode_with_error(inp, out, error_mask)
                err_t = torch.tensor([err_seq], dtype=torch.long).to(DEVICE)
                
                # Generate corrected program
                generated = model.generate(err_t)
                current_prog = detokenize(generated[0].cpu().tolist())
                
                parts = current_prog.split()
                is_valid = len(parts) > 0 and parts[0] in DSL_OPS
                is_match = current_prog.strip() == true_prog.strip()
                
                results_per_step[step]['total'] += 1
                if is_valid:
                    results_per_step[step]['syntax_valid'] += 1
                if is_match:
                    results_per_step[step]['em'] += 1
    
    # Print results
    print("\n--- Results ---")
    em_rates = []
    syntax_rates = []
    for step in range(MAX_REFINEMENTS + 1):
        r = results_per_step[step]
        total = max(r['total'], 1)
        em = r['em'] / total
        syn = r['syntax_valid'] / total
        em_rates.append(em)
        syntax_rates.append(syn)
        label = "Initial" if step == 0 else f"Refine {step}"
        print(f"  {label}: EM={em:.1%}, Syntax={syn:.1%} (n={r['total']})")
    
    elapsed = time.time() - t0
    
    # Save results
    out_results = {
        'per_step': {str(k): v for k, v in results_per_step.items()},
        'em_rates': em_rates,
        'syntax_rates': syntax_rates,
        'max_refinements': MAX_REFINEMENTS,
        'model_params': params,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase5_self_debugging.json'), 'w', encoding='utf-8') as f:
        json.dump(out_results, f, indent=2, ensure_ascii=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    steps = ['Initial', 'Refine 1', 'Refine 2', 'Refine 3']
    
    axes[0].plot(range(len(em_rates)), em_rates, 'o-', color='#4CAF50', linewidth=2, markersize=8)
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels(steps)
    axes[0].set_ylabel('Exact Match Rate')
    axes[0].set_title('Self-Debugging: EM Improvement per Step')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    for i, em in enumerate(em_rates):
        axes[0].annotate(f'{em:.1%}', (i, em), textcoords="offset points", xytext=(0, 10), ha='center')
    
    # Improvement bars
    improvements = [0] + [em_rates[i] - em_rates[i-1] for i in range(1, len(em_rates))]
    colors = ['gray'] + ['#4CAF50' if d > 0 else '#F44336' for d in improvements[1:]]
    axes[1].bar(steps, improvements, color=colors)
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_ylabel('Delta EM')
    axes[1].set_title('EM Improvement per Refinement Step')
    for i, d in enumerate(improvements):
        if d != 0:
            axes[1].text(i, d + 0.005 * np.sign(d), f'{d:+.1%}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase5_self_debugging.png'), dpi=150)
    plt.close()
    
    print(f"\nElapsed: {elapsed:.1f}s")
    return out_results

if __name__ == '__main__':
    main()
