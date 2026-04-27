"""
Phase 6: Algorithmic Cross-Attention Grounding
================================================
Visualize what the Transformer "sees" when writing each DSL token.

For the Phase 1 Program Synthesizer:
1. Feed a task through the encoder
2. Generate DSL tokens autoregressively
3. At each decode step, extract the cross-attention map from decoder -> encoder
4. Reshape the attention map back to the input grid shape
5. Create heatmap overlays showing which pixels the model attends to
   when generating each DSL token

This is the ultimate white-box: we can literally SEE the AI's reasoning
process as it writes code, token by token.
"""

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# Modified Transformer with attention capture
# ============================================================
class AttentionCaptureSynthesizer(nn.Module):
    """Program Synthesizer with cross-attention capture."""
    def __init__(self, grid_dim=256, d_model=128, nhead=4, num_layers=3,
                 vocab_size=VOCAB_SIZE, max_prog_len=32):
        super().__init__()
        self.d_model = d_model
        self.max_prog_len = max_prog_len
        self.nhead = nhead
        
        # Encoder
        self.grid_embed = nn.Embedding(101, d_model)
        self.pos_enc_e = nn.Embedding(grid_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                                    dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder with explicit cross-attention for capture
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc_d = nn.Embedding(max_prog_len, d_model)
        
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }))
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._cross_attention_weights = []
    
    def encode(self, grid_seq):
        pos = torch.arange(grid_seq.size(1), device=grid_seq.device)
        x = self.grid_embed(grid_seq.clamp(0, 100)) + self.pos_enc_e(pos)
        return self.encoder(x)
    
    def decode(self, memory, tgt_tokens, capture_attention=False):
        T = tgt_tokens.size(1)
        pos = torch.arange(T, device=tgt_tokens.device)
        x = self.tok_embed(tgt_tokens) + self.pos_enc_d(pos)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_tokens.device)
        
        self._cross_attention_weights = []
        
        for layer in self.decoder_layers:
            # Self-attention
            normed = layer['norm1'](x)
            attn_out, _ = layer['self_attn'](normed, normed, normed, attn_mask=causal_mask)
            x = x + attn_out
            
            # Cross-attention (encoder-decoder)
            normed = layer['norm2'](x)
            attn_out, attn_weights = layer['cross_attn'](
                normed, memory, memory,
                need_weights=True, average_attn_weights=False
            )
            x = x + attn_out
            
            if capture_attention:
                self._cross_attention_weights.append(attn_weights.detach())
            
            # Feedforward
            normed = layer['norm3'](x)
            x = x + layer['ff'](normed)
        
        return self.output_proj(x)
    
    def generate_with_attention(self, grid_seq, max_len=32):
        """Autoregressive generation with attention capture."""
        memory = self.encode(grid_seq)
        B = grid_seq.size(0)
        tokens = torch.full((B, 1), TOK2ID['<BOS>'], dtype=torch.long, device=grid_seq.device)
        
        all_attention_maps = []
        
        for step in range(max_len - 1):
            logits = self.decode(memory, tokens, capture_attention=True)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            
            # Capture the last layer's cross-attention for the current step
            if self._cross_attention_weights:
                # Shape: (B, nhead, T_dec, T_enc) - take last decode position
                last_layer_attn = self._cross_attention_weights[-1]
                # Average over heads, take last decode position
                step_attn = last_layer_attn[:, :, -1, :].mean(dim=1)  # (B, T_enc)
                all_attention_maps.append(step_attn.cpu())
            
            if (next_tok == TOK2ID['<EOS>']).all():
                break
        
        return tokens, all_attention_maps

# ============================================================
# Visualization
# ============================================================
def visualize_attention_grounding(inp_grid, generated_tokens, attention_maps, 
                                  grid_seq_len, save_path):
    """Create a visualization showing attention heatmaps for each DSL token."""
    h, w = len(inp_grid), len(inp_grid[0])
    tokens = [ID2TOK.get(t, '?') for t in generated_tokens]
    
    # Filter to meaningful tokens only (skip BOS, match attention index)
    meaningful = []
    attn_idx = 0
    for i, tok in enumerate(tokens):
        if tok == '<EOS>':
            break
        if tok in ('<BOS>', '<PAD>'):
            attn_idx += 1
            continue
        if attn_idx < len(attention_maps):
            meaningful.append((tok, attention_maps[attn_idx]))
        attn_idx += 1
    
    if not meaningful:
        return
    
    n_tokens = min(len(meaningful), 6)  # Show at most 6 tokens
    
    fig, axes = plt.subplots(2, max(n_tokens, 1), figsize=(4 * max(n_tokens, 1), 8))
    if n_tokens == 1:
        axes = axes.reshape(2, 1)
    
    # Row 1: Input grid with attention overlay
    arc_cmap = mcolors.ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    
    for i in range(n_tokens):
        tok, attn = meaningful[i]
        
        # Top row: input grid
        axes[0, i].imshow(inp_grid, cmap=arc_cmap, vmin=0, vmax=9, aspect='equal')
        axes[0, i].set_title(f'Token: "{tok}"', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')
        
        # Bottom row: attention heatmap
        attn_vals = attn.numpy()
        # Squeeze batch dim if present
        while attn_vals.ndim > 1:
            attn_vals = attn_vals[0]
        
        # The grid_seq layout: [h_val, w_val, pixel_0, pixel_1, ..., pixel_{h*w-1}, 99(sep), ...]
        grid_start = 2
        grid_end = grid_start + h * w
        
        if grid_end <= len(attn_vals):
            grid_attn = attn_vals[grid_start:grid_end]
        else:
            # Use whatever we can get and pad
            usable = attn_vals[grid_start:min(grid_end, len(attn_vals))]
            grid_attn = np.zeros(h * w)
            grid_attn[:len(usable)] = usable
        
        grid_attn = grid_attn.reshape(h, w)
        
        # Normalize
        if grid_attn.max() > grid_attn.min():
            grid_attn = (grid_attn - grid_attn.min()) / (grid_attn.max() - grid_attn.min())
        
        axes[1, i].imshow(inp_grid, cmap=arc_cmap, vmin=0, vmax=9, aspect='equal', alpha=0.3)
        im = axes[1, i].imshow(grid_attn, cmap='hot', alpha=0.7, aspect='equal')
        axes[1, i].set_title(f'Attention for "{tok}"', fontsize=10)
        axes[1, i].axis('off')
    
    # Hide unused axes
    for i in range(n_tokens, axes.shape[1]):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    plt.suptitle('Cross-Attention Grounding: Where does the AI look when writing code?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 6: Algorithmic Cross-Attention Grounding")
    print("=" * 60)
    t0 = time.time()
    
    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")
    
    # Prepare dataset
    grid_seqs = []
    prog_seqs = []
    raw_pairs = []
    
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
    print(f"Train: {split}, Test: {N-split}")
    
    # Train model
    print("\n--- Training Attention-Capture Model ---")
    model = AttentionCaptureSynthesizer().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH = 32
    losses = []
    
    for epoch in range(80):
        model.train()
        perm = torch.randperm(split)
        epoch_loss = 0
        n_b = 0
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
        avg_loss = epoch_loss / max(n_b, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/80: loss={avg_loss:.4f}")
    
    # Generate with attention capture
    print("\n--- Generating with Attention Capture ---")
    model.eval()
    
    grounding_examples = []
    n_correct = 0
    n_total = 0
    n_grounded = 0  # tokens where attention peak matches semantically
    n_grounding_total = 0
    
    with torch.no_grad():
        for i in range(split, min(N, split + 50)):
            inp, out, true_prog = raw_pairs[i]
            g = grid_seqs[i:i+1].to(DEVICE)
            
            generated, attn_maps = model.generate_with_attention(g)
            pred_str = detokenize(generated[0].cpu().tolist())
            
            n_total += 1
            is_match = pred_str.strip() == true_prog.strip()
            if is_match:
                n_correct += 1
            
            # Analyze attention grounding
            gen_tokens = generated[0].cpu().tolist()
            for j, tok_id in enumerate(gen_tokens):
                tok = ID2TOK.get(tok_id, '')
                if tok in DSL_OPS and j < len(attn_maps):
                    attn = attn_maps[j][0].numpy()  # (T_enc,)
                    h, w = len(inp), len(inp[0])
                    grid_start = 2
                    grid_end = grid_start + h * w
                    if grid_end <= len(attn):
                        grid_attn = attn[grid_start:grid_end]
                        peak_pixel = np.argmax(grid_attn)
                        peak_r, peak_c = peak_pixel // w, peak_pixel % w
                        
                        # Check if peak is on a non-background pixel (semantic grounding)
                        bg = int(np.bincount(np.array(inp).flatten()).argmax())
                        is_grounded = inp[peak_r][peak_c] != bg
                        n_grounding_total += 1
                        if is_grounded:
                            n_grounded += 1
            
            # Save example for visualization
            if i < split + 5:
                grounding_examples.append({
                    'idx': i,
                    'true_prog': true_prog,
                    'pred_prog': pred_str,
                    'match': is_match,
                    'inp': inp,
                    'gen_tokens': gen_tokens,
                    'attn_maps': attn_maps,
                })
    
    em_rate = n_correct / max(n_total, 1)
    grounding_rate = n_grounded / max(n_grounding_total, 1)
    
    print(f"\n--- Results ---")
    print(f"Exact Match: {em_rate:.1%} ({n_correct}/{n_total})")
    print(f"Visual Grounding Rate: {grounding_rate:.1%} ({n_grounded}/{n_grounding_total})")
    print(f"  (fraction of DSL op tokens where attention peak is on a non-bg pixel)")
    
    # Visualize examples
    print("\n--- Generating Visualizations ---")
    for j, ex in enumerate(grounding_examples[:3]):
        save_path = os.path.join(FIGURES_DIR, f'phase6_grounding_example_{j+1}.png')
        visualize_attention_grounding(
            ex['inp'], ex['gen_tokens'], ex['attn_maps'],
            256, save_path
        )
        print(f"  Saved: {save_path}")
        print(f"    True:  {ex['true_prog']}")
        print(f"    Pred:  {ex['pred_prog']}")
        print(f"    Match: {ex['match']}")
    
    elapsed = time.time() - t0
    
    # Save results
    out = {
        'exact_match_rate': em_rate,
        'visual_grounding_rate': grounding_rate,
        'n_grounded': n_grounded,
        'n_grounding_total': n_grounding_total,
        'n_correct': n_correct,
        'n_total': n_total,
        'model_params': params,
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase6_attention_grounding.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training loss
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # EM and Grounding rates
    bars = axes[1].bar(['Exact Match', 'Visual Grounding'], [em_rate, grounding_rate],
                        color=['#4CAF50', '#2196F3'])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Rate')
    axes[1].set_title('Phase 6: Code Generation Quality')
    for bar, val in zip(bars, [em_rate, grounding_rate]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.1%}', ha='center', fontsize=12)
    
    # Comparison with Phase 1
    p1_path = os.path.join(RESULTS_DIR, 'phase1_neural_program_synthesis.json')
    if os.path.exists(p1_path):
        with open(p1_path, 'r', encoding='utf-8') as f:
            p1 = json.load(f)
        p1_em = p1.get('exec_match_rate', 0)
        axes[2].bar(['Phase 1\n(base)', 'Phase 6\n(attention)'], [p1_em, em_rate],
                     color=['#FF9800', '#4CAF50'])
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel('Exact Match Rate')
        axes[2].set_title('Phase 1 vs Phase 6 EM')
    else:
        axes[2].text(0.5, 0.5, 'Phase 1 results\nnot found', ha='center', va='center',
                     transform=axes[2].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase6_attention_grounding.png'), dpi=150)
    plt.close()
    
    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
