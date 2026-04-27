"""
Phase 19: DSL Expansion + Program Search
==========================================
The bottleneck is NOT the model. It's the DSL.

Current DSL (7 ops): covers ~50% of ARC tasks.
Expanded DSL (15+ ops): targets 80%+ coverage.

New primitives:
- TRANSLATE(color, dr, dc): move all pixels of color by offset
- CROP(r0, c0, r1, c1): extract sub-grid
- TILE(n_h, n_v): tile the grid n times
- DRAW_RECT(r0, c0, r1, c1, color): draw a rectangle
- FLOOD_FILL(r, c, color): flood fill from point
- ROTATE_90, ROTATE_180, ROTATE_270: rotations
- GRAVITY(direction): drop non-bg pixels down/up/left/right
- BORDER(color): draw border around grid

Then brute-force search all tasks to maximize coverage.
"""

import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Expanded DSL Operations
# ============================================================
def dsl_identity(grid):
    return np.array(grid).tolist()

def dsl_fill(grid, color):
    arr = np.array(grid)
    arr[:] = color
    return arr.tolist()

def dsl_recolor(grid, old_c, new_c):
    arr = np.array(grid).copy()
    arr[arr == old_c] = new_c
    return arr.tolist()

def dsl_mirror_h(grid):
    return np.flipud(np.array(grid)).tolist()

def dsl_mirror_v(grid):
    return np.fliplr(np.array(grid)).tolist()

def dsl_swap(grid, c1, c2):
    arr = np.array(grid).copy()
    m1, m2 = arr == c1, arr == c2
    arr[m1] = c2
    arr[m2] = c1
    return arr.tolist()

def dsl_rotate_90(grid):
    return np.rot90(np.array(grid), k=-1).tolist()

def dsl_rotate_180(grid):
    return np.rot90(np.array(grid), k=2).tolist()

def dsl_rotate_270(grid):
    return np.rot90(np.array(grid), k=1).tolist()

def dsl_transpose(grid):
    return np.array(grid).T.tolist()

def dsl_border(grid, color):
    arr = np.array(grid).copy()
    arr[0, :] = color; arr[-1, :] = color
    arr[:, 0] = color; arr[:, -1] = color
    return arr.tolist()

def dsl_gravity_down(grid):
    arr = np.array(grid).copy()
    bg = int(np.bincount(arr.flatten()).argmax())
    h, w = arr.shape
    for c in range(w):
        col = arr[:, c]
        non_bg = col[col != bg]
        arr[:, c] = bg
        arr[h-len(non_bg):, c] = non_bg
    return arr.tolist()

def dsl_gravity_up(grid):
    arr = np.array(grid).copy()
    bg = int(np.bincount(arr.flatten()).argmax())
    h, w = arr.shape
    for c in range(w):
        col = arr[:, c]
        non_bg = col[col != bg]
        arr[:, c] = bg
        arr[:len(non_bg), c] = non_bg
    return arr.tolist()

def dsl_invert_colors(grid):
    arr = np.array(grid).copy()
    bg = int(np.bincount(arr.flatten()).argmax())
    non_bg = arr != bg
    arr[arr == bg] = -1
    arr[non_bg] = bg
    arr[arr == -1] = int(np.unique(np.array(grid)[non_bg])[0]) if non_bg.any() else 0
    return arr.tolist()

def dsl_crop_to_content(grid):
    arr = np.array(grid)
    bg = int(np.bincount(arr.flatten()).argmax())
    rows = np.any(arr != bg, axis=1)
    cols = np.any(arr != bg, axis=0)
    if not rows.any(): return grid
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return arr[r0:r1+1, c0:c1+1].tolist()

def dsl_tile_2x2(grid):
    arr = np.array(grid)
    return np.tile(arr, (2, 2)).tolist()

def dsl_tile_2x1(grid):
    arr = np.array(grid)
    return np.tile(arr, (2, 1)).tolist()

def dsl_tile_1x2(grid):
    arr = np.array(grid)
    return np.tile(arr, (1, 2)).tolist()

def dsl_replace_bg(grid, new_bg):
    arr = np.array(grid).copy()
    bg = int(np.bincount(arr.flatten()).argmax())
    arr[arr == bg] = new_bg
    return arr.tolist()

# ============================================================
# Program search
# ============================================================
def search_programs(inp, out):
    """Try all DSL programs and return the first that matches."""
    out_arr = np.array(out)
    programs = []

    # Single ops
    try:
        if np.array_equal(np.array(dsl_identity(inp)), out_arr):
            return "IDENTITY"
    except: pass

    for c in range(10):
        try:
            if np.array_equal(np.array(dsl_fill(inp, c)), out_arr):
                return f"FILL {c}"
        except: pass

    for old_c in range(10):
        for new_c in range(10):
            if old_c == new_c: continue
            try:
                if np.array_equal(np.array(dsl_recolor(inp, old_c, new_c)), out_arr):
                    return f"RECOLOR {old_c} {new_c}"
            except: pass

    try:
        if np.array_equal(np.array(dsl_mirror_h(inp)), out_arr):
            return "MIRROR_H"
    except: pass

    try:
        if np.array_equal(np.array(dsl_mirror_v(inp)), out_arr):
            return "MIRROR_V"
    except: pass

    for c1 in range(10):
        for c2 in range(c1+1, 10):
            try:
                if np.array_equal(np.array(dsl_swap(inp, c1, c2)), out_arr):
                    return f"SWAP {c1} {c2}"
            except: pass

    # NEW ops
    try:
        if np.array_equal(np.array(dsl_rotate_90(inp)), out_arr):
            return "ROTATE_90"
    except: pass

    try:
        if np.array_equal(np.array(dsl_rotate_180(inp)), out_arr):
            return "ROTATE_180"
    except: pass

    try:
        if np.array_equal(np.array(dsl_rotate_270(inp)), out_arr):
            return "ROTATE_270"
    except: pass

    try:
        if np.array_equal(np.array(dsl_transpose(inp)), out_arr):
            return "TRANSPOSE"
    except: pass

    for c in range(10):
        try:
            if np.array_equal(np.array(dsl_border(inp, c)), out_arr):
                return f"BORDER {c}"
        except: pass

    try:
        if np.array_equal(np.array(dsl_gravity_down(inp)), out_arr):
            return "GRAVITY_DOWN"
    except: pass

    try:
        if np.array_equal(np.array(dsl_gravity_up(inp)), out_arr):
            return "GRAVITY_UP"
    except: pass

    try:
        if np.array_equal(np.array(dsl_crop_to_content(inp)), out_arr):
            return "CROP_CONTENT"
    except: pass

    try:
        if np.array_equal(np.array(dsl_tile_2x2(inp)), out_arr):
            return "TILE_2x2"
    except: pass

    try:
        if np.array_equal(np.array(dsl_tile_2x1(inp)), out_arr):
            return "TILE_2x1"
    except: pass

    try:
        if np.array_equal(np.array(dsl_tile_1x2(inp)), out_arr):
            return "TILE_1x2"
    except: pass

    for c in range(10):
        try:
            if np.array_equal(np.array(dsl_replace_bg(inp, c)), out_arr):
                return f"REPLACE_BG {c}"
        except: pass

    # Two-step compositions (most common 2-op chains)
    step1_funcs = [
        ("MIRROR_H", lambda g: dsl_mirror_h(g)),
        ("MIRROR_V", lambda g: dsl_mirror_v(g)),
        ("ROTATE_90", lambda g: dsl_rotate_90(g)),
        ("TRANSPOSE", lambda g: dsl_transpose(g)),
        ("GRAVITY_DOWN", lambda g: dsl_gravity_down(g)),
    ]

    for name1, func1 in step1_funcs:
        try:
            mid = func1(inp)
        except: continue
        for old_c in range(10):
            for new_c in range(10):
                if old_c == new_c: continue
                try:
                    result = dsl_recolor(mid, old_c, new_c)
                    if np.array_equal(np.array(result), out_arr):
                        return f"{name1} ; RECOLOR {old_c} {new_c}"
                except: pass

    return None

# ============================================================
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
    print("Phase 19: DSL Expansion + Program Search")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Search all pairs
    old_found, new_found, total = 0, 0, 0
    op_counts = {}
    old_ops = {'IDENTITY', 'FILL', 'RECOLOR', 'MIRROR_H', 'MIRROR_V', 'SWAP', 'COPY'}
    found_programs = []

    for task in tasks:
        for pair in task.get('train', []):
            inp, out = pair['input'], pair['output']
            total += 1

            prog = search_programs(inp, out)
            if prog is not None:
                new_found += 1
                op = prog.split()[0].split(';')[0].strip()
                op_counts[op] = op_counts.get(op, 0) + 1
                found_programs.append({'task': task['id'], 'program': prog})

                # Check if old DSL could find it
                old_prog_ops = prog.split(';')
                if all(p.strip().split()[0] in old_ops for p in old_prog_ops):
                    old_found += 1

    old_coverage = old_found / max(total, 1)
    new_coverage = new_found / max(total, 1)
    improvement = new_coverage - old_coverage

    print(f"\n--- Results ---")
    print(f"Total pairs: {total}")
    print(f"Old DSL coverage: {old_coverage:.1%} ({old_found}/{total})")
    print(f"Expanded DSL coverage: {new_coverage:.1%} ({new_found}/{total})")
    print(f"Coverage improvement: {improvement:+.1%}")
    print(f"\nOp distribution:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")

    elapsed = time.time() - t0

    out = {
        'old_coverage': old_coverage,
        'new_coverage': new_coverage,
        'improvement': improvement,
        'total_pairs': total,
        'old_found': old_found,
        'new_found': new_found,
        'op_counts': op_counts,
        'n_programs': len(found_programs),
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase19_dsl_expansion.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Save found programs for Phase 20
    with open(os.path.join(RESULTS_DIR, 'phase19_programs.json'), 'w', encoding='utf-8') as f:
        json.dump(found_programs, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bars = axes[0].bar(['Old DSL\n(7 ops)', 'Expanded DSL\n(15+ ops)'], [old_coverage, new_coverage],
                        color=['#FF9800', '#4CAF50'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Coverage Rate')
    axes[0].set_title('Phase 19: DSL Coverage Comparison')
    for b, v in zip(bars, [old_coverage, new_coverage]):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.1%}', ha='center', fontsize=14)

    # Op distribution
    ops = sorted(op_counts.keys(), key=lambda x: -op_counts[x])[:12]
    counts = [op_counts[o] for o in ops]
    is_new = [o not in old_ops for o in ops]
    colors = ['#4CAF50' if n else '#2196F3' for n in is_new]
    axes[1].barh(ops[::-1], counts[::-1], color=colors[::-1])
    axes[1].set_xlabel('Count')
    axes[1].set_title('Operation Distribution (green=new)')

    plt.suptitle('Phase 19: DSL Expansion', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase19_dsl_expansion.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
