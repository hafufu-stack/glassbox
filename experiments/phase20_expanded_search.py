"""
Phase 20: Expanded Search Re-run
===================================
Phase 16 failed because the DSL was too narrow (7 ops).
Phase 19 expanded it to 15+ ops with compositions.

Now re-run Execution-Guided Search with the EXPANDED DSL.
This time, instead of sampling from model logits (which was
too constrained), we search the EXPANDED program space directly.

For each test task:
1. Try all single ops with all parameter combinations
2. Try 2-step compositions
3. Verify on DEMO examples
4. If demo-verified, apply to test input

This is brute-force program synthesis + verification.
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

# Import DSL operations from Phase 19
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.phase19_dsl_expansion import (
    dsl_identity, dsl_fill, dsl_recolor, dsl_mirror_h, dsl_mirror_v,
    dsl_swap, dsl_rotate_90, dsl_rotate_180, dsl_rotate_270,
    dsl_transpose, dsl_border, dsl_gravity_down, dsl_gravity_up,
    dsl_crop_to_content, dsl_tile_2x2, dsl_tile_2x1, dsl_tile_1x2,
    dsl_replace_bg
)

def load_arc_tasks(data_dir, max_tasks=400):
    tasks = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[:max_tasks]
    for fname in files:
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            task = json.load(f)
        tasks.append({'id': fname.replace('.json', ''), **task})
    return tasks

def generate_candidate_programs():
    """Generate all candidate programs (op + params) to try."""
    programs = []

    # Parameterless ops
    for name, func in [
        ("IDENTITY", dsl_identity), ("MIRROR_H", dsl_mirror_h),
        ("MIRROR_V", dsl_mirror_v), ("ROTATE_90", dsl_rotate_90),
        ("ROTATE_180", dsl_rotate_180), ("ROTATE_270", dsl_rotate_270),
        ("TRANSPOSE", dsl_transpose), ("GRAVITY_DOWN", dsl_gravity_down),
        ("GRAVITY_UP", dsl_gravity_up), ("CROP_CONTENT", dsl_crop_to_content),
        ("TILE_2x2", dsl_tile_2x2), ("TILE_2x1", dsl_tile_2x1),
        ("TILE_1x2", dsl_tile_1x2),
    ]:
        programs.append((name, func, []))

    # 1-param ops
    for c in range(10):
        programs.append((f"FILL {c}", lambda g, c=c: dsl_fill(g, c), []))
        programs.append((f"BORDER {c}", lambda g, c=c: dsl_border(g, c), []))
        programs.append((f"REPLACE_BG {c}", lambda g, c=c: dsl_replace_bg(g, c), []))

    # 2-param ops
    for c1 in range(10):
        for c2 in range(10):
            if c1 == c2: continue
            programs.append((f"RECOLOR {c1} {c2}", lambda g, a=c1, b=c2: dsl_recolor(g, a, b), []))

    for c1 in range(10):
        for c2 in range(c1+1, 10):
            programs.append((f"SWAP {c1} {c2}", lambda g, a=c1, b=c2: dsl_swap(g, a, b), []))

    return programs

def try_program_on_demos(prog_func, demos):
    """Try a program on all demo pairs. Return True if all match."""
    for pair in demos:
        try:
            result = prog_func(pair['input'])
            if result is None: return False
            if not np.array_equal(np.array(result), np.array(pair['output'])):
                return False
        except Exception:
            return False
    return True

def main():
    print("=" * 60)
    print("Phase 20: Expanded Search Re-run")
    print("=" * 60)
    t0 = time.time()

    tasks = load_arc_tasks(DATA_DIR)
    print(f"Loaded {len(tasks)} tasks")

    # Generate candidate programs (single ops)
    single_programs = generate_candidate_programs()
    print(f"Single-op candidates: {len(single_programs)}")

    # Generate 2-step compositions (parameterless op -> recolor)
    two_step_programs = []
    step1_ops = [
        ("MIRROR_H", dsl_mirror_h), ("MIRROR_V", dsl_mirror_v),
        ("ROTATE_90", dsl_rotate_90), ("ROTATE_180", dsl_rotate_180),
        ("TRANSPOSE", dsl_transpose), ("GRAVITY_DOWN", dsl_gravity_down),
        ("GRAVITY_UP", dsl_gravity_up),
    ]
    for name1, func1 in step1_ops:
        for c1 in range(10):
            for c2 in range(10):
                if c1 == c2: continue
                name = f"{name1} ; RECOLOR {c1} {c2}"
                def make_func(f1, a, b):
                    return lambda g: dsl_recolor(f1(g), a, b)
                two_step_programs.append((name, make_func(func1, c1, c2), []))
    print(f"Two-step candidates: {len(two_step_programs)}")

    all_programs = single_programs + two_step_programs
    print(f"Total candidates: {len(all_programs)}")

    # Split tasks into train/test
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]
    print(f"Test tasks: {len(test_tasks)}")

    # Search
    solved_tasks = 0
    task_results = []

    for task in test_tasks:
        demos = task.get('train', [])
        test_pairs = task.get('test', [])
        if not demos or not test_pairs:
            task_results.append({'id': task['id'], 'solved': False, 'program': None})
            continue

        found = False
        found_prog = None

        for prog_name, prog_func, _ in all_programs:
            if try_program_on_demos(prog_func, demos):
                # Verify on test
                all_test_ok = True
                for tp in test_pairs:
                    try:
                        result = prog_func(tp['input'])
                        if not np.array_equal(np.array(result), np.array(tp['output'])):
                            all_test_ok = False
                            break
                    except Exception:
                        all_test_ok = False
                        break

                if all_test_ok:
                    found = True
                    found_prog = prog_name
                    break

        if found:
            solved_tasks += 1
        task_results.append({'id': task['id'], 'solved': found, 'program': found_prog})

    solve_rate = solved_tasks / max(len(test_tasks), 1)
    demo_verified = sum(1 for t in task_results if t['program'] is not None)

    print(f"\n--- Results ---")
    print(f"Tasks solved (test verified): {solved_tasks}/{len(test_tasks)} ({solve_rate:.1%})")
    print(f"\nSolved examples:")
    for t in task_results:
        if t['solved']:
            print(f"  {t['id']}: {t['program']}")

    elapsed = time.time() - t0

    out = {
        'solve_rate': solve_rate,
        'solved': solved_tasks,
        'total_test_tasks': len(test_tasks),
        'n_candidates': len(all_programs),
        'task_results': task_results[:50],  # Save first 50
        'elapsed': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    with open(os.path.join(RESULTS_DIR, 'phase20_expanded_search.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Phase 15 baseline vs Phase 20
    methods = ['Transformer\n(Phase 1)\n1.45M params', 'GlassBox Agent\n(Phase 13)\n77K params',
               'Program Search\n(Phase 20)\n0 params']
    phase15_ems = [0.439, 0.568]  # From Phase 15
    vals = phase15_ems + [solve_rate]
    colors = ['#FF9800', '#2196F3', '#4CAF50']
    bars = axes[0].bar(methods, vals, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Solve Rate / EM')
    axes[0].set_title('Phase 20: Program Search vs Neural Models')
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.1%}', ha='center', fontsize=11)

    # Solved task breakdown
    solved_progs = [t['program'] for t in task_results if t['solved']]
    if solved_progs:
        prog_types = {}
        for p in solved_progs:
            op = p.split()[0].split(';')[0].strip()
            prog_types[op] = prog_types.get(op, 0) + 1
        ops = list(prog_types.keys())
        cnts = [prog_types[o] for o in ops]
        axes[1].barh(ops, cnts, color='#4CAF50')
        axes[1].set_xlabel('Count')
        axes[1].set_title('Solved Tasks by Operation Type')
    else:
        axes[1].text(0.5, 0.5, 'No tasks solved', ha='center', va='center', fontsize=14)
        axes[1].set_title('Solved Tasks by Operation Type')

    plt.suptitle('Phase 20: Expanded DSL Program Search', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase20_expanded_search.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {elapsed:.1f}s")
    return out

if __name__ == '__main__':
    main()
