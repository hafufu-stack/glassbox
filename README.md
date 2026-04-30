# Project GlassBox: Structure Over Scale in Neural Reasoning

> **An 81-Phase Experimental Campaign on Architectural Transparency, Antifragile Adaptation, and the AGI Horizon**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19808285.svg)](https://doi.org/10.5281/zenodo.19808285)

## 📄 Paper

**[Project GlassBox: Structure Over Scale in Neural Reasoning — An 81-Phase Campaign on Architectural Transparency, Antifragile Adaptation, and the AGI Horizon](https://doi.org/10.5281/zenodo.19808285)**

Funasaki, H. (2026). *Project GlassBox: Structure Over Scale in Neural Reasoning.* Zenodo. https://doi.org/10.5281/zenodo.19808285

## 🔬 Overview

Project GlassBox is a systematic **81-phase experimental campaign** demonstrating that small, structurally constrained neural architectures can simultaneously achieve **superior task performance** and **unprecedented interpretability** compared to large unconstrained models.

Using ARC-AGI as a benchmark for abstract visual reasoning, a **77K-parameter Graph Neural Network** with Pointer attention (the "GlassBox Agent") outperforms a **1.45M-parameter Transformer** baseline (56.8% vs 43.9% full match accuracy). Through test-time gradient adaptation, data augmentation, and the Ultimate Configuration (L2 ablation + high LR + Model Soup), accuracy reaches **88.9%** (multi-seed validated). Latent graph dynamics with multi-step reasoning in hidden space achieves the campaign's peak of **90.8%**.

### Key Results

| Finding | Result |
|---------|--------|
| **Structure > Scale** | 77K params outperforms 1.45M params (19× smaller, higher accuracy) |
| **Adaptation Supremacy** | Test-time gradient adaptation: +34.5pp improvement |
| **85% Ceiling Breakthrough** | D8 geometric augmentation pushes accuracy to 87.4% |
| **Hydra Self-Repair** | 50% neuron destruction → 95.8% recovery via adaptation |
| **82.8% Attribution** | 3.3× beyond Anthropic's 25% attribution wall |
| **Antifragile Intelligence** | Gradient-based ablation achieves super-compensation beyond baseline |
| **Variance Regularization** | 15% ablation reduces performance std by 4.3× (3.0% → 0.7%) |
| **Ultimate Configuration** | L2 20% + LR 0.1 + Soup K=5 = **88.9%** (multi-seed validated) |
| **Latent Graph Dynamics** | Multi-step reasoning in latent space: **90.8%** peak accuracy |
| **Zero-Shot Rule Synthesis** | TTT creates novel rules from 2–3 examples (50% on unseen ops) |

### What's New in v3

- **Phases 60–64 (Optimal Portfolio):** Oracle strategy selection achieves 90.3% across 3 adaptation intensities, proving optimal configuration is task-dependent.
- **Phase 67 (Mechanistic Anatomy):** Linear probes reveal GNN L1 encodes low-level features (color: 90%) while L2 specializes in high-level rules (operation: 78%), explaining the mechanistic basis for L2 ablation's super-recovery.
- **Phase 68 (Zero-Shot Rule Synthesis):** TTT recovers 50% accuracy on completely novel operations absent from training — proving on-the-fly rule creation, not memorization.
- **Phase 72 (Prior Knowledge Dominance):** Handcrafted BFS outperforms learned Slot Attention by 27× (62.1% vs 2.3%), proving human prior knowledge is a decisive advantage in low-data regimes.
- **Phase 75 (Ultimate Configuration):** L2 Ablate 20% + LR 0.1 + Model Soup K=5 = 88.9% mean with 2.0% std, the campaign's most reliable multi-seed configuration.
- **Phase 79 (Latent Graph Dynamics):** Multi-step reasoning in latent space achieves 90.8% — the campaign's peak accuracy — bypassing the DSL expressiveness bottleneck.
- **Phase 78 (Continual Self-Play):** Experience replay eliminates catastrophic forgetting, enabling stable self-improvement (+1.1% per iteration).
- **Phases 80–81 (Practical TTT Engineering):** Early stopping (2.2× speedup) and weight decay analysis for deployment optimization.
- **5 new summary figures** including the 81-phase journey timeline and innovation waterfall chart.

<details>
<summary><strong>What was new in v2</strong></summary>

- **Phases 34–50 (Antifragile Intelligence):** Systematic exploration of controlled destruction as a performance enhancement mechanism. Gradient-based ablation outperforms random ablation. 10 complex optimization strategies tested — all fail to beat simple ablation + adaptation ("Simplicity Wins").
- **Phases 51–55 (Neural Metabolism):** Fine-grained grid search over ablation × neurogenesis rates. Peak single-seed accuracy of 90.8% at Ablate 25% + Neurogenesis 5%.
- **Phases 58–59 (Statistical Reckoning):** Rigorous multi-seed validation reveals the 90.8% peak is not statistically significant (p=0.849). The true benefit of ablation is *variance reduction*: 15% ablation reduces seed-dependent std from 3.0% to 0.7%.

</details>

## 🧪 Experimental Phases

| Phase | Name | Key Discovery |
|-------|------|---------------|
| 1–15 | Structure Over Scale | 77K GNN+Pointer beats 1.45M Transformer |
| 16–20 | The Bitter Lesson | Symbolic DSL covers only 5.4% of tasks |
| 21–26 | Adaptation Supremacy | Few-shot gradient adaptation is strictly superior |
| 27–30 | Breaking the 85% Ceiling | Augmented adaptation reaches 87.4% |
| 31 | Hydra Effect | First quantitative neural self-repair measurement |
| 32 | Superposition Analysis | SAE decomposition of internal representations |
| 33 | Attribution Breakthrough | 82.8% full causal path tracing |
| 34 | Superposition Computation | 12 superposition clusters identified in 77K params |
| 35–37 | Gradient-Based Ablation | Targeted ablation outperforms random (87.4% vs 83.1%) |
| 38–47 | The Optimization Graveyard | 10 complex strategies all fail vs simple ablation |
| 48 | Neurogenesis | Neural metabolism cycle achieves 89.7% |
| 50 | Sweet Spot | 2D ablation × neurogenesis landscape mapped |
| 51–55 | Neural Metabolic Optimum | Peak 90.8% at A=25% + N=5% (single seed) |
| 58 | Statistical Significance | 10-seed test: A=25%+N=10% not significant (p=0.849) |
| 59 | Ablation-Only Manifesto | Ablation = variance regularizer (4.3× std reduction) |
| 60–64 | Optimal Portfolio | Oracle strategy selection achieves 90.3% |
| 67 | Mechanistic Anatomy | L1=perception (90%), L2=cognition (78%) |
| 68 | Zero-Shot Rule Synthesis | TTT creates novel rules unseen in training (50%) |
| 69 | Model Soups | Weight averaging: +2pp with zero inference cost |
| 70–71 | Optimal L2 + LR | L2=20% (std=1.1%), LR=0.1 (88.1%) |
| 72 | Differentiable Object Discovery | Slot Attention fails (2.3% vs BFS 62.1%) |
| 74, 78 | TTT-Guided Self-Play | Experience replay eliminates catastrophic forgetting |
| 75 | Ultimate Configuration | L2 20% + LR 0.1 + Soup K=5 = **88.9%** |
| 76 | Multi-Strategy Ensemble | Per-strategy soup matches full 9-way vote (87.9%) |
| 77 | Amortized TTT | Hypernetwork: 582× speedup but 62.1% accuracy |
| 79 | Latent Graph Dynamics | Multi-step latent reasoning: **90.8%** peak |
| 80 | Early Stopping | 2.2× speedup at −0.8pp cost |
| 81 | Weight Decay | wd=10⁻³ reduces variance 44% with negligible accuracy cost |

## 📁 Project Structure

```
glassbox/
├── experiments/          # All 81 phase scripts
│   ├── phase1_neural_program_synthesis.py
│   ├── phase2_architecture_assembly.py
│   ├── ...
│   └── phase81_weightdecay.py
├── papers/               # LaTeX source and PDF (shared via Zenodo)
│   ├── paper_v1.tex
│   ├── paper_v2.tex
│   ├── paper_v3.tex
│   └── figures/
├── results/              # Experiment results (JSON)
├── figures/              # All experiment figures (PNG)
├── data/                 # ARC-AGI training tasks
├── LICENSE
└── README.md
```

## 🚀 Quick Start

```bash
git clone https://github.com/hafufu-stack/glassbox.git
cd glassbox
pip install torch matplotlib numpy scikit-learn
```

## 🔗 Related Work

- **[SNN-Genesis](https://github.com/hafufu-stack/SNN-Genesis)** — Stochastic resonance in neural reasoning (v1–v20, 100+ phases)
- **[SNN-Synthesis](https://github.com/hafufu-stack/SNN-Synthesis)** — Neural Cellular Automata for ARC-AGI (v1–v15, 241 phases)

## 🤖 AI Collaboration

This research was conducted as a collaborative effort between the human author and AI research assistants. All experimental decisions, research direction, and final interpretation were made by the human author.

## 📜 License

MIT License

## 👤 Author

**Hiroto Funasaki** — Independent Researcher, Japan
- ORCID: [0009-0004-2517-0177](https://orcid.org/0009-0004-2517-0177)
- GitHub: [@hafufu-stack](https://github.com/hafufu-stack)
- Sponsor: [github.com/sponsors/hafufu-stack](https://github.com/sponsors/hafufu-stack)
