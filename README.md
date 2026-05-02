# Project GlassBox: Structure Over Scale in Neural Reasoning

> **A 101-Phase Experimental Campaign on Architectural Transparency, Antifragile Adaptation, and the AGI Horizon**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19808285.svg)](https://doi.org/10.5281/zenodo.19808285)

## 📄 Paper

**[Project GlassBox: Structure Over Scale in Neural Reasoning — A 101-Phase Campaign on Architectural Transparency, Antifragile Adaptation, and the AGI Horizon](https://doi.org/10.5281/zenodo.19808285)**

Funasaki, H. (2026). *Project GlassBox: Structure Over Scale in Neural Reasoning.* Zenodo. https://doi.org/10.5281/zenodo.19808285

## 🔬 Overview

Project GlassBox is a systematic **101-phase experimental campaign** demonstrating that small, structurally constrained neural architectures can simultaneously achieve **superior task performance** and **unprecedented interpretability** compared to large unconstrained models.

Using ARC-AGI as a benchmark for abstract visual reasoning, a **77K-parameter Graph Neural Network** with Pointer attention (the "GlassBox Agent") outperforms a **1.45M-parameter Transformer** baseline (56.8% vs 43.9% full match accuracy). Through test-time gradient adaptation, data augmentation, and MCTS-based strategy search, accuracy reaches **91.95%** — the campaign's peak — with just 8 rollouts. A learned MuZero-style latent simulator achieves **117x speedup**, and latent self-prediction probes reveal the AI can predict its own success with **99.2% accuracy**.

### Key Results

| Finding | Result |
|---------|--------|
| **Structure > Scale** | 77K params outperforms 1.45M params (19x smaller, higher accuracy) |
| **Meta-MCTS Peak** | 91.95% via strategy-space search with 8 rollouts (P87) |
| **MuZero Latent Dynamics** | 117x speedup: 55s vs 6449s at 88.5% accuracy (P94) |
| **AI Self-Knowledge** | 99.2% success prediction from internal states (P97) |
| **Latent Verifier** | Self-prediction as inference selector: 89.7% (P100) |
| **Hydra Self-Repair** | 50% neuron destruction -> 95.8% recovery via adaptation |
| **82.8% Attribution** | 3.3x beyond Anthropic's 25% attribution wall |
| **Antifragile Intelligence** | Gradient-based ablation achieves super-compensation |
| **Ultimate Configuration** | L2 20% + LR 0.1 + Soup K=5 = **88.9%** (multi-seed) |
| **Latent Graph Dynamics** | Multi-step latent reasoning: **90.8%** (P79) |

### What's New in v4

- **Chapter XIV — Test-Time Compute Frontier (P82-90):** Dynamic pondering, 10-step TTT sufficiency via Reptile meta-initialization, Meta-MCTS discovery achieving **91.95%** campaign peak, and PRM-guided MCTS scaling laws.
- **Chapter XV — The AlphaZero Paradigm (P91-97):** Expert iteration, macro-action discovery, MuZero-style latent dynamics (**117x speedup**), and **99.2%** latent self-prediction probes.
- **Chapter XVI — The Latent Liberation (P98-101):** Continuous action embeddings, Latent Verifier (**89.7%**), and unified Verifier-Guided MCTS at half computation cost.
- **10 new experiment figures** and updated summary visualizations (journey, waterfall, breakthrough map).

<details>
<summary><strong>What was new in v3</strong></summary>

- **Phases 60-64 (Optimal Portfolio):** Oracle strategy selection achieves 90.3% across 3 adaptation intensities.
- **Phase 67 (Mechanistic Anatomy):** L1=perception (90%), L2=cognition (78%).
- **Phase 68 (Zero-Shot Rule Synthesis):** TTT creates novel rules unseen in training (50%).
- **Phase 75 (Ultimate Configuration):** L2 20% + LR 0.1 + Soup K=5 = 88.9% (multi-seed validated).
- **Phase 79 (Latent Graph Dynamics):** Multi-step latent reasoning achieves 90.8% peak.
- **Phases 80-81 (Practical TTT Engineering):** Early stopping and weight decay optimization.

</details>

<details>
<summary><strong>What was new in v2</strong></summary>

- **Phases 34-50 (Antifragile Intelligence):** Controlled destruction as performance enhancement.
- **Phases 51-55 (Neural Metabolism):** Fine-grained ablation x neurogenesis grid search.
- **Phases 58-59 (Statistical Reckoning):** Ablation's true benefit is variance reduction (4.3x).

</details>

## 🧪 Experimental Phases

| Phase | Name | Key Discovery |
|-------|------|---------------|
| 1-15 | Structure Over Scale | 77K GNN+Pointer beats 1.45M Transformer |
| 16-20 | The Bitter Lesson | Symbolic DSL covers only 5.4% of tasks |
| 21-26 | Adaptation Supremacy | Few-shot gradient adaptation is strictly superior |
| 27-30 | Breaking the 85% Ceiling | Augmented adaptation reaches 87.4% |
| 31 | Hydra Effect | First quantitative neural self-repair measurement |
| 32 | Superposition Analysis | SAE decomposition of internal representations |
| 33 | Attribution Breakthrough | 82.8% full causal path tracing |
| 34 | Superposition Computation | 12 superposition clusters in 77K params |
| 35-37 | Gradient-Based Ablation | Targeted ablation outperforms random (87.4% vs 83.1%) |
| 38-47 | The Optimization Graveyard | 10 complex strategies all fail vs simple ablation |
| 48 | Neurogenesis | Neural metabolism cycle achieves 89.7% |
| 50 | Sweet Spot | 2D ablation x neurogenesis landscape mapped |
| 51-55 | Neural Metabolic Optimum | Peak 90.8% at A=25% + N=5% (single seed) |
| 58 | Statistical Significance | 10-seed: A=25%+N=10% not significant (p=0.849) |
| 59 | Ablation-Only Manifesto | Ablation = variance regularizer (4.3x std reduction) |
| 60-64 | Optimal Portfolio | Oracle strategy selection achieves 90.3% |
| 67 | Mechanistic Anatomy | L1=perception (90%), L2=cognition (78%) |
| 68 | Zero-Shot Rule Synthesis | TTT creates novel rules unseen in training (50%) |
| 69 | Model Soups | Weight averaging: +2pp with zero inference cost |
| 70-71 | Optimal L2 + LR | L2=20% (std=1.1%), LR=0.1 (88.1%) |
| 72 | Differentiable Object Discovery | Slot Attention fails (2.3% vs BFS 62.1%) |
| 74, 78 | TTT-Guided Self-Play | Experience replay eliminates catastrophic forgetting |
| 75 | Ultimate Configuration | L2 20% + LR 0.1 + Soup K=5 = **88.9%** |
| 76 | Multi-Strategy Ensemble | Per-strategy soup matches full 9-way vote |
| 77 | Amortized TTT | Hypernetwork: 582x speedup but 62.1% accuracy |
| 79 | Latent Graph Dynamics | Multi-step latent reasoning: **90.8%** peak |
| 80-81 | Practical TTT Engineering | Early stopping (2.2x speedup), weight decay |
| **82** | **Dynamic Pondering** | **Adaptive computation depth: +2.1pp** |
| **83** | **Amortized Meta-Init** | **TTT-10 = TTT-100 via Reptile (9.6x speedup)** |
| **87** | **Meta-MCTS** | **Strategy-space search: 91.95% campaign peak** |
| **89-90** | **PRM + Scaling Law** | **PRM-guided MCTS: 90.8% at 64 rollouts** |
| **91** | **Expert Iteration** | **Self-improvement via best-strategy distillation** |
| **93** | **Macro-Actions** | **Two-step strategy compositions** |
| **94** | **MuZero Latent Dynamics** | **117x speedup via learned latent simulator** |
| **97** | **Latent Self-Prediction** | **99.2% success prediction from internal states** |
| **98** | **Continuous Actions** | **Continuous embeddings replace discrete actions** |
| **100** | **Latent Verifier** | **Self-prediction as inference selector: 89.7%** |
| **101** | **Unified V-MCTS** | **Verifier-guided MCTS at half computation** |

## 📁 Project Structure

```
glassbox/
├── experiments/          # Phase 1-100 experiment scripts
├── experiments2/         # Phase 101+ experiment scripts
├── papers/               # LaTeX source and PDF (shared via Zenodo)
│   ├── paper_v1.tex
│   ├── paper_v2.tex
│   ├── paper_v3.tex
│   ├── paper_v4.tex
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

- **[SNN-Genesis](https://github.com/hafufu-stack/SNN-Genesis)** — Stochastic resonance in neural reasoning (v1-v20, 100+ phases)
- **[SNN-Synthesis](https://github.com/hafufu-stack/SNN-Synthesis)** — Neural Cellular Automata for ARC-AGI (v1-v15, 241 phases)

## 🤖 AI Collaboration

This research was conducted as a collaborative effort between the human author and AI research assistants. All experimental decisions, research direction, and final interpretation were made by the human author.

## 📜 License

MIT License

## 👤 Author

**Hiroto Funasaki** — Independent Researcher, Japan
- ORCID: [0009-0004-2517-0177](https://orcid.org/0009-0004-2517-0177)
- GitHub: [@hafufu-stack](https://github.com/hafufu-stack)
- Sponsor: [github.com/sponsors/hafufu-stack](https://github.com/sponsors/hafufu-stack)
