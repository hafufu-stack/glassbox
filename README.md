# Project GlassBox: Structure Over Scale in Neural Reasoning

> **Breaking the Black Box Through Architectural Transparency and Antifragile Adaptation**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19808285.svg)](https://doi.org/10.5281/zenodo.19808285)

## 📄 Paper

**[Project GlassBox: Structure Over Scale in Neural Reasoning — Breaking the Black Box Through Architectural Transparency and Antifragile Adaptation](https://doi.org/10.5281/zenodo.19808285)**

Funasaki, H. (2026). *Project GlassBox: Structure Over Scale in Neural Reasoning — Breaking the Black Box Through Architectural Transparency and Antifragile Adaptation.* Zenodo. https://doi.org/10.5281/zenodo.19808285

## 🔬 Overview

Project GlassBox is a systematic **59-phase experimental campaign** demonstrating that small, structurally constrained neural architectures can simultaneously achieve **superior task performance** and **unprecedented interpretability** compared to large unconstrained models.

Using ARC-AGI as a benchmark for abstract visual reasoning, a **77K-parameter Graph Neural Network** with Pointer attention (the "GlassBox Agent") outperforms a **1.45M-parameter Transformer** baseline (56.8% vs 43.9% full match accuracy).

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

### What's New in v2

- **Phases 34–50 (Antifragile Intelligence):** Systematic exploration of controlled destruction as a performance enhancement mechanism. Gradient-based ablation outperforms random ablation. 10 complex optimization strategies tested — all fail to beat simple ablation + adaptation ("Simplicity Wins").
- **Phases 51–55 (Neural Metabolism):** Fine-grained grid search over ablation × neurogenesis rates. Peak single-seed accuracy of 90.8% at Ablate 25% + Neurogenesis 5%.
- **Phases 58–59 (Statistical Reckoning):** Rigorous multi-seed validation reveals the 90.8% peak is not statistically significant (p=0.849). The true benefit of ablation is *variance reduction*: 15% ablation reduces seed-dependent std from 3.0% to 0.7%, making the system dramatically more reliable.

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

## 📁 Project Structure

```
glassbox/
├── experiments/          # All 59 phase scripts
│   ├── phase1_neural_program_synthesis.py
│   ├── phase2_architecture_assembly.py
│   ├── ...
│   └── phase59_ablation_manifesto.py
├── papers/               # LaTeX source and PDF
│   ├── paper_v1.tex
│   ├── paper_v2.tex
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
