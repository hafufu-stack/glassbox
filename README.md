# Project GlassBox: Structure Over Scale in Neural Reasoning

> **Breaking the Black Box Through Architectural Transparency**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19808285.svg)](https://doi.org/10.5281/zenodo.19808285)

## 📄 Paper

**[Project GlassBox: Structure Over Scale in Neural Reasoning — Breaking the Black Box Through Architectural Transparency](https://doi.org/10.5281/zenodo.19808285)**

Funasaki, H. (2026). *Project GlassBox: Structure Over Scale in Neural Reasoning — Breaking the Black Box Through Architectural Transparency.* Zenodo. https://doi.org/10.5281/zenodo.19808285

## 🔬 Overview

Project GlassBox is a systematic **33-phase experimental campaign** demonstrating that small, structurally constrained neural architectures can simultaneously achieve **superior task performance** and **unprecedented interpretability** compared to large unconstrained models.

Using ARC-AGI as a benchmark for abstract visual reasoning, a **77K-parameter Graph Neural Network** with Pointer attention (the "GlassBox Agent") outperforms a **1.45M-parameter Transformer** baseline (56.8% vs 43.9% full match accuracy).

### Key Results

| Finding | Result |
|---------|--------|
| **Structure > Scale** | 77K params outperforms 1.45M params (19× smaller, higher accuracy) |
| **Adaptation Supremacy** | Test-time gradient adaptation: +34.5pp improvement |
| **85% Ceiling Breakthrough** | D8 geometric augmentation pushes accuracy to 87.4% |
| **Hydra Self-Repair** | 50% neuron destruction → 95.8% recovery via adaptation |
| **82.8% Attribution** | 3.3× beyond Anthropic's 25% attribution wall |

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

## 📁 Project Structure

```
glassbox/
├── experiments/          # All 33 phase scripts
│   ├── phase1_neural_program_synthesis.py
│   ├── phase2_architecture_assembly.py
│   ├── ...
│   └── phase33_attribution.py
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
