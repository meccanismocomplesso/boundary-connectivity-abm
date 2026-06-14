# Boundary-Mediated Global Connectivity in Structured Agent-Based Systems

Python implementation of an agent-based model for studying the emergence of global connectivity through boundary-mediated transitions in structured attractor landscapes.

## Related Resources

**arXiv preprint**

From Boundary Crossings to Global Connectivity: A Minimal Mechanism in Structured Agent-Based Landscapes

arXiv:2606.07344  
https://arxiv.org/abs/2606.07344

**CoMSES Computational Model Library**

Peer-reviewed computational model (review in progress)

https://www.comses.net/codebases/20606999-844a-40ca-b423-1a5404453d83/

**Zenodo archive**

https://doi.org/10.5281/zenodo.20145382

This repository contains the complete simulation framework, experimental pipeline, data aggregation workflow, and figure-generation scripts associated with the paper:

**From Boundary Crossings to Global Connectivity: A Minimal Mechanism in Structured Agent-Based Landscapes**

Fabio Nelli

Preprint available at:

https://arxiv.org/abs/2606.07344

---

# Overview

The model investigates how heterogeneous mobility interacting with a structured multi-attractor landscape can generate large-scale connectivity without requiring optimization, learning, or goal-directed exploration.

Agents move in a continuous two-dimensional environment organized into attractor basins. A fraction of agents exhibits higher-mobility exploratory dynamics, while the remaining agents remain locally constrained within basins.

The study analyzes how localized boundary-crossing events accumulate into transition networks that progressively integrate the explored state space.

---

# Repository Structure

```text
.
├── agents.py
├── config.py
├── model.py
├── utils.py
│
├── run.py
├── main_paper_submission.py
├── figure_S2_robustness_K.py
├── figure_S3_transition_slope.py
│
├── paper_submission_outputs/
├── supplementary_robustness_K_outputs/
├── supplementary_slope_outputs/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/meccanismocomplesso/boundary-connectivity-abm.git
cd boundary-connectivity-abm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Quick Start

## Main experimental pipeline

Run the main experimental workflow:

```bash
python main_paper_submission.py
```

This script reproduces the main experiments and generates the figures reported in the paper.

---

## Supplementary analyses

### Robustness to the number of attractors (Supplementary Figure S2)

```bash
python figure_S2_robustness_K.py
```

### Transition-slope analysis (Supplementary Figure S3)

```bash
python figure_S3_transition_slope.py
```

Generated outputs will be stored in:

- `paper_submission_outputs/`
- `supplementary_robustness_K_outputs/`
- `supplementary_slope_outputs/`

---

# Outputs

Simulation outputs and generated figures are stored in:

- `paper_submission_outputs/`
- `supplementary_robustness_K_outputs/`
- `supplementary_slope_outputs/`

These directories contain processed data and figures used in the manuscript.

---

# Model Description

The model consists of:

- a population of agents evolving in continuous 2D space,
- a structured landscape composed of multiple attractor basins,
- heterogeneous mobility dynamics,
- transition-network construction from observed state changes.

The study compares:

- baseline random-walk exploration,
- interface-sensitive exploration,
- novelty-biased exploration,
- full candidate-selection dynamics,
- flat-landscape controls.

The resulting transition graphs are analyzed using network-science metrics including:

- graph density,
- mean out-degree,
- connected components,
- giant weakly connected component,
- giant strongly connected component.

---

# Reproducibility

All simulations are fully reproducible.

- Random seeds are explicitly controlled.
- Experimental configurations are defined in configuration files.
- All figures are generated directly from simulation outputs.
- The repository contains the complete workflow used for the paper.

Tested with:

- Python 3.11

The model is publicly archived through:

- GitHub (source repository)
- Zenodo (citable archive)
- CoMSES Computational Model Library (peer-review and reproducibility review)

The associated manuscript is available as an arXiv preprint.

---

# License

This project is distributed under the MIT License.

See the `LICENSE` file for details.

---

# Citation

If you use this repository in academic work, please cite the associated paper and archived code release.

### Paper

Nelli, F. (2026).

*From Boundary Crossings to Global Connectivity: A Minimal Mechanism in Structured Agent-Based Landscapes.*

arXiv:2606.07344

https://arxiv.org/abs/2606.07344

### Code archive

Zenodo DOI:

https://doi.org/10.5281/zenodo.20145382

### Repository

https://github.com/meccanismocomplesso/boundary-connectivity-abm


---

## Project Status

Current status:

- arXiv preprint published (arXiv:2606.07344)
- Code archived on Zenodo
- Computational model under peer review at CoMSES
- Journal submission in preparation

---

# Keywords

Agent-based modeling, complex systems, emergence, self-organization, network science, attractor landscapes, exploratory dynamics, stochastic processes, percolation, computational social science.
