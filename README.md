# Boundary crossings as a minimal mechanism for global connectivity in structured landscapes

Python implementation and reproducibility workflow for the agent-based model described in:

> **Boundary crossings as a minimal mechanism for global connectivity in structured landscapes**
> Fabio Nelli (Independent Researcher)

The model asks whether heterogeneous stochastic mobility in a fixed, structured multi-attractor landscape is sufficient to generate a globally connected transition network. It compares a minimal random-walk baseline with interface-sensitive and novelty-sensitive exploratory rules and with a flat-landscape control.

## Release

This repository documents software release **v1.1.0**, prepared for the revised manuscript and supplementary material. It supersedes the public v1.0.x implementation while preserving the fixed-landscape research design.

The archived software record is available through the Zenodo concept DOI:

<https://doi.org/10.5281/zenodo.20144948>

## Repository structure

```text
.
|-- agents.py
|-- config.py
|-- model.py
|-- run.py
|-- utils.py
|-- main_paper_submission.py
|-- supplementary_network_snapshot.py
|-- supplementary_population_scaling.py
|-- supplementary_robustness_scaling.py
|-- supplementary_transition_slope.py
|-- requirements.txt
|-- CITATION.cff
|-- CHANGELOG.md
|-- LICENSE
`-- README.md
```

### Core modules

- `agents.py`: embedded and exploratory agent dynamics.
- `config.py`: default model and simulation parameters.
- `model.py`: fixed-landscape model, transition tracking and data collection.
- `run.py`: reusable single-run entry point and optional CSV export.
- `utils.py`: geometry, state assignment and network metrics.

### Experimental pipelines

- `main_paper_submission.py`: main experimental conditions and Figures 1--7. It calls the canonical native-\(K\) robustness routine used for Figure 5.
- `supplementary_network_snapshot.py`: representative transition-network snapshot.
- `supplementary_population_scaling.py`: population-size and explorer-count scaling analyses.
- `supplementary_robustness_scaling.py`: native-geometry, constant-spacing, time-horizon and crowding controls across multiple values of \(K\).
- `supplementary_transition_slope.py`: transition-slope analysis derived from the main full-model aggregate output.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/meccanismocomplesso/boundary-connectivity-abm.git
cd boundary-connectivity-abm
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The release has been checked with Python 3.12 using the pinned environment. The original workflow was also developed under Python 3.11.

## Quick smoke test

Run the default single simulation:

```bash
python run.py
```

This writes optional run-level tables to `output/`. For a faster programmatic check:

```bash
python -c "from run import run_model; m = run_model(n_agents=40, n_attractors=3, steps=10, seed=42); print(m.transition_graph.number_of_edges())"
```

## Reproducing the analyses

Run the main pipeline first:

```bash
python main_paper_submission.py
```

It creates:

```text
paper_submission_outputs/
|-- data/
`-- figures/
```

Then run the supplementary pipelines as required:

```bash
python supplementary_network_snapshot.py
python supplementary_population_scaling.py
python supplementary_robustness_scaling.py
python supplementary_transition_slope.py
```

`supplementary_transition_slope.py` expects `paper_submission_outputs/data/full_model_aggregated.csv`, so the main pipeline must be completed before that script is executed.

The scripts create the following output directories:

- `supplementary_network_snapshots_outputs/`
- `supplementary_population_scaling_outputs/`
- `supplementary_K_robustness_outputs/`
- `supplementary_transition_slope_outputs/`

The complete pipelines include multiple parameter values and independent simulation replicates and can therefore require substantial runtime and memory. Output directories are intentionally excluded from version control because they are generated from the released source and fixed seeds.

## Experimental design and reproducibility

- The attractor set is fixed during each run.
- The main landscape uses \(K=3\) attractors.
- Robustness analyses examine larger \(K\), including native-geometry and constant-spacing designs.
- Explorer assignment is Bernoulli by default.
- Independent runs use explicit deterministic seeds.
- The transition network is directed; reported mean degree refers to total degree unless a script labels a different quantity explicitly.
- Figures and processed tables are generated directly by the corresponding scripts.

Default values are defined in `config.py`. Experiment-specific constants and seed schedules are declared near the top of each pipeline script.

## Version history

See [CHANGELOG.md](CHANGELOG.md). Release v1.1.0 expands the robustness and scaling analyses, strengthens diagnostic exports and aligns the software with the revised manuscript and supplementary material.

## License

The software is distributed under the MIT License. See [LICENSE](LICENSE).

## Citation

Use the repository's `CITATION.cff` metadata or cite the archived software record:

> Nelli, F. (2026). *Boundary crossings as a minimal mechanism for global connectivity in structured landscapes* (Version 1.1.0) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.20144948>

The manuscript and software are separate research objects. When using the scientific results as well as the implementation, cite both the relevant manuscript version and the software release.

## Contact

Fabio Nelli, Independent Researcher, Rome, Italy.
