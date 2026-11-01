# Changelog

All notable changes to this project are documented in this file.

The project follows semantic versioning for the public software releases.

## [1.1.1] - 2026-11-01

### Fixed

- Removed ten stale generated-data files inherited from the v1.0.x repository history.
- Ensured that the release archive follows the documented source-only packaging policy: simulation outputs are regenerated from the released scripts and fixed seeds.

This packaging correction does not modify the model, experimental design, analysis code or scientific conclusions introduced in v1.1.0.

## [1.1.0] - 2026-11-01

### Added

- Native-geometry robustness analyses for multiple numbers of attractors.
- Constant-spacing scaling analyses that adjust domain size and population with \(K\).
- Time-horizon and zero-explorer crowding controls for large-\(K\) regimes.
- Population-size and expected-explorer-count scaling analyses.
- A dedicated transition-network snapshot workflow.
- Run-level, transition-event and diagnostic CSV exports supporting the revised figures.
- A main-text visualization of the canonical multi-\(K\) robustness results.
- Explicit argument validation and optional data-collection controls in the reusable run interface.

### Changed

- Aligned the computational workflow with the revised manuscript and supplementary material.
- Expanded the main pipeline from the earlier figure set to seven manuscript figures.
- Strengthened the separation between the minimal random-walk baseline, interface-only dynamics, the full interface-plus-novelty model and the flat-landscape control.
- Revised transition-network, interface-occupation and boundary-distance diagnostics.
- Consolidated robustness calculations in `supplementary_robustness_scaling.py`, which is also called by the main pipeline.
- Renamed and reorganized supplementary scripts and generated-output directories.
- Updated documentation, citation metadata and reproducibility instructions.

### Removed

- Superseded scripts `figure_S2_robustness_K.py` and `figure_S3_transition_slope.py`; their revised functionality is provided by the new supplementary modules.
- Stale generated figures and processed outputs from the v1.0.x source tree.

## [1.0.0] - 2026-05-12

### Added

- Initial public implementation of the fixed-landscape agent-based model.
- Main experimental pipeline and the first robustness and transition-slope workflows.
- Reproducibility metadata, pinned dependencies and MIT license.

[1.1.1]: https://github.com/meccanismocomplesso/boundary-connectivity-abm/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/meccanismocomplesso/boundary-connectivity-abm/compare/v1.0.1...v1.1.0
[1.0.0]: https://github.com/meccanismocomplesso/boundary-connectivity-abm/releases/tag/v.1.0.0
