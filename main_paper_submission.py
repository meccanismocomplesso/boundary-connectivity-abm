"""
Main-paper executable for the fixed-landscape minimal-mechanism study.

This script generates the primary experimental dataset and the figures used
in the main manuscript:

01. Figure 1  - Baseline vs full-model spatial comparison
02. Figure 2  - Boundary-localized transition events
03. Figure 3  - Transition-network expansion
04. Figure 4  - Emergence of global connectivity
05. Figure 5  - Robustness across landscape size
06. Figure 6  - Mechanism decomposition
07. Figure 7  - Flat-landscape control

The native-K robustness calculation remains maintained in the dedicated
supplementary_robustness_scaling module.  This script calls that canonical
routine and produces a complementary main-paper visualization from the same
aggregated results.  All other supplementary analyses remain separate from
the main-paper pipeline.

Scientific design
-----------------
Main landscape:
    K = 3 fixed attractor regions.

Mechanism hierarchy:
    baseline
        pure exploratory random walk

    interface_only
        candidate selection with interface score only; no drift

    full_model
        candidate selection with interface + novelty; no drift

    flat_landscape
        no attractor-derived influence on movement; only the two stochastic
        displacement scales remain.  Fixed attractor labels are retained
        operationally for identical state/transition measurement.

The attractor set is fixed throughout every run.
"""
from __future__ import annotations
from pathlib import Path
import gc
from typing import Dict, Iterable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import config
from model import ExploratorySystemModel
from run import run_model
import supplementary_robustness_scaling as robustness
OUTPUT_DIR = Path('paper_submission_outputs')
DATA_DIR = OUTPUT_DIR / 'data'
FIG_DIR = OUTPUT_DIR / 'figures'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
P_VALUES = np.round(np.arange(0.0, 0.051, 0.005), 3)
N_RUNS = 20
N_STEPS = 300
CI_Z = 1.96
BASE_SEED = 1000
N_AGENTS = int(getattr(config, 'N_AGENTS', 200))
N_ATTRACTORS = 3
GRID_WIDTH = float(getattr(config, 'GRID_WIDTH', 50))
GRID_HEIGHT = float(getattr(config, 'GRID_HEIGHT', 50))
TRANSITION_EVENT_P_VALUES = [0.01, 0.02, 0.03]
TRANSITION_EVENT_BASE_SEED = 9000
TRANSITION_EVENT_CONDITIONS = ['baseline', 'interface_only', 'full_model']
TRANSITION_PROBABILITY_MAX_NORMALIZED_DISTANCE = 2.0
PAIRED_BOOTSTRAP_SAMPLES = 10000
PAIRED_BOOTSTRAP_SEED = 47001
FIGURE_FILENAMES = [
    '01_Figure_1_baseline_vs_full.png',
    '02_Figure_2_boundary_localized_transition_events.png',
    '03_Figure_3_transition_network_expansion.png',
    '04_Figure_4_global_connectivity.png',
    '05_Figure_5_K_robustness.png',
    '06_Figure_6_mechanism_decomposition.png',
    '07_Figure_7_flat_landscape_control.png',
]
CONDITIONS = {
    'baseline': {
        'label': 'Baseline',
        'NOVELTY_BONUS': 0.0,
        'INTERFACE_BONUS_MODE': 'none',
        'EXPLORATORY_RULE': 'random_walk',
        'LANDSCAPE_MODE': 'structured'
    },
    'interface_only': {
        'label': 'Interface-only',
        'NOVELTY_BONUS': 0.0,
        'INTERFACE_BONUS_MODE': 'real',
        'EXPLORATORY_RULE': 'interface_only',
        'LANDSCAPE_MODE': 'structured'
    },
    'full_model': {
        'label': 'Full model',
        'NOVELTY_BONUS': 1.0,
        'INTERFACE_BONUS_MODE': 'real',
        'EXPLORATORY_RULE': 'interface_novelty',
        'LANDSCAPE_MODE': 'structured'
    },
    'flat_landscape': {
        'label': 'Flat landscape control',
        'NOVELTY_BONUS': 1.0,
        'INTERFACE_BONUS_MODE': 'real',
        'EXPLORATORY_RULE': 'interface_novelty',
        'LANDSCAPE_MODE': 'flat_control'
    }
}
MAIN_CONDITIONS = ['baseline', 'interface_only', 'full_model', 'flat_landscape']
STRUCTURED_CONDITIONS = ['baseline', 'interface_only', 'full_model']
CONFIG_KEYS = ['NOVELTY_BONUS', 'INTERFACE_BONUS_MODE', 'EXPLORATORY_RULE', 'LANDSCAPE_MODE']
FIGURE_NOTES = {
    '01_Figure_1_baseline_vs_full.png': 'Figure 1. Baseline versus full-model spatial comparison. Interface occupation and distance to the nearest inter-basin boundary for embedded and exploratory agents as functions of the exploratory fraction p. Shaded regions indicate 95% confidence intervals across independent runs.',
    '02_Figure_2_boundary_localized_transition_events.png': 'Figure 2. Boundary-localized transition events across exploratory mechanisms. For exploratory agents, the event-distance distribution and conditional transition probability are shown for the baseline random walk, interface-only and full interface-plus-novelty conditions. Boundary distances are normalized by the exploratory movement scale; the probability panel focuses on the transition-relevant interval from zero to two exploratory movement scales. Shaded regions indicate approximate 95% confidence intervals across contributing run-level estimates.',
    '03_Figure_3_transition_network_expansion.png': 'Figure 3. Event-rate and topological-efficiency decomposition. Exploratory transition rate, unique directed-edge production per exploratory movement, and unique-edge yield per transition are shown as functions of the exploratory fraction p. Metrics are calculated within each independent run before aggregation; runs with a zero denominator do not contribute to the corresponding ratio. Shaded regions indicate approximate 95% confidence intervals across contributing runs.',
    '04_Figure_4_global_connectivity.png': 'Figure 4. Emergence of global connectivity. Relative sizes of the largest weakly and strongly connected components as functions of p across structured experimental conditions.',
    '05_Figure_5_K_robustness.png': 'Figure 5. Robustness across landscape size. Relative size of the largest weakly connected component and absolute number of observed directed transition edges as functions of the exploratory fraction p for K = 3, 5, 10 and 20 under native fixed-domain geometry. Shaded regions indicate approximate 95% confidence intervals across independent runs.',
    '06_Figure_6_mechanism_decomposition.png': 'Figure 6. Mechanism decomposition of spatial organization. Exploratory boundary distance, exploratory interface occupation, boundary-distance separation and interface-occupation asymmetry are shown across baseline, interface-only, and full interface-plus-novelty exploratory mechanisms.',
    '07_Figure_7_flat_landscape_control.png': 'Figure 7. Flat-landscape control. Comparison between the full structured model and the flat control, in which attractor labels are retained only for measurement and exert no influence on movement.',
}

def clean_figure_directory() -> None:
    """Remove only image outputs from previous main-paper runs."""
    for path in FIG_DIR.iterdir():
        if path.suffix.lower() in {'.png', '.pdf', '.svg', '.jpg', '.jpeg'}:
            path.unlink()

def write_figure_notes() -> None:
    notes_path = OUTPUT_DIR / 'FIGURE_NOTES.md'
    with notes_path.open('w', encoding='utf-8') as handle:
        handle.write('# Main-paper figure order and captions\n\n')
        for filename in FIGURE_FILENAMES:
            handle.write(f'## {filename}\n\n{FIGURE_NOTES[filename]}\n\n')
    print(f'Saved notes: {notes_path}')

def snapshot_config() -> Dict[str, object]:
    """Snapshot only mutable condition-level configuration."""
    return {key: getattr(config, key) for key in CONFIG_KEYS}

def restore_config(snapshot: Dict[str, object]) -> None:
    for key, value in snapshot.items():
        setattr(config, key, value)

def apply_condition(condition_name: str) -> None:
    if condition_name not in CONDITIONS:
        raise KeyError(f'Unknown experimental condition: {condition_name!r}')
    spec = CONDITIONS[condition_name]
    for key in CONFIG_KEYS:
        setattr(config, key, spec[key])

def apply_full_model() -> None:
    apply_condition('full_model')

def validate_condition_configuration(condition_name: str) -> None:
    """
    Fail loudly if the runtime configuration does not match the
    declared experimental condition.
    """
    spec = CONDITIONS[condition_name]
    mismatches = []
    for key in CONFIG_KEYS:
        actual = getattr(config, key)
        expected = spec[key]
        if actual != expected:
            mismatches.append((key, actual, expected))
    if mismatches:
        raise RuntimeError(f'Condition configuration mismatch: {mismatches}')

def compute_ci(std_values: np.ndarray, n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError('n_samples must be > 0.')
    return CI_Z * std_values / np.sqrt(n_samples)

def save_current_figure(filename: str) -> None:
    path = FIG_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved figure: {path}')

def plot_with_condition_lines(
    ax,
    all_results: Dict[str, pd.DataFrame],
    y_col: str,
    title: str,
    ylabel: str,
    include_conditions: Iterable[str],
    show_ci: bool = True,
    bounds=None,
) -> None:
    for condition_name in include_conditions:
        df = all_results[condition_name]
        x = df['exploratory_fraction'].to_numpy()
        y = df[y_col].to_numpy()
        ax.plot(x, y, marker='o', label=CONDITIONS[condition_name]['label'])
        std_col = f'{y_col}_std'
        if show_ci and std_col in df.columns:
            std = df[std_col].fillna(0.0).to_numpy()
            n_col = f'{y_col}_n'
            n_eff = df[n_col].to_numpy() if n_col in df.columns else np.full(len(df), N_RUNS)
            ci = np.full(len(df), np.nan, dtype=float)
            valid = n_eff > 0
            ci[valid] = CI_Z * std[valid] / np.sqrt(n_eff[valid])
            lower = y - ci
            upper = y + ci
            if bounds is not None:
                lower = np.clip(lower, bounds[0], bounds[1])
                upper = np.clip(upper, bounds[0], bounds[1])
            ax.fill_between(x, lower, upper, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel('Exploratory fraction')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

def extract_agent_metrics(model: ExploratorySystemModel) -> Dict[str, float]:
    embedded_boundary = []
    exploratory_boundary = []
    embedded_interface_ratios = []
    exploratory_interface_ratios = []
    for agent in model.agents_list:
        if agent.boundary_distance is not None:
            if agent.agent_type == 'embedded':
                embedded_boundary.append(agent.boundary_distance)
            else:
                exploratory_boundary.append(agent.boundary_distance)
        ratio = agent.interface_visits / max(1, agent.total_moves)
        if agent.agent_type == 'embedded':
            embedded_interface_ratios.append(ratio)
        else:
            exploratory_interface_ratios.append(ratio)
    return {
        'boundary_embedded': float(np.mean(embedded_boundary)) if embedded_boundary else np.nan,
        'boundary_exploratory': float(np.mean(exploratory_boundary)) if exploratory_boundary else np.nan,
        'interface_embedded': float(np.mean(embedded_interface_ratios)) if embedded_interface_ratios else np.nan,
        'interface_exploratory': float(np.mean(exploratory_interface_ratios)) if exploratory_interface_ratios else np.nan,
        'n_embedded_realized': int(len(embedded_interface_ratios)),
        'n_exploratory_realized': int(len(exploratory_interface_ratios)),
    }

def extract_graph_metrics(model: ExploratorySystemModel) -> Dict[str, float]:
    G = model.transition_graph
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = float(nx.density(G)) if n_nodes > 1 else 0.0
    mean_degree = float(sum(dict(G.degree()).values()) / n_nodes) if n_nodes else 0.0
    if n_nodes:
        weak_components = list(nx.weakly_connected_components(G))
        strong_components = list(nx.strongly_connected_components(G))
        giant_weak = float(max((len(component) for component in weak_components)) / n_nodes) if weak_components else 0.0
        giant_strong = float(max((len(component) for component in strong_components)) / n_nodes) if strong_components else 0.0
    else:
        giant_weak = 0.0
        giant_strong = 0.0
    return {'graph_nodes': float(n_nodes), 'graph_edges': float(n_edges), 'graph_density': density, 'graph_mean_degree': mean_degree, 'giant_weak_component': giant_weak, 'giant_strong_component': giant_strong}


def extract_transition_efficiency_metrics(model: ExploratorySystemModel) -> Dict[str, float]:
    """Decompose network production into event rate and edge yield.

    All ratios are computed within a run.  Undefined ratios retain NaN so
    zero-explorer and zero-transition runs are not silently treated as
    observed zero efficiencies.
    """
    if not bool(getattr(model, 'collect_transition_events', False)):
        raise RuntimeError(
            'Transition-efficiency metrics require collect_transition_events=True.'
        )

    exploratory_agents = [
        agent for agent in model.agents_list
        if agent.agent_type == 'exploratory'
    ]
    exploratory_moves = int(sum(agent.total_moves for agent in exploratory_agents))

    events = model.export_transition_event_dataframe()
    if events.empty:
        exploratory_events = events
        embedded_transitions = 0
    else:
        exploratory_events = events[events['agent_type'] == 'exploratory']
        embedded_transitions = int((events['agent_type'] == 'embedded').sum())

    exploratory_transitions = int(len(exploratory_events))
    if exploratory_transitions:
        exploratory_unique_edges = int(
            exploratory_events[['prev_attractor', 'new_attractor']]
            .drop_duplicates()
            .shape[0]
        )
    else:
        exploratory_unique_edges = 0

    transition_rate = (
        exploratory_transitions / exploratory_moves
        if exploratory_moves > 0 else np.nan
    )
    edge_production = (
        exploratory_unique_edges / exploratory_moves
        if exploratory_moves > 0 else np.nan
    )
    unique_edge_yield = (
        exploratory_unique_edges / exploratory_transitions
        if exploratory_transitions > 0 else np.nan
    )
    total_transitions = exploratory_transitions + embedded_transitions
    if total_transitions != int(model.total_transitions):
        raise RuntimeError(
            'Transition-event audit failed: the event log and model counter disagree.'
        )
    exploratory_transition_share = (
        exploratory_transitions / total_transitions
        if total_transitions > 0 else np.nan
    )

    return {
        'exploratory_moves': float(exploratory_moves),
        'exploratory_transitions': float(exploratory_transitions),
        'embedded_transitions': float(embedded_transitions),
        'exploratory_unique_edges': float(exploratory_unique_edges),
        'exploratory_transitions_per_1000_moves': (
            1000.0 * transition_rate if np.isfinite(transition_rate) else np.nan
        ),
        'exploratory_unique_edges_per_1000_moves': (
            1000.0 * edge_production if np.isfinite(edge_production) else np.nan
        ),
        'exploratory_unique_edge_yield': float(unique_edge_yield),
        'exploratory_transition_share': float(exploratory_transition_share),
    }

def extract_model_metrics(model: ExploratorySystemModel) -> Dict[str, float]:
    metrics = {}
    metrics.update(extract_agent_metrics(model))
    metrics.update(extract_graph_metrics(model))
    metrics.update(extract_transition_efficiency_metrics(model))
    metrics['num_attractors'] = float(len(model.attractors))
    metrics['exploration_entropy'] = float(model.exploration_entropy())
    return metrics

def main_seed(exploratory_fraction: float, replicate: int) -> int:
    """
    Paired-seed design.

    For a given (p, replicate), every experimental condition starts
    from the same pseudo-random seed.  Initialization therefore uses
    the same Bernoulli type assignments and initial positions across
    conditions.  Movement trajectories may subsequently diverge
    because the mechanisms consume random numbers differently.
    """
    return BASE_SEED + int(exploratory_fraction * 10000) + replicate

def run_single_simulation(exploratory_fraction: float, seed: int, n_attractors: int=N_ATTRACTORS, n_agents: int=N_AGENTS, steps: int=N_STEPS, width: float=GRID_WIDTH, height: float=GRID_HEIGHT, collect_transition_events: bool=True, collect_step_observations: bool=False):
    model = run_model(exploratory_fraction=float(exploratory_fraction), n_agents=int(n_agents), n_attractors=int(n_attractors), steps=int(steps), seed=int(seed), width=float(width), height=float(height), output_dir=None, collect_transition_events=collect_transition_events, collect_step_observations=collect_step_observations, collect_agent_history=False, collect_datacollector=False)
    if len(model.attractors) != n_attractors:
        raise RuntimeError('Main-paper run changed the fixed attractor count.')
    return (model, extract_model_metrics(model))

def aggregate_condition(condition_name: str) -> pd.DataFrame:
    print(f'\n=== Running condition: {condition_name} ===')
    apply_condition(condition_name)
    validate_condition_configuration(condition_name)
    output_rows = []
    run_level_rows = []

    for p in P_VALUES:
        run_metrics = []

        for replicate in range(N_RUNS):
            seed = main_seed(float(p), replicate)
            model, metrics = run_single_simulation(
                float(p),
                seed,
                n_attractors=N_ATTRACTORS,
                collect_transition_events=True,
                collect_step_observations=False,
            )
            metrics['boundary_separation'] = metrics['boundary_embedded'] - metrics['boundary_exploratory']
            metrics['interface_asymmetry'] = metrics['interface_exploratory'] - metrics['interface_embedded']
            run_level_rows.append({
                'condition': condition_name,
                'condition_label': CONDITIONS[condition_name]['label'],
                'exploratory_fraction': float(p),
                'replicate': int(replicate),
                'seed': int(seed),
                **metrics,
            })
            run_metrics.append(metrics)
            del model

        run_df = pd.DataFrame(run_metrics)
        row = {'condition': condition_name, 'condition_label': CONDITIONS[condition_name]['label'], 'exploratory_fraction': float(p), 'n_runs': N_RUNS, 'n_agents': N_AGENTS, 'K': N_ATTRACTORS, 'steps': N_STEPS, 'width': GRID_WIDTH, 'height': GRID_HEIGHT}
        for column in run_df.columns:
            valid = run_df[column].dropna()
            row[column] = valid.mean() if not valid.empty else np.nan
            row[f'{column}_std'] = valid.std(ddof=1) if len(valid) > 1 else np.nan
            row[f'{column}_n'] = int(valid.count())
        output_rows.append(row)

        gc.collect()
        print(f'p={p:.3f} done')

    result = pd.DataFrame(output_rows)
    result.to_csv(DATA_DIR / f'{condition_name}_aggregated.csv', index=False)
    pd.DataFrame(run_level_rows).to_csv(DATA_DIR / f'{condition_name}_run_level.csv', index=False)
    return result


def run_main_conditions():
    all_results = {}

    for condition_name in MAIN_CONDITIONS:
        all_results[condition_name] = aggregate_condition(condition_name)

    merged = pd.concat(all_results.values(), ignore_index=True)
    merged.to_csv(DATA_DIR / 'all_conditions_aggregated.csv', index=False)

    if not np.allclose(merged['num_attractors'].to_numpy(), float(N_ATTRACTORS)):
        raise RuntimeError('Fixed-landscape audit failed: aggregated num_attractors is not identically K.')

    return all_results


def _paired_bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator):
    """Percentile CI for a mean paired difference, resampling run pairs."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    indices = rng.integers(
        0,
        values.size,
        size=(PAIRED_BOOTSTRAP_SAMPLES, values.size),
    )
    boot_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(boot_means, [0.025, 0.975])
    return (float(lower), float(upper))


def write_paired_efficiency_effects() -> None:
    """Save paired condition contrasts for the Figure 3 mechanisms."""
    metric_names = [
        'exploratory_transitions_per_1000_moves',
        'exploratory_unique_edges_per_1000_moves',
        'exploratory_unique_edge_yield',
    ]
    comparisons = [
        ('interface_only', 'baseline'),
        ('full_model', 'interface_only'),
        ('full_model', 'baseline'),
    ]
    run_level = {
        name: pd.read_csv(DATA_DIR / f'{name}_run_level.csv')
        for name in STRUCTURED_CONDITIONS
    }
    rng = np.random.default_rng(PAIRED_BOOTSTRAP_SEED)
    rows = []

    for minuend, subtrahend in comparisons:
        left = run_level[minuend]
        right = run_level[subtrahend]
        for p in P_VALUES:
            left_p = left[left['exploratory_fraction'] == float(p)]
            right_p = right[right['exploratory_fraction'] == float(p)]
            for metric in metric_names:
                paired = left_p[['replicate', 'seed', metric]].merge(
                    right_p[['replicate', 'seed', metric]],
                    on=['replicate', 'seed'],
                    how='inner',
                    suffixes=('_left', '_right'),
                    validate='one_to_one',
                )
                differences = (
                    paired[f'{metric}_left'] - paired[f'{metric}_right']
                ).to_numpy(dtype=float)
                differences = differences[np.isfinite(differences)]
                ci_low, ci_high = _paired_bootstrap_mean_ci(differences, rng)
                rows.append({
                    'comparison': f'{minuend}_minus_{subtrahend}',
                    'condition_left': minuend,
                    'condition_right': subtrahend,
                    'exploratory_fraction': float(p),
                    'metric': metric,
                    'n_valid_pairs': int(differences.size),
                    'mean_paired_difference': (
                        float(differences.mean()) if differences.size else np.nan
                    ),
                    'bootstrap_ci_95_low': ci_low,
                    'bootstrap_ci_95_high': ci_high,
                    'bootstrap_samples': PAIRED_BOOTSTRAP_SAMPLES,
                })

    pd.DataFrame(rows).to_csv(
        DATA_DIR / 'figure_3_paired_efficiency_effects.csv',
        index=False,
    )
    print(
        'Saved paired Figure 3 effects: '
        f'{DATA_DIR / "figure_3_paired_efficiency_effects.csv"}'
    )

def make_figure_1(all_results: Dict[str, pd.DataFrame]) -> None:
    baseline = all_results['baseline']
    full = all_results['full_model']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for df, prefix, linestyle in [(baseline, 'Baseline', '-'), (full, 'Full', '--')]:
        for role in ['embedded', 'exploratory']:
            x = df['exploratory_fraction'].to_numpy()
            y = df[f'interface_{role}'].to_numpy()
            std = df[f'interface_{role}_std'].fillna(0.0).to_numpy()
            n_eff = df[f'interface_{role}_n'].to_numpy()
            ci = np.full(len(df), np.nan, dtype=float)
            valid = n_eff > 0
            ci[valid] = CI_Z * std[valid] / np.sqrt(n_eff[valid])
            axes[0].plot(x, y, marker='o', linestyle=linestyle, label=f'{prefix} {role}')
            axes[0].fill_between(x, y - ci, y + ci, alpha=0.18)
    axes[0].set_title('Interface occupation')
    axes[0].set_xlabel('Exploratory fraction')
    axes[0].set_ylabel('Mean interface occupation ratio')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for df, prefix, linestyle in [(baseline, 'Baseline', '-'), (full, 'Full', '--')]:
        for role in ['embedded', 'exploratory']:
            x = df['exploratory_fraction'].to_numpy()
            y = df[f'boundary_{role}'].to_numpy()
            std = df[f'boundary_{role}_std'].fillna(0.0).to_numpy()
            n_eff = df[f'boundary_{role}_n'].to_numpy()
            ci = np.full(len(df), np.nan, dtype=float)
            valid = n_eff > 0
            ci[valid] = CI_Z * std[valid] / np.sqrt(n_eff[valid])
            axes[1].plot(x, y, marker='o', linestyle=linestyle, label=f'{prefix} {role}')
            axes[1].fill_between(x, y - ci, y + ci, alpha=0.18)
    axes[1].set_title('Inter-basin boundary distance')
    axes[1].set_xlabel('Exploratory fraction')
    axes[1].set_ylabel('Mean boundary distance')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    save_current_figure('01_Figure_1_baseline_vs_full.png')

def _summarize_one_run_probability(step_df: pd.DataFrame, *, condition_name: str, exploratory_fraction: float, replicate: int, movement_scale: float, bins: np.ndarray) -> pd.DataFrame:
    """
    Reduce one run to bin-level conditional transition probabilities.

    Crucially, boundary distance is taken directly from the model's
    exact Voronoi-boundary metric.  No |d2-d1| proxy is recomputed.
    """
    if step_df.empty:
        return pd.DataFrame()
    if movement_scale <= 0.0:
        raise ValueError('movement_scale must be > 0 for normalized boundary distance.')
    df = step_df[['agent_type', 'transition_occurred', 'pre_boundary_distance']].dropna(subset=['pre_boundary_distance']).copy()
    df['normalized_pre_boundary_distance'] = df['pre_boundary_distance'] / movement_scale
    df['distance_bin'] = pd.cut(df['normalized_pre_boundary_distance'], bins=bins, include_lowest=True)
    df = df[df['distance_bin'].notna()]
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(['agent_type', 'distance_bin'], observed=True).agg(n_obs=('transition_occurred', 'size'), n_trans=('transition_occurred', 'sum'), mean_normalized_d=('normalized_pre_boundary_distance', 'mean')).reset_index()
    grouped = grouped[grouped['n_obs'] > 0].copy()
    grouped['transition_probability'] = grouped['n_trans'] / grouped['n_obs']
    grouped['condition'] = condition_name
    grouped['condition_label'] = CONDITIONS[condition_name]['label']
    grouped['exploratory_fraction'] = float(exploratory_fraction)
    grouped['replicate'] = int(replicate)
    return grouped

def collect_transition_event_data(n_bins: int=20):
    probability_chunks = []
    transition_chunks = []
    diagnostic_rows = []
    movement_scale = float(getattr(config, 'EXPLORATORY_MOVE_SCALE', 2.0))
    if movement_scale <= 0.0:
        raise ValueError('EXPLORATORY_MOVE_SCALE must be > 0 for Figure 2.')
    max_normalized_distance = TRANSITION_PROBABILITY_MAX_NORMALIZED_DISTANCE
    bins = np.linspace(0.0, max_normalized_distance, n_bins + 1)
    print('\n=== Running transition-event collection for Figure 2 ===')
    for condition_name in TRANSITION_EVENT_CONDITIONS:
        apply_condition(condition_name)
        validate_condition_configuration(condition_name)
        print(f'condition={condition_name}')
        for p_value in TRANSITION_EVENT_P_VALUES:
            print(f'  p={p_value:.3f}')
            for replicate in range(N_RUNS):
                # Identical (p, replicate) seeds pair initialization across
                # conditions; movement histories diverge after initialization.
                seed = TRANSITION_EVENT_BASE_SEED + int(p_value * 10000) + replicate
                model, _ = run_single_simulation(float(p_value), seed, n_attractors=N_ATTRACTORS, collect_transition_events=True, collect_step_observations=True)
                step_df = model.export_step_observation_dataframe()
                transition_df = model.export_transition_event_dataframe()
                n_exploratory = sum((agent.agent_type == 'exploratory' for agent in model.agents_list))
                n_transitions = int(model.total_transitions)
                diagnostic_rows.append({'condition': condition_name, 'condition_label': CONDITIONS[condition_name]['label'], 'exploratory_fraction': float(p_value), 'replicate': int(replicate), 'seed': int(seed), 'n_exploratory_realized': int(n_exploratory), 'total_transitions': n_transitions, 'transition_log_rows': int(len(transition_df)), 'step_observation_rows': int(len(step_df)), 'exploratory_move_scale': movement_scale})
                run_summary = _summarize_one_run_probability(step_df, condition_name=condition_name, exploratory_fraction=float(p_value), replicate=replicate, movement_scale=movement_scale, bins=bins)
                if not run_summary.empty:
                    probability_chunks.append(run_summary)
                if not transition_df.empty:
                    transition_df = transition_df[['agent_type', 'event_boundary_distance']].dropna(subset=['event_boundary_distance']).copy()
                    transition_df['normalized_event_boundary_distance'] = transition_df['event_boundary_distance'] / movement_scale
                    transition_df['condition'] = condition_name
                    transition_df['condition_label'] = CONDITIONS[condition_name]['label']
                    transition_df['exploratory_fraction'] = float(p_value)
                    transition_df['replicate'] = int(replicate)
                    transition_chunks.append(transition_df)
                del step_df
                del transition_df
                del model
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(DATA_DIR / 'figure_2_transition_diagnostics.csv', index=False)
    run_level_columns = ['agent_type', 'distance_bin', 'n_obs', 'n_trans', 'mean_normalized_d', 'transition_probability', 'condition', 'condition_label', 'exploratory_fraction', 'replicate']
    transition_columns = ['agent_type', 'event_boundary_distance', 'normalized_event_boundary_distance', 'condition', 'condition_label', 'exploratory_fraction', 'replicate']
    run_level = pd.concat(probability_chunks, ignore_index=True) if probability_chunks else pd.DataFrame(columns=run_level_columns)
    transitions = pd.concat(transition_chunks, ignore_index=True) if transition_chunks else pd.DataFrame(columns=transition_columns)
    run_level.to_csv(DATA_DIR / 'figure_2_transition_probability_run_level.csv', index=False)
    transitions.to_csv(DATA_DIR / 'figure_2_transition_events.csv', index=False)
    if not transitions.empty:
        transition_counts = transitions.groupby(['condition', 'condition_label', 'agent_type'], as_index=False).size().rename(columns={'size': 'n_transition_events'})
        transition_counts.to_csv(DATA_DIR / 'figure_2_transition_counts_by_condition_agent.csv', index=False)
    total_events = int(diagnostics['total_transitions'].sum())
    print(f'Figure 2 diagnostic: {total_events} total transitions across {len(diagnostics)} runs.')
    by_condition_p = diagnostics.groupby(['condition', 'exploratory_fraction'], as_index=False).agg(mean_realized_explorers=('n_exploratory_realized', 'mean'), runs_with_transitions=('total_transitions', lambda s: int((s > 0).sum())), total_transitions=('total_transitions', 'sum'), mean_transitions=('total_transitions', 'mean'))
    print(by_condition_p.to_string(index=False))
    if total_events == 0:
        raise RuntimeError(f'Figure 2 cannot be generated because the structured comparison produced zero attractor transitions for p={TRANSITION_EVENT_P_VALUES} over {N_RUNS} runs per condition. See paper_submission_outputs/data/figure_2_transition_diagnostics.csv.')
    if run_level.empty:
        raise RuntimeError('Transition events were recorded, but no valid pre-boundary-distance probability bins were produced. Inspect figure_2_transition_diagnostics.csv and step-observation logging.')
    summary = run_level.groupby(['condition', 'condition_label', 'agent_type', 'distance_bin'], observed=False).agg(mean_prob=('transition_probability', 'mean'), std_prob=('transition_probability', 'std'), mean_normalized_d=('mean_normalized_d', 'mean'), n_run_bins=('transition_probability', 'count')).reset_index()
    n_eff = summary['n_run_bins'].clip(lower=1).to_numpy()
    summary['ci_prob'] = np.nan
    valid = summary['n_run_bins'].to_numpy() > 1
    summary.loc[valid, 'ci_prob'] = CI_Z * summary.loc[valid, 'std_prob'].to_numpy() / np.sqrt(n_eff[valid])
    summary.to_csv(DATA_DIR / 'figure_2_transition_probability_by_distance.csv', index=False)
    return (summary, transitions)

def make_figure_2() -> bool:
    grouped, transitions = collect_transition_event_data(n_bins=20)
    if transitions.empty or grouped.empty:
        status_path = DATA_DIR / 'figure_2_status.txt'
        status_path.write_text('Figure 2 was not generated because the dedicated event-level runs did not provide sufficient transition events and/or valid boundary-distance probability bins.\nInspect figure_2_transition_diagnostics.csv, figure_2_transition_events.csv and figure_2_transition_probability_run_level.csv.\n', encoding='utf-8')
        print(f'Figure 2 not generated. Diagnostic status saved to: {status_path}')
        return False
    for condition_name in TRANSITION_EVENT_CONDITIONS:
        event_rows = transitions[(transitions['condition'] == condition_name) & (transitions['agent_type'] == 'exploratory')]
        probability_rows = grouped[(grouped['condition'] == condition_name) & (grouped['agent_type'] == 'exploratory') & (grouped['n_run_bins'] > 0)]
        if event_rows.empty or probability_rows.empty:
            raise RuntimeError(
                'Figure 2 comparison is incomplete for condition '
                f'{condition_name!r}: event_rows={len(event_rows)}, '
                f'probability_rows={len(probability_rows)}.'
            )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    event_bins = np.linspace(0.0, 1.5, 26)
    for condition_name in TRANSITION_EVENT_CONDITIONS:
        sub = transitions[(transitions['condition'] == condition_name) & (transitions['agent_type'] == 'exploratory')]
        if sub.empty:
            continue
        axes[0].hist(sub['normalized_event_boundary_distance'], bins=event_bins, density=True, histtype='step', linewidth=2, label=CONDITIONS[condition_name]['label'])
    axes[0].set_title('Exploratory transition-event density')
    axes[0].set_xlabel(r'Normalized event distance $d_B/\delta_e$')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for condition_name in TRANSITION_EVENT_CONDITIONS:
        sub = grouped[(grouped['condition'] == condition_name) & (grouped['agent_type'] == 'exploratory')].copy()
        sub = sub[sub['n_run_bins'] > 0].sort_values('mean_normalized_d')
        if sub.empty:
            continue
        x = sub['mean_normalized_d'].to_numpy()
        y = sub['mean_prob'].to_numpy()
        ci = sub['ci_prob'].fillna(0.0).to_numpy()
        axes[1].plot(x, y, marker='o', label=CONDITIONS[condition_name]['label'])
        axes[1].fill_between(x, np.maximum(0.0, y - ci), np.minimum(1.0, y + ci), alpha=0.18)
    axes[1].set_title('Exploratory conditional transition probability')
    axes[1].set_xlabel(r'Normalized pre-move distance $d_{B,\mathrm{pre}}/\delta_e$')
    axes[1].set_ylabel('P(transition | distance bin)')
    axes[1].set_xlim(0.0, TRANSITION_PROBABILITY_MAX_NORMALIZED_DISTANCE)
    axes[1].set_ylim(-0.01, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    save_current_figure('02_Figure_2_boundary_localized_transition_events.png')
    return True

def make_figure_3(all_results: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_with_condition_lines(
        axes[0],
        all_results,
        'exploratory_transitions_per_1000_moves',
        'Exploratory transition rate',
        r'Transitions per $10^3$ exploratory moves',
        STRUCTURED_CONDITIONS,
    )
    plot_with_condition_lines(
        axes[1],
        all_results,
        'exploratory_unique_edges_per_1000_moves',
        'Unique-edge production',
        r'Unique directed edges per $10^3$ exploratory moves',
        STRUCTURED_CONDITIONS,
    )
    plot_with_condition_lines(
        axes[2],
        all_results,
        'exploratory_unique_edge_yield',
        'Unique-edge yield',
        'Unique directed edges per transition',
        STRUCTURED_CONDITIONS,
        bounds=(0.0, 1.0),
    )
    for ax in axes:
        ax.set_ylim(bottom=0.0)
    axes[0].legend()
    plt.tight_layout()
    save_current_figure('03_Figure_3_transition_network_expansion.png')

def make_figure_4(all_results: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_with_condition_lines(axes[0], all_results, 'giant_weak_component', 'Giant weak component', 'Relative size', STRUCTURED_CONDITIONS, bounds=(0.0, 1.0))
    plot_with_condition_lines(axes[1], all_results, 'giant_strong_component', 'Giant strong component', 'Relative size', STRUCTURED_CONDITIONS, bounds=(0.0, 1.0))
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend()
    plt.tight_layout()
    save_current_figure('04_Figure_4_global_connectivity.png')

def plot_derived_metric_with_ci(ax, all_results, include, metric, title, ylabel) -> None:
    for condition_name in include:
        df = all_results[condition_name]
        x = df['exploratory_fraction'].to_numpy()
        y = df[metric].to_numpy()
        std = df[f'{metric}_std'].fillna(0.0).to_numpy()
        n_eff = df[f'{metric}_n'].to_numpy()
        ci = np.full(len(df), np.nan, dtype=float)
        valid = n_eff > 0
        ci[valid] = CI_Z * std[valid] / np.sqrt(n_eff[valid])
        ax.plot(x, y, marker='o', label=CONDITIONS[condition_name]['label'])
        ax.fill_between(x, y - ci, y + ci, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel('Exploratory fraction')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

def collect_main_K_robustness() -> pd.DataFrame:
    """
    Run the canonical native-geometry K-robustness experiment maintained in
    supplementary_robustness_scaling.py.

    The supplementary module remains the single source of truth for the S2
    simulation design, seeds, parameter values and aggregation.  Only the
    main-paper visualization is defined in this file.
    """
    apply_full_model()
    validate_condition_configuration('full_model')
    results = robustness.run_S2()

    required_columns = {
        'K',
        'exploratory_fraction',
        'n_runs',
        'giant_weak_component',
        'giant_weak_component_std',
        'n_edges',
        'n_edges_std',
    }
    missing = sorted(required_columns - set(results.columns))
    if missing:
        raise RuntimeError(
            'Native-K robustness output is missing columns required by '
            f'Figure 5: {missing}'
        )

    observed_k = sorted(int(k) for k in results['K'].dropna().unique())
    expected_k = sorted(int(k) for k in robustness.S2_K_VALUES)
    if observed_k != expected_k:
        raise RuntimeError(
            'Native-K robustness output does not match the canonical S2 K '
            f'values: observed={observed_k}, expected={expected_k}'
        )

    results.to_csv(DATA_DIR / 'figure_5_K_robustness.csv', index=False)
    return results


def make_figure_5_K_robustness(results: pd.DataFrame) -> None:
    """
    Produce the main-paper view of the canonical S2 robustness results.

    Supplementary Figure S2 reports Cw and graph density.  The main-paper
    figure instead pairs Cw with the absolute number of directed edges so the
    two figures are complementary rather than exact duplicates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for k in robustness.S2_K_VALUES:
        df = results[results['K'] == k].sort_values('exploratory_fraction').copy()
        if df.empty:
            raise RuntimeError(f'Figure 5 has no native-K robustness rows for K={k}.')

        x = df['exploratory_fraction'].to_numpy(dtype=float)
        n_eff = df['n_runs'].to_numpy(dtype=float)
        if np.any(n_eff <= 0):
            raise RuntimeError(f'Figure 5 contains a non-positive run count for K={k}.')

        cw = df['giant_weak_component'].to_numpy(dtype=float)
        cw_std = df['giant_weak_component_std'].fillna(0.0).to_numpy(dtype=float)
        cw_ci = CI_Z * cw_std / np.sqrt(n_eff)
        axes[0].plot(x, cw, marker='o', label=f'K={k}')
        axes[0].fill_between(
            x,
            np.clip(cw - cw_ci, 0.0, 1.0),
            np.clip(cw + cw_ci, 0.0, 1.0),
            alpha=0.18,
        )

        edges = df['n_edges'].to_numpy(dtype=float)
        edges_std = df['n_edges_std'].fillna(0.0).to_numpy(dtype=float)
        edges_ci = CI_Z * edges_std / np.sqrt(n_eff)
        axes[1].plot(x, edges, marker='o', label=f'K={k}')
        axes[1].fill_between(
            x,
            np.maximum(edges - edges_ci, 0.0),
            edges + edges_ci,
            alpha=0.18,
        )

    axes[0].set_title('Global connectivity across landscape size')
    axes[0].set_xlabel('Exploratory fraction')
    axes[0].set_ylabel(r'Largest weak component $C_w$')
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title('Absolute transition-network growth')
    axes[1].set_xlabel('Exploratory fraction')
    axes[1].set_ylabel(r'Observed directed edges $|E|$')
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_current_figure('05_Figure_5_K_robustness.png')


def make_figure_6(all_results: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_with_condition_lines(axes[0, 0], all_results, 'boundary_exploratory', 'Exploratory boundary distance', 'Distance', STRUCTURED_CONDITIONS)
    plot_with_condition_lines(axes[0, 1], all_results, 'interface_exploratory', 'Exploratory interface occupation', 'Interface ratio', STRUCTURED_CONDITIONS)
    plot_derived_metric_with_ci(axes[1, 0], all_results, STRUCTURED_CONDITIONS, 'boundary_separation', 'Boundary separation', 'Embedded - exploratory distance')
    plot_derived_metric_with_ci(axes[1, 1], all_results, STRUCTURED_CONDITIONS, 'interface_asymmetry', 'Interface occupation asymmetry', 'Exploratory - embedded ratio')
    plt.tight_layout()
    save_current_figure('06_Figure_6_mechanism_decomposition.png')

def make_figure_7(all_results: Dict[str, pd.DataFrame]) -> None:
    include = ['full_model', 'flat_landscape']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_derived_metric_with_ci(axes[0, 0], all_results, include, 'boundary_separation', 'Boundary separation', 'Embedded - exploratory distance')
    plot_derived_metric_with_ci(axes[0, 1], all_results, include, 'interface_asymmetry', 'Interface occupation asymmetry', 'Exploratory - embedded ratio')
    plot_with_condition_lines(axes[1, 0], all_results, 'graph_density', 'Network density', 'Density', include, bounds=(0.0, 1.0))
    plot_with_condition_lines(axes[1, 1], all_results, 'giant_weak_component', 'Giant weak component', 'Relative size', include, bounds=(0.0, 1.0))
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 0].legend()
    axes[1, 1].legend()
    plt.tight_layout()
    save_current_figure('07_Figure_7_flat_landscape_control.png')

def verify_figure_set(figure_2_generated: bool) -> None:
    generated = sorted((path.name for path in FIG_DIR.iterdir() if path.suffix.lower() == '.png'))
    required = [name for name in FIGURE_FILENAMES if name != '02_Figure_2_boundary_localized_transition_events.png']
    missing_required = sorted(set(required) - set(generated))
    extra = sorted(set(generated) - set(FIGURE_FILENAMES))
    if missing_required or extra:
        raise RuntimeError(f'Unexpected main-paper figure set. Missing required={missing_required}; extra={extra}')
    figure_2_name = '02_Figure_2_boundary_localized_transition_events.png'
    if figure_2_generated and figure_2_name not in generated:
        raise RuntimeError('Figure 2 reported as generated but its PNG is missing.')
    if not figure_2_generated and figure_2_name in generated:
        raise RuntimeError('Figure 2 reported as skipped but a stale Figure 2 PNG is present.')
    print(f'Verified main-paper figure set: {len(generated)} PNG files generated.')
    if not figure_2_generated:
        print('Figure 2 was intentionally skipped because the event-level diagnostic did not support its generation.')

def write_experiment_manifest() -> None:
    """
    Export the exact experiment specification used by this script.
    """
    rows = []
    for condition_name in MAIN_CONDITIONS:
        spec = CONDITIONS[condition_name]
        row = {'condition': condition_name, 'label': spec['label'], 'N': N_AGENTS, 'K': N_ATTRACTORS, 'T': N_STEPS, 'W': GRID_WIDTH, 'H': GRID_HEIGHT, 'n_runs': N_RUNS, 'p_min': float(P_VALUES.min()), 'p_max': float(P_VALUES.max()), 'p_step': 0.005}
        for key in CONFIG_KEYS:
            row[key] = spec[key]
        rows.append(row)
    pd.DataFrame(rows).to_csv(DATA_DIR / 'main_experiment_manifest.csv', index=False)

def run_all_experiments() -> None:
    original_config = snapshot_config()
    try:
        clean_figure_directory()
        write_figure_notes()
        write_experiment_manifest()
        all_results = run_main_conditions()
        write_paired_efficiency_effects()
        make_figure_1(all_results)
        figure_2_generated = make_figure_2()
        make_figure_3(all_results)
        make_figure_4(all_results)
        robustness_results = collect_main_K_robustness()
        make_figure_5_K_robustness(robustness_results)
        make_figure_6(all_results)
        make_figure_7(all_results)
        verify_figure_set(figure_2_generated)
        print('\nMain-paper run completed.')
        print('Generated the complete requested figure set.' if figure_2_generated else 'Generated Figures 1 and 3-7; Figure 2 was skipped with diagnostics.')
        print('The canonical S2 native-K robustness routine supplied the data for main-text Figure 5.')
        print('Figure 3 separates transition frequency from topological edge yield using run-level exploratory events.')
        print('Run the remaining supplementary analyses and visualizations using their dedicated supplementary scripts.')
        print('\nOutputs: paper_submission_outputs/')
    finally:
        restore_config(original_config)
if __name__ == '__main__':
    run_all_experiments()
