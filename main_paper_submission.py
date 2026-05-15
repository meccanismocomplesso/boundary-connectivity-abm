"""
Clean executable for the Complexity paper submission.

Running this file generates exactly the nine final figures, in order:

01. Figure 1  - Baseline vs full model comparison
02. Figure 2  - Boundary-localized transition events
03. Figure 3  - Transition network expansion
04. Figure 4  - Emergence of global connectivity
05. Figure 5  - Mechanism decomposition of spatial organization
06. Figure 6  - Flat-landscape control
07. Figure S1 - Transition graph snapshots
08. Figure S2 - Robustness to the number of attractors K
09. Figure S3 - Numerical transition slope dC_w/dp

All figures are saved in paper_submission_outputs/figures.
All numerical tables are saved in paper_submission_outputs/data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg") #no rendering on screen

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import config
from model import ExploratorySystemModel
from utils import competitive_boundary_distance


# ============================================================
# Output locations
# ============================================================

OUTPUT_DIR = Path("paper_submission_outputs")
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

P_VALUES = np.round(np.arange(0.0, 0.051, 0.005), 3)
N_RUNS = 20
N_STEPS = 300
CI_Z = 1.96
BASE_SEED = 1000

N_AGENTS = getattr(config, "N_AGENTS", 200)
N_ATTRACTORS = 3

# Figure 2 uses the dedicated transition-event collection.
TRANSITION_EVENT_P_VALUES = [0.01, 0.02, 0.03]
TRANSITION_EVENT_BASE_SEED = 9000

# Figure S2 robustness control.
K_VALUES = [3, 4, 5]
ROBUSTNESS_K_BASE_SEED = 12000

FIGURE_FILENAMES = [
    "01_Figure_1_baseline_vs_full.png",
    "02_Figure_2_boundary_localized_transition_events.png",
    "03_Figure_3_transition_network_expansion.png",
    "04_Figure_4_global_connectivity.png",
    "05_Figure_5_mechanism_decomposition.png",
    "06_Figure_6_flat_landscape_control.png",
    "07_Supplementary_Figure_S1_transition_graph_snapshots.png",
    "08_Supplementary_Figure_S2_robustness_K.png",
    "09_Supplementary_Figure_S3_transition_slope.png",
]


# ============================================================
# Experimental conditions
# ============================================================

CONDITIONS = {
    "baseline": {
        "label": "Baseline",
        "NOVELTY_BONUS": 0.0,
        "INTERFACE_BONUS_MODE": "none",
        "EXPLORATORY_RULE": "random_walk",
        "LANDSCAPE_MODE": "structured",
    },
    "interface_only": {
        "label": "Interface-only",
        "NOVELTY_BONUS": 0.0,
        "INTERFACE_BONUS_MODE": "real",
        "EXPLORATORY_RULE": "random_walk",
        "LANDSCAPE_MODE": "structured",
    },
    "interface_novelty": {
        "label": "Interface + novelty",
        "NOVELTY_BONUS": 1.0,
        "INTERFACE_BONUS_MODE": "real",
        "EXPLORATORY_RULE": "random_walk",
        "LANDSCAPE_MODE": "structured",
    },
    "full_model": {
        "label": "Full model",
        "NOVELTY_BONUS": 1.0,
        "INTERFACE_BONUS_MODE": "real",
        "EXPLORATORY_RULE": "structured",
        "LANDSCAPE_MODE": "structured",
    },
    "flat_landscape": {
        "label": "Flat landscape control",
        "NOVELTY_BONUS": 1.0,
        "INTERFACE_BONUS_MODE": "real",
        "EXPLORATORY_RULE": "structured",
        "LANDSCAPE_MODE": "flat_control",
    },
}

MAIN_CONDITIONS = [
    "baseline",
    "interface_only",
    "interface_novelty",
    "full_model",
    "flat_landscape",
]

STRUCTURED_CONDITIONS = [
    "baseline",
    "interface_only",
    "interface_novelty",
    "full_model",
]

CONFIG_KEYS = [
    "NOVELTY_BONUS",
    "INTERFACE_BONUS_MODE",
    "EXPLORATORY_RULE",
    "LANDSCAPE_MODE",
]


# ============================================================
# Captions / notes used by the executable output
# ============================================================

FIGURE_NOTES = {
    "01_Figure_1_baseline_vs_full.png": (
        "Figure 1. Baseline vs full model comparison. Interface occupation and "
        "boundary distance for embedded and exploratory agents as a function of "
        "the exploratory fraction p. Shaded regions indicate 95% confidence "
        "intervals across independent runs."
    ),
    "02_Figure_2_boundary_localized_transition_events.png": (
        "Figure 2. Boundary-localized transition events. Distribution of competitive "
        "boundary distance at transition events and conditional transition probability "
        "as a function of pre-move boundary distance, shown separately for embedded "
        "and exploratory agents."
    ),
    "03_Figure_3_transition_network_expansion.png": (
        "Figure 3. Transition network expansion. Network density, number of edges, "
        "and mean degree as functions of the exploratory fraction p across experimental "
        "conditions. Shaded regions indicate variability across independent runs."
    ),
    "04_Figure_4_global_connectivity.png": (
        "Figure 4. Emergence of global connectivity. Relative size of the largest "
        "weakly and strongly connected components as a function of p. As p increases, "
        "the system transitions from a fragmented regime to a globally connected "
        "structure, consistent with a percolation-like process."
    ),
    "05_Figure_5_mechanism_decomposition.png": (
        "Figure 5. Mechanism decomposition of spatial organization. Exploratory "
        "boundary distance, interface occupation, boundary separation, and interface "
        "occupation asymmetry are shown as functions of p across progressively enriched "
        "mechanisms."
    ),
    "06_Figure_6_flat_landscape_control.png": (
        "Figure 6. Flat-landscape control. Comparison between the full structured "
        "model and the flat landscape for spatial observables and network observables."
    ),
    "07_Supplementary_Figure_S1_transition_graph_snapshots.png": (
        "Supplementary Figure S1. Transition graph snapshots for the full model at "
        "p = 0.000, 0.010, and 0.030. Nodes represent discretized visited states and "
        "directed edges represent observed transitions."
    ),
    "08_Supplementary_Figure_S2_robustness_K.png": (
        "Supplementary Figure S2. Robustness to the number of attractors K. Relative "
        "size of the largest weakly connected component and network density as functions "
        "of p for K = 3, 4, and 5."
    ),
    "09_Supplementary_Figure_S3_transition_slope.png": (
        "Supplementary Figure S3. Numerical estimate of the transition slope dC_w/dp, "
        "where C_w is the size of the giant weakly connected component."
    ),
}


# ============================================================
# General helpers
# ============================================================

def clean_figure_directory() -> None:
    """Remove old image files so the run leaves exactly the final figure set."""
    for path in FIG_DIR.iterdir():
        if path.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
            path.unlink()


def write_figure_notes() -> None:
    notes_path = OUTPUT_DIR / "FIGURE_NOTES.md"
    with notes_path.open("w", encoding="utf-8") as f:
        f.write("# Final figure order and captions\n\n")
        for filename in FIGURE_FILENAMES:
            f.write(f"## {filename}\n\n{FIGURE_NOTES[filename]}\n\n")
    print(f"Saved notes: {notes_path}")


def snapshot_config() -> Dict[str, object]:
    return {k: getattr(config, k, None) for k in CONFIG_KEYS}


def restore_config(snapshot: Dict[str, object]) -> None:
    for k, v in snapshot.items():
        if v is not None:
            setattr(config, k, v)


def apply_condition(condition_name: str) -> None:
    spec = CONDITIONS[condition_name]
    for k in CONFIG_KEYS:
        setattr(config, k, spec[k])


def apply_full_model() -> None:
    apply_condition("full_model")


def compute_ci(std_values: np.ndarray, n_runs: int) -> np.ndarray:
    return CI_Z * std_values / np.sqrt(n_runs)


def save_current_figure(filename: str) -> None:
    path = FIG_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def plot_with_condition_lines(
    ax,
    all_results: Dict[str, pd.DataFrame],
    y_col: str,
    title: str,
    ylabel: str,
    include_conditions: Iterable[str],
    show_ci: bool = True,
) -> None:
    for cond in include_conditions:
        df = all_results[cond]
        x = df["exploratory_fraction"].to_numpy()
        y = df[y_col].to_numpy()
        ax.plot(x, y, marker="o", label=CONDITIONS[cond]["label"])

        std_col = f"{y_col}_std"
        if show_ci and std_col in df.columns:
            std = df[std_col].fillna(0.0).to_numpy()
            ci = compute_ci(std, N_RUNS)
            ax.fill_between(x, y - ci, y + ci, alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel("Exploratory fraction")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


# ============================================================
# Metrics
# ============================================================

def extract_agent_metrics(model: ExploratorySystemModel) -> Dict[str, float]:
    embedded_boundary, exploratory_boundary = [], []
    embedded_interface_ratios, exploratory_interface_ratios = [], []

    for agent in model.agents_list:
        if agent.boundary_distance is not None:
            if agent.agent_type == "embedded":
                embedded_boundary.append(agent.boundary_distance)
            else:
                exploratory_boundary.append(agent.boundary_distance)

        ratio = agent.interface_visits / max(1, agent.total_moves)
        if agent.agent_type == "embedded":
            embedded_interface_ratios.append(ratio)
        else:
            exploratory_interface_ratios.append(ratio)

    return {
        "boundary_embedded": float(np.mean(embedded_boundary)) if embedded_boundary else 0.0,
        "boundary_exploratory": float(np.mean(exploratory_boundary)) if exploratory_boundary else 0.0,
        "interface_embedded": float(np.mean(embedded_interface_ratios)) if embedded_interface_ratios else 0.0,
        "interface_exploratory": float(np.mean(exploratory_interface_ratios)) if exploratory_interface_ratios else 0.0,
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
        giant_weak = float(max(len(c) for c in weak_components) / n_nodes) if weak_components else 0.0
        giant_strong = float(max(len(c) for c in strong_components) / n_nodes) if strong_components else 0.0
    else:
        giant_weak = 0.0
        giant_strong = 0.0

    return {
        "graph_nodes": float(n_nodes),
        "graph_edges": float(n_edges),
        "graph_density": density,
        "graph_mean_degree": mean_degree,
        "giant_weak_component": giant_weak,
        "giant_strong_component": giant_strong,
    }


def extract_model_metrics(model: ExploratorySystemModel) -> Dict[str, float]:
    metrics = {}
    metrics.update(extract_agent_metrics(model))
    metrics.update(extract_graph_metrics(model))
    metrics["num_attractors"] = float(len(model.attractors))
    metrics["exploration_entropy"] = float(model.exploration_entropy())
    return metrics


# ============================================================
# Simulation runners
# ============================================================

def run_single_simulation(exploratory_fraction: float, seed: int, n_attractors: int = N_ATTRACTORS):
    model = ExploratorySystemModel(
        n_agents=N_AGENTS,
        n_attractors=n_attractors,
        exploratory_fraction=float(exploratory_fraction),
        seed=seed,
    )
    for _ in range(N_STEPS):
        model.step()
    return model, extract_model_metrics(model)


def aggregate_condition(condition_name: str) -> Tuple[pd.DataFrame, Dict[float, ExploratorySystemModel]]:
    rows = []
    representative_models = {}

    print(f"\n=== Running condition: {condition_name} ===")
    apply_condition(condition_name)

    for p in P_VALUES:
        run_metrics = []
        for r in range(N_RUNS):
            seed = BASE_SEED + 100_000 * MAIN_CONDITIONS.index(condition_name) + int(p * 10_000) + r
            model, metrics = run_single_simulation(float(p), seed, n_attractors=N_ATTRACTORS)
            run_metrics.append(metrics)
            if r == 0:
                representative_models[float(p)] = model

        df = pd.DataFrame(run_metrics)
        row = {
            "condition": condition_name,
            "condition_label": CONDITIONS[condition_name]["label"],
            "exploratory_fraction": float(p),
        }
        row.update({col: df[col].mean() for col in df.columns})
        row.update({f"{col}_std": df[col].std() for col in df.columns})
        rows.append(row)
        print(f"p={p:.3f} done")

    result = pd.DataFrame(rows)
    result.to_csv(DATA_DIR / f"{condition_name}_aggregated.csv", index=False)
    return result, representative_models


def run_main_conditions() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[float, ExploratorySystemModel]]]:
    all_results = {}
    representative_models = {}

    for condition_name in MAIN_CONDITIONS:
        df, reps = aggregate_condition(condition_name)
        all_results[condition_name] = df
        representative_models[condition_name] = reps

    merged = pd.concat(all_results.values(), ignore_index=True)
    merged.to_csv(DATA_DIR / "all_conditions_aggregated.csv", index=False)
    return all_results, representative_models


# ============================================================
# Figure 1: baseline vs full model comparison
# ============================================================

def make_figure_1(all_results: Dict[str, pd.DataFrame]) -> None:
    baseline = all_results["baseline"]
    full = all_results["full_model"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for df, prefix, linestyle in [(baseline, "Baseline", "-"), (full, "Full", "--")]:
        for role in ["embedded", "exploratory"]:
            x = df["exploratory_fraction"].to_numpy()
            y = df[f"interface_{role}"].to_numpy()
            ci = compute_ci(df[f"interface_{role}_std"].fillna(0.0).to_numpy(), N_RUNS)
            axes[0].plot(x, y, marker="o", linestyle=linestyle, label=f"{prefix} {role}")
            axes[0].fill_between(x, y - ci, y + ci, alpha=0.18)

    axes[0].set_title("Interface occupation")
    axes[0].set_xlabel("Exploratory fraction")
    axes[0].set_ylabel("Mean interface occupation ratio")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for df, prefix, linestyle in [(baseline, "Baseline", "-"), (full, "Full", "--")]:
        for role in ["embedded", "exploratory"]:
            x = df["exploratory_fraction"].to_numpy()
            y = df[f"boundary_{role}"].to_numpy()
            ci = compute_ci(df[f"boundary_{role}_std"].fillna(0.0).to_numpy(), N_RUNS)
            axes[1].plot(x, y, marker="o", linestyle=linestyle, label=f"{prefix} {role}")
            axes[1].fill_between(x, y - ci, y + ci, alpha=0.18)

    axes[1].set_title("Boundary distance")
    axes[1].set_xlabel("Exploratory fraction")
    axes[1].set_ylabel("Mean boundary distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_current_figure("01_Figure_1_baseline_vs_full.png")


# ============================================================
# Figure 2: boundary-localized transition events
# ============================================================


def _add_competitive_boundary_columns(df: pd.DataFrame, attractors: Iterable) -> pd.DataFrame:
    """
    Add competitive-boundary distances without row-wise apply.

    This keeps Figure 2 memory-safe on machines with limited RAM. The previous
    implementation concatenated every step observation across all runs
    (~3.6M rows) and then copied the full table during filtering. Here each run
    is processed and reduced immediately.
    """
    if df.empty:
        return df

    centers = np.asarray([attr.center for attr in attractors], dtype=float)
    if len(centers) < 2:
        df["competitive_pre_boundary_distance"] = np.nan
        df["competitive_post_boundary_distance"] = np.nan
        df["competitive_event_boundary_distance"] = np.nan
        return df

    def two_nearest_gap(x_col: str, y_col: str) -> np.ndarray:
        points = df[[x_col, y_col]].to_numpy(dtype=float)
        dists = np.sqrt(((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        two = np.partition(dists, 1, axis=1)[:, :2]
        return np.abs(two[:, 1] - two[:, 0])

    df = df.copy()
    df["competitive_pre_boundary_distance"] = two_nearest_gap("pre_x", "pre_y")
    df["competitive_post_boundary_distance"] = two_nearest_gap("post_x", "post_y")
    df["competitive_event_boundary_distance"] = df[
        ["competitive_pre_boundary_distance", "competitive_post_boundary_distance"]
    ].min(axis=1)
    return df


def _summarize_step_observations_for_probability(
    step_df: pd.DataFrame,
    exploratory_fraction: float,
    bins: np.ndarray,
) -> pd.DataFrame:
    """Reduce one run of step observations to small bin counts."""
    if step_df.empty:
        return pd.DataFrame()

    df = step_df[
        ["agent_type", "transition_occurred", "competitive_pre_boundary_distance"]
    ].dropna(subset=["competitive_pre_boundary_distance"]).copy()

    df["distance_bin"] = pd.cut(
        df["competitive_pre_boundary_distance"],
        bins=bins,
        include_lowest=True,
    )
    df = df[df["distance_bin"].notna()]
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(["agent_type", "distance_bin"], observed=False)
        .agg(
            n_obs=("transition_occurred", "size"),
            n_trans=("transition_occurred", "sum"),
            sum_d=("competitive_pre_boundary_distance", "sum"),
        )
        .reset_index()
    )
    grouped["exploratory_fraction"] = float(exploratory_fraction)
    return grouped


def collect_transition_event_data(n_bins: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect the reduced data needed for Figure 2.

    Returns
    -------
    probability_summary:
        Small table with binned conditional transition probabilities.
    transitions:
        Raw transition-event rows only, used for the density panel.
    """
    apply_full_model()
    probability_chunks = []
    transition_chunks = []

    # Fixed plotting range used by Figure 2. Values beyond this range are not
    # needed for the panel and would only inflate memory use.
    bins = np.linspace(0.0, 25.0, n_bins + 1)

    print("\n=== Running transition-event collection for Figure 2 ===")
    for p_value in TRANSITION_EVENT_P_VALUES:
        print(f"p={p_value:.3f}")
        for r in range(N_RUNS):
            seed = TRANSITION_EVENT_BASE_SEED + int(p_value * 10_000) + r
            model, _ = run_single_simulation(float(p_value), seed, n_attractors=N_ATTRACTORS)

            step_df = _add_competitive_boundary_columns(
                model.export_step_observation_dataframe(),
                model.attractors,
            )
            probability_chunks.append(
                _summarize_step_observations_for_probability(step_df, float(p_value), bins)
            )

            trans_df = _add_competitive_boundary_columns(
                model.export_transition_event_dataframe(),
                model.attractors,
            )
            if not trans_df.empty:
                trans_df["exploratory_fraction"] = float(p_value)
                keep_cols = [
                    "exploratory_fraction",
                    "agent_type",
                    "competitive_event_boundary_distance",
                ]
                transition_chunks.append(trans_df[keep_cols])

    reduced_steps = (
        pd.concat([x for x in probability_chunks if not x.empty], ignore_index=True)
        if probability_chunks
        else pd.DataFrame()
    )
    transitions = (
        pd.concat(transition_chunks, ignore_index=True)
        if transition_chunks
        else pd.DataFrame()
    )

    if reduced_steps.empty:
        probability_summary = pd.DataFrame(
            columns=["agent_type", "distance_bin", "mean_prob", "std_prob", "mean_d", "ci_prob"]
        )
    else:
        by_p = (
            reduced_steps.groupby(["exploratory_fraction", "agent_type", "distance_bin"], observed=False)
            .agg(n_obs=("n_obs", "sum"), n_trans=("n_trans", "sum"), sum_d=("sum_d", "sum"))
            .reset_index()
        )
        by_p["transition_probability"] = by_p["n_trans"] / by_p["n_obs"].clip(lower=1)
        by_p["mean_d"] = by_p["sum_d"] / by_p["n_obs"].clip(lower=1)
        by_p.to_csv(DATA_DIR / "figure_2_transition_probability_by_p_and_distance.csv", index=False)

        probability_summary = (
            by_p.groupby(["agent_type", "distance_bin"], observed=False)
            .agg(
                mean_prob=("transition_probability", "mean"),
                std_prob=("transition_probability", "std"),
                mean_d=("mean_d", "mean"),
            )
            .reset_index()
        )
        probability_summary["std_prob"] = probability_summary["std_prob"].fillna(0.0)
        probability_summary["ci_prob"] = compute_ci(
            probability_summary["std_prob"].to_numpy(), len(TRANSITION_EVENT_P_VALUES)
        )

    probability_summary.to_csv(DATA_DIR / "figure_2_transition_probability_by_distance.csv", index=False)
    transitions.to_csv(DATA_DIR / "figure_2_transition_events_competitive.csv", index=False)
    return probability_summary, transitions


def make_figure_2() -> None:
    grouped, transitions = collect_transition_event_data(n_bins=20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for agent_type in ["embedded", "exploratory"]:
        sub = transitions[
            (transitions["competitive_event_boundary_distance"].notna())
            & (transitions["agent_type"] == agent_type)
        ]
        if len(sub) == 0:
            continue
        axes[0].hist(
            sub["competitive_event_boundary_distance"],
            bins=25,
            density=True,
            histtype="step",
            linewidth=2,
            label=agent_type,
        )

    axes[0].set_title("Transition-event density")
    axes[0].set_xlabel("Competitive boundary distance at transition")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for agent_type in ["embedded", "exploratory"]:
        sub = grouped[grouped["agent_type"] == agent_type].sort_values("mean_d")
        x = sub["mean_d"].to_numpy()
        y = sub["mean_prob"].to_numpy()
        ci = sub["ci_prob"].to_numpy()
        axes[1].plot(x, y, marker="o", label=agent_type)
        axes[1].fill_between(x, y - ci, y + ci, alpha=0.18)

    axes[1].set_title("Conditional transition probability")
    axes[1].set_xlabel("Competitive pre-move boundary distance")
    axes[1].set_ylabel("P(transition | distance bin)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_current_figure("02_Figure_2_boundary_localized_transition_events.png")


# ============================================================
# Figure 3: transition network expansion
# ============================================================

def make_figure_3(all_results: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_with_condition_lines(axes[0], all_results, "graph_density", "Network density", "Density", STRUCTURED_CONDITIONS)
    plot_with_condition_lines(axes[1], all_results, "graph_edges", "Transition graph edges", "Mean edges", STRUCTURED_CONDITIONS)
    plot_with_condition_lines(axes[2], all_results, "graph_mean_degree", "Transition graph mean degree", "Mean degree", STRUCTURED_CONDITIONS)
    axes[0].legend()
    plt.tight_layout()
    save_current_figure("03_Figure_3_transition_network_expansion.png")


# ============================================================
# Figure 4: emergence of global connectivity
# ============================================================

def make_figure_4(all_results: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_with_condition_lines(axes[0], all_results, "giant_weak_component", "Giant weak component", "Ratio", STRUCTURED_CONDITIONS)
    plot_with_condition_lines(axes[1], all_results, "giant_strong_component", "Giant strong component", "Ratio", STRUCTURED_CONDITIONS)
    axes[0].legend()
    plt.tight_layout()
    save_current_figure("04_Figure_4_global_connectivity.png")


# ============================================================
# Figure 5: mechanism decomposition
# ============================================================

def plot_gap_with_ci(ax, all_results, include, metric_a, metric_b, title, ylabel) -> None:
    for cond in include:
        df = all_results[cond]
        x = df["exploratory_fraction"].to_numpy()
        y = (df[metric_a] - df[metric_b]).to_numpy()
        std_a = df[f"{metric_a}_std"].fillna(0.0).to_numpy()
        std_b = df[f"{metric_b}_std"].fillna(0.0).to_numpy()
        ci = compute_ci(np.sqrt(std_a**2 + std_b**2), N_RUNS)
        ax.plot(x, y, marker="o", label=CONDITIONS[cond]["label"])
        ax.fill_between(x, y - ci, y + ci, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Exploratory fraction")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def make_figure_5(all_results: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_with_condition_lines(
        axes[0, 0], all_results, "boundary_exploratory", "Exploratory boundary distance", "Distance", STRUCTURED_CONDITIONS
    )
    plot_with_condition_lines(
        axes[0, 1], all_results, "interface_exploratory", "Exploratory interface occupation", "Interface ratio", STRUCTURED_CONDITIONS
    )
    plot_gap_with_ci(
        axes[1, 0], all_results, STRUCTURED_CONDITIONS, "boundary_embedded", "boundary_exploratory",
        "Boundary separation", "Embedded - exploratory distance"
    )
    plot_gap_with_ci(
        axes[1, 1], all_results, STRUCTURED_CONDITIONS, "interface_exploratory", "interface_embedded",
        "Interface occupation asymmetry", "Exploratory - embedded ratio"
    )
    plt.tight_layout()
    save_current_figure("05_Figure_5_mechanism_decomposition.png")


# ============================================================
# Figure 6: flat landscape control
# ============================================================

def make_figure_6(all_results: Dict[str, pd.DataFrame]) -> None:
    include = ["full_model", "flat_landscape"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_gap_with_ci(
        axes[0, 0], all_results, include, "boundary_embedded", "boundary_exploratory",
        "Boundary separation", "Embedded - exploratory distance"
    )
    plot_gap_with_ci(
        axes[0, 1], all_results, include, "interface_exploratory", "interface_embedded",
        "Interface occupation asymmetry", "Exploratory - embedded ratio"
    )
    plot_with_condition_lines(axes[1, 0], all_results, "graph_density", "Network density", "Density", include)
    plot_with_condition_lines(axes[1, 1], all_results, "giant_weak_component", "Giant weak component", "Ratio", include)
    axes[1, 0].legend()
    axes[1, 1].legend()
    plt.tight_layout()
    save_current_figure("06_Figure_6_flat_landscape_control.png")


# ============================================================
# Supplementary Figure S1: transition graph snapshots
# ============================================================

def make_figure_s1(representative_models: Dict[str, Dict[float, ExploratorySystemModel]]) -> None:
    chosen_ps = [0.0, 0.01, 0.03]
    reps = representative_models["full_model"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, p in zip(axes, chosen_ps):
        model = reps[float(p)]
        G = model.transition_graph
        pos = nx.spring_layout(G, seed=42) if G.number_of_nodes() else {}
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=450, font_size=8, arrows=True)
        ax.set_title(f"Full model, p = {p:.3f}")

    plt.tight_layout()
    save_current_figure("07_Supplementary_Figure_S1_transition_graph_snapshots.png")


# ============================================================
# Supplementary Figure S2: robustness to K
# ============================================================

def run_robustness_k() -> pd.DataFrame:
    rows = []
    apply_full_model()
    print("\n=== Running robustness to K for Supplementary Figure S2 ===")

    for k in K_VALUES:
        print(f"K={k}")
        for p in P_VALUES:
            run_metrics = []
            for r in range(N_RUNS):
                seed = ROBUSTNESS_K_BASE_SEED + 1000 * k + int(p * 10_000) + r
                model, metrics = run_single_simulation(float(p), seed, n_attractors=k)
                run_metrics.append({
                    "giant_weak_component": metrics["giant_weak_component"],
                    "graph_density": metrics["graph_density"],
                })
            df = pd.DataFrame(run_metrics)
            rows.append({
                "K": k,
                "exploratory_fraction": float(p),
                "giant_weak_component": df["giant_weak_component"].mean(),
                "giant_weak_component_std": df["giant_weak_component"].std(),
                "graph_density": df["graph_density"].mean(),
                "graph_density_std": df["graph_density"].std(),
            })

    result = pd.DataFrame(rows)
    result.to_csv(DATA_DIR / "supplementary_s2_robustness_K_results.csv", index=False)
    return result


def make_figure_s2() -> None:
    results = run_robustness_k()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for k in K_VALUES:
        df = results[results["K"] == k].copy()
        x = df["exploratory_fraction"].to_numpy()

        y = df["giant_weak_component"].to_numpy()
        ci = compute_ci(df["giant_weak_component_std"].fillna(0.0).to_numpy(), N_RUNS)
        axes[0].plot(x, y, marker="o", label=f"K={k}")
        axes[0].fill_between(x, y - ci, y + ci, alpha=0.18)

        y = df["graph_density"].to_numpy()
        ci = compute_ci(df["graph_density_std"].fillna(0.0).to_numpy(), N_RUNS)
        axes[1].plot(x, y, marker="o", label=f"K={k}")
        axes[1].fill_between(x, y - ci, y + ci, alpha=0.18)

    axes[0].set_title("Giant weak component")
    axes[0].set_xlabel("Exploratory fraction")
    axes[0].set_ylabel("Ratio")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Network density")
    axes[1].set_xlabel("Exploratory fraction")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_current_figure("08_Supplementary_Figure_S2_robustness_K.png")


# ============================================================
# Supplementary Figure S3: transition slope dC_w/dp
# ============================================================

def make_figure_s3(all_results: Dict[str, pd.DataFrame]) -> None:
    df = all_results["full_model"].sort_values("exploratory_fraction")
    p = df["exploratory_fraction"].to_numpy()
    C = df["giant_weak_component"].to_numpy()
    dCdp = np.gradient(C, p)
    idx_max = int(np.argmax(dCdp))
    p_star = float(p[idx_max])

    pd.DataFrame({"exploratory_fraction": p, "giant_weak_component": C, "dCdp": dCdp}).to_csv(
        DATA_DIR / "supplementary_s3_transition_slope.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(p, C, marker="o")
    axes[0].axvline(p_star, linestyle="--", alpha=0.6)
    axes[0].set_title("Giant weak component")
    axes[0].set_xlabel("Exploratory fraction")
    axes[0].set_ylabel("C_w")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(p, dCdp, marker="o")
    axes[1].axvline(p_star, linestyle="--", alpha=0.6, label=f"p* ≈ {p_star:.3f}")
    axes[1].set_title("Transition slope dC/dp")
    axes[1].set_xlabel("Exploratory fraction")
    axes[1].set_ylabel("dC/dp")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_current_figure("09_Supplementary_Figure_S3_transition_slope.png")


# ============================================================
# Master run
# ============================================================

def verify_exact_figure_set() -> None:
    generated = sorted(p.name for p in FIG_DIR.iterdir() if p.suffix.lower() == ".png")
    expected = sorted(FIGURE_FILENAMES)
    if generated != expected:
        missing = sorted(set(expected) - set(generated))
        extra = sorted(set(generated) - set(expected))
        raise RuntimeError(f"Unexpected figure set. Missing={missing}; extra={extra}")
    print("\nVerified: exactly 9 final figure files generated.")


def run_all_experiments() -> None:
    original_config = snapshot_config()
    try:
        clean_figure_directory()
        write_figure_notes()

        all_results, representative_models = run_main_conditions()

        make_figure_1(all_results)
        make_figure_2()
        make_figure_3(all_results)
        make_figure_4(all_results)
        make_figure_5(all_results)
        make_figure_6(all_results)
        make_figure_s1(representative_models)
        #make_figure_s2()
        print("\nSkipped Supplementary Figure S2 and S3 in main.")
        print("Run S2 separately with: python figure_S2_robustness_K.py")
        #make_figure_s3(all_results)
        print("Run S3 separately with: python figure_S3_transition_slope.py")
        print("\nMain run completed. Generated Figures 1-6 and S1.")

        # verify_exact_figure_set()
        print("\nDone. Final figures and data generated in paper_submission_outputs/.")
    finally:
        restore_config(original_config)


if __name__ == "__main__":
    run_all_experiments()
