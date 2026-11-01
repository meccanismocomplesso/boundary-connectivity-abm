"""
Supplementary Figure S1: representative transition-network snapshots.

This script is intentionally separate from the main-paper pipeline.

The figure uses K=10 fixed attractor regions so that the progressive emergence
of connectivity is visually informative. Four exploratory fractions are shown:
p = 0.000, 0.005, 0.015 and 0.030.

Nodes are placed at their physical attractor centres. Directed-edge width is a
square-root transform of the accumulated transition count, normalized within
each panel.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import config
from run import run_model


OUTPUT_DIR = Path("supplementary_network_snapshots_outputs")
FIG_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

K = 10
N_AGENTS = 200
N_STEPS = 300
P_VALUES = [0.000, 0.005, 0.015, 0.030]
BASE_SEED = 17000

GRID_WIDTH = float(getattr(config, "GRID_WIDTH", 50))
GRID_HEIGHT = float(getattr(config, "GRID_HEIGHT", 50))

OUTPUT_FILE = FIG_DIR / "Supplementary_Figure_S1_transition_network_snapshots_K10.png"


def snapshot_seed(p: float) -> int:
    return BASE_SEED + int(round(p * 10000))


def configure_full_model() -> dict:
    original = {
        "NOVELTY_BONUS": config.NOVELTY_BONUS,
        "INTERFACE_BONUS_MODE": config.INTERFACE_BONUS_MODE,
        "EXPLORATORY_RULE": config.EXPLORATORY_RULE,
        "LANDSCAPE_MODE": config.LANDSCAPE_MODE,
    }

    config.NOVELTY_BONUS = 1.0
    config.INTERFACE_BONUS_MODE = "real"
    config.EXPLORATORY_RULE = "interface_novelty"
    config.LANDSCAPE_MODE = "structured"

    return original


def restore_config(original: dict) -> None:
    for key, value in original.items():
        setattr(config, key, value)


def run_snapshot(p: float):
    model = run_model(
        exploratory_fraction=p,
        n_agents=N_AGENTS,
        n_attractors=K,
        steps=N_STEPS,
        seed=snapshot_seed(p),
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        output_dir=None,
        collect_transition_events=False,
        collect_step_observations=False,
        collect_agent_history=False,
        collect_datacollector=False,
    )

    if len(model.attractors) != K:
        raise RuntimeError(f"Fixed-landscape invariant violated: requested K={K}, observed K={len(model.attractors)}.")

    positions = {attr.id: attr.center for attr in model.attractors}
    edge_rows = []

    for source, target, data in model.transition_graph.edges(data=True):
        edge_rows.append((source, target, int(data.get("weight", 1))))

    graph = model.transition_graph.copy()
    weak_components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))
    summary = {
        'exploratory_fraction': float(p),
        'seed': int(snapshot_seed(p)),
        'n_exploratory_realized': int(sum(agent.agent_type == 'exploratory' for agent in model.agents_list)),
        'total_transitions': int(model.total_transitions),
        'n_edges': int(graph.number_of_edges()),
        'giant_weak_component': float(max((len(component) for component in weak_components), default=0) / K),
        'giant_strong_component': float(max((len(component) for component in strong_components), default=0) / K),
    }

    return graph, positions, edge_rows, summary


def edge_widths(graph: nx.DiGraph) -> list[float]:
    if graph.number_of_edges() == 0:
        return []

    weights = np.array([graph[u][v].get("weight", 1) for u, v in graph.edges()], dtype=float)
    max_weight = max(float(weights.max()), 1.0)

    return (0.8 + 4.2 * np.sqrt(weights / max_weight)).tolist()


def save_edge_table(p: float, edge_rows: list[tuple]) -> None:
    path = DATA_DIR / f"S1_edges_p_{p:.3f}.csv"

    with path.open("w", encoding="utf-8") as handle:
        handle.write("source,target,transition_count\n")
        for source, target, weight in edge_rows:
            handle.write(f"{source},{target},{weight}\n")


def save_snapshot_summary(rows: list[dict]) -> None:
    path = DATA_DIR / 'S1_snapshot_summary.csv'
    fieldnames = [
        'exploratory_fraction',
        'seed',
        'n_exploratory_realized',
        'total_transitions',
        'n_edges',
        'giant_weak_component',
        'giant_strong_component',
        'caption_check',
    ]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_reported_caption(summary: dict) -> bool:
    """Check the numerical statements currently made in Figure S1's caption."""
    p = summary['exploratory_fraction']
    expected = {
        0.000: {'n_edges': 0, 'giant_weak_component': 0.10},
        0.005: {'n_edges': 0, 'giant_weak_component': 0.10},
        0.015: {'giant_weak_component': 1.00},
        0.030: {'giant_weak_component': 1.00},
    }[p]
    return all(np.isclose(summary[key], value) for key, value in expected.items())


def make_figure() -> None:
    original = configure_full_model()

    try:
        snapshots = []
        summary_rows = []

        for p in P_VALUES:
            graph, positions, edge_rows, summary = run_snapshot(p)
            snapshots.append((p, graph, positions))
            save_edge_table(p, edge_rows)
            summary['caption_check'] = 'PASS' if check_reported_caption(summary) else 'REVIEW'
            summary_rows.append(summary)

            print(
                f"p={p:.3f}: explorers={summary['n_exploratory_realized']}, "
                f"transitions={summary['total_transitions']}, edges={summary['n_edges']}, "
                f"Cw={summary['giant_weak_component']:.3f}, caption={summary['caption_check']}"
            )

        save_snapshot_summary(summary_rows)

        fig, axes = plt.subplots(1, len(P_VALUES), figsize=(18, 4.8))

        for ax, (p, graph, positions) in zip(axes, snapshots):
            widths = edge_widths(graph)

            nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=420)
            nx.draw_networkx_labels(graph, positions, ax=ax, font_size=8)

            if graph.number_of_edges() > 0:
                nx.draw_networkx_edges(
                    graph,
                    positions,
                    ax=ax,
                    width=widths,
                    arrows=True,
                    arrowsize=14,
                    connectionstyle="arc3,rad=0.08",
                )

            components = list(nx.weakly_connected_components(graph))
            cw = max((len(component) for component in components), default=0) / K

            ax.set_title(f"p = {p:.3f}\n$C_w$ = {cw:.2f}")
            ax.set_xlim(-1, GRID_WIDTH + 1)
            ax.set_ylim(-1, GRID_HEIGHT + 1)
            ax.set_aspect("equal")
            ax.axis("off")

        fig.suptitle("Transition-network growth in the full model ($K=10$)", y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved figure: {OUTPUT_FILE}")
        print(f"Saved edge tables in: {DATA_DIR}")
        if any(row['caption_check'] != 'PASS' for row in summary_rows):
            print('WARNING: at least one numerical statement in the current Figure S1 caption requires revision.')

    finally:
        restore_config(original)


if __name__ == "__main__":
    make_figure()
