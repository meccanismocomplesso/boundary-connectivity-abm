import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from model import ExploratorySystemModel


OUTPUT_DIR = "supplementary_robustness_K_outputs"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

P_VALUES = np.round(np.arange(0.0, 0.051, 0.005), 3)
N_RUNS = 20
N_STEPS = 300
BASE_SEED = 12000

N_AGENTS = getattr(config, "N_AGENTS", 200)
K_VALUES = [3, 4, 5]
CI_Z = 1.96


def set_full_model():
    config.NOVELTY_BONUS = 1.0
    config.INTERFACE_BONUS_MODE = "real"
    config.EXPLORATORY_RULE = "structured"
    config.LANDSCAPE_MODE = "structured"


def compute_ci(std, n):
    return CI_Z * std / np.sqrt(n)


def extract_metrics(model):
    G = model.transition_graph
    n_nodes = G.number_of_nodes()

    if n_nodes == 0:
        giant_weak = 0.0
        density = 0.0
    else:
        import networkx as nx
        weak_components = list(nx.weakly_connected_components(G))
        giant_weak = max(len(c) for c in weak_components) / n_nodes if weak_components else 0.0
        density = nx.density(G) if n_nodes > 1 else 0.0

    return {
        "giant_weak_component": giant_weak,
        "graph_density": density,
    }


def run_all():
    set_full_model()

    rows = []

    for k in K_VALUES:
        print(f"\nRunning K={k}")

        for p in P_VALUES:
            run_metrics = []

            for r in range(N_RUNS):
                seed = BASE_SEED + 1000 * k + int(p * 10000) + r

                model = ExploratorySystemModel(
                    n_agents=N_AGENTS,
                    n_attractors=k,
                    exploratory_fraction=float(p),
                    seed=seed,
                )

                for _ in range(N_STEPS):
                    model.step()

                run_metrics.append(extract_metrics(model))

            df = pd.DataFrame(run_metrics)

            row = {
                "K": k,
                "exploratory_fraction": p,
                "giant_weak_component": df["giant_weak_component"].mean(),
                "giant_weak_component_std": df["giant_weak_component"].std(),
                "graph_density": df["graph_density"].mean(),
                "graph_density_std": df["graph_density"].std(),
            }
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(DATA_DIR, "robustness_K_results.csv"), index=False)
    return results


def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for k in K_VALUES:
        df = results[results["K"] == k].copy()
        x = df["exploratory_fraction"].to_numpy()

        # giant weak component
        y1 = df["giant_weak_component"].to_numpy()
        ci1 = compute_ci(df["giant_weak_component_std"].fillna(0.0).to_numpy(), N_RUNS)
        axes[0].plot(x, y1, marker="o", label=f"K={k}")
        axes[0].fill_between(x, y1 - ci1, y1 + ci1, alpha=0.18)

        # graph density
        y2 = df["graph_density"].to_numpy()
        ci2 = compute_ci(df["graph_density_std"].fillna(0.0).to_numpy(), N_RUNS)
        axes[1].plot(x, y2, marker="o", label=f"K={k}")
        axes[1].fill_between(x, y2 - ci2, y2 + ci2, alpha=0.18)

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
    plt.savefig(os.path.join(FIG_DIR, "Supplementary_Figure_S2_robustness_K.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    results = run_all()
    plot_results(results)
    print("Done.")



