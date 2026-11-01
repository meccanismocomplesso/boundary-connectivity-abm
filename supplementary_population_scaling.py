import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import config
from model import ExploratorySystemModel
OUTPUT_DIR = 'supplementary_population_scaling_outputs'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
K = 10
WIDTH = 50.0
HEIGHT = 50.0
N_STEPS = 300
N_RUNS = 20
BASE_SEED_A = 41000
BASE_SEED_B = 51000
CI_Z = 1.96
N_VALUES = [200, 500, 1000]
P_VALUES = np.round(np.arange(0.0, 0.051, 0.005), 3)
EXPLORER_COUNTS = [0, 1, 2, 4, 6, 8, 10, 15, 20]

def set_full_model():
    """
    Configure the same structured full model used in the paper.
    """
    config.NOVELTY_BONUS = 1.0
    config.INTERFACE_BONUS_MODE = 'real'
    config.EXPLORATORY_RULE = 'interface_novelty'
    config.LANDSCAPE_MODE = 'structured'

def clear_unused_logs(model):
    """
    Clear logs that are not required for this robustness test.
    """
    if hasattr(model, 'step_observation_log'):
        model.step_observation_log.clear()
    if hasattr(model, 'agent_history_rows'):
        model.agent_history_rows.clear()
    if hasattr(model, 'transition_event_log'):
        model.transition_event_log.clear()

def prepare_lightweight_model(model):
    """
    Disable agent-level DataCollector reporters.
    """
    if hasattr(model, 'datacollector'):
        try:
            model.datacollector.agent_reporters = {}
        except Exception:
            pass

def compute_ci(std, n):
    return CI_Z * std / np.sqrt(n)

def aggregate_runs(df):
    """
    Aggregate repeated simulations.
    """
    metrics = ['giant_weak_component', 'giant_strong_component', 'graph_density', 'n_edges', 'mean_degree', 'transition_rate']
    output = {}
    for metric in metrics:
        output[metric] = df[metric].mean()
        output[f'{metric}_std'] = df[metric].std(ddof=1)
    output['n_runs'] = len(df)
    return output

def extract_metrics(model):
    """
    Extract final network observables.
    """
    G = model.transition_graph
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        giant_weak = 0.0
        giant_strong = 0.0
        density = 0.0
        mean_degree = 0.0
    else:
        weak_components = list(nx.weakly_connected_components(G))
        strong_components = list(nx.strongly_connected_components(G))
        giant_weak = max((len(c) for c in weak_components)) / n_nodes if weak_components else 0.0
        giant_strong = max((len(c) for c in strong_components)) / n_nodes if strong_components else 0.0
        density = nx.density(G) if n_nodes > 1 else 0.0
        degrees = [degree for _, degree in G.degree()]
        mean_degree = float(np.mean(degrees)) if degrees else 0.0
    transition_rate = model.total_transitions / max(1, model.n_agents * N_STEPS)
    return {'giant_weak_component': float(giant_weak), 'giant_strong_component': float(giant_strong), 'graph_density': float(density), 'n_edges': int(n_edges), 'mean_degree': float(mean_degree), 'transition_rate': float(transition_rate)}

def run_single_simulation(n_agents, exploratory_fraction, seed):
    model = ExploratorySystemModel(n_agents=n_agents, n_attractors=K, exploratory_fraction=float(exploratory_fraction), width=WIDTH, height=HEIGHT, seed=seed)
    prepare_lightweight_model(model)
    for _ in range(N_STEPS):
        model.step()
        clear_unused_logs(model)
    if len(model.attractors) != K:
        raise RuntimeError(f'Fixed-landscape invariant violated: requested K={K}, observed K={len(model.attractors)}.')
    if getattr(model, 'attractor_creation_log', []):
        raise RuntimeError('Attractor creation occurred during a fixed-landscape population-scaling run.')
    metrics = extract_metrics(model)
    realized_explorers = sum((1 for agent in model.agents_list if agent.agent_type == 'exploratory'))
    metrics['realized_explorers'] = realized_explorers
    del model
    gc.collect()
    return metrics

def run_fraction_scaling():
    print('\n')
    print('=' * 70)
    print('S8A - Population size at constant exploratory fraction')
    print('=' * 70)
    rows = []
    total_jobs = len(N_VALUES) * len(P_VALUES) * N_RUNS
    job = 0
    for n_agents in N_VALUES:
        print(f'\nPopulation N={n_agents}')
        for p in P_VALUES:
            run_metrics = []
            for r in range(N_RUNS):
                job += 1
                seed = BASE_SEED_A + 100000 * n_agents + int(p * 100000) + r
                metrics = run_single_simulation(n_agents=n_agents, exploratory_fraction=p, seed=seed)
                run_metrics.append(metrics)
            df = pd.DataFrame(run_metrics)
            aggregate = aggregate_runs(df)
            row = {'N': n_agents, 'exploratory_fraction': p, 'expected_explorers': p * n_agents, 'realized_explorers_mean': df['realized_explorers'].mean(), 'realized_explorers_std': df['realized_explorers'].std(ddof=1), **aggregate}
            rows.append(row)
            print(f"N={n_agents:4d} | p={p:.3f} | Ne≈{row['realized_explorers_mean']:.2f} | Cw={row['giant_weak_component']:.3f} | {job}/{total_jobs}")
    results = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'population_fraction_scaling.csv')
    results.to_csv(path, index=False)
    return results

def run_explorer_count_scaling():
    print('\n')
    print('=' * 70)
    print('S8B - Constant expected number of exploratory agents')
    print('=' * 70)
    rows = []
    total_jobs = len(N_VALUES) * len(EXPLORER_COUNTS) * N_RUNS
    job = 0
    for n_agents in N_VALUES:
        print(f'\nPopulation N={n_agents}')
        for expected_explorers in EXPLORER_COUNTS:
            p = expected_explorers / n_agents
            run_metrics = []
            for r in range(N_RUNS):
                job += 1
                seed = BASE_SEED_B + 100000 * n_agents + 1000 * expected_explorers + r
                metrics = run_single_simulation(n_agents=n_agents, exploratory_fraction=p, seed=seed)
                run_metrics.append(metrics)
            df = pd.DataFrame(run_metrics)
            aggregate = aggregate_runs(df)
            row = {'N': n_agents, 'expected_explorers': expected_explorers, 'exploratory_fraction': p, 'realized_explorers_mean': df['realized_explorers'].mean(), 'realized_explorers_std': df['realized_explorers'].std(ddof=1), **aggregate}
            rows.append(row)
            print(f"N={n_agents:4d} | expected Ne={expected_explorers:2d} | p={p:.5f} | realized Ne={row['realized_explorers_mean']:.2f} | Cw={row['giant_weak_component']:.3f} | {job}/{total_jobs}")
    results = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'population_expected_explorer_count_scaling.csv')
    results.to_csv(path, index=False)
    return results

def plot_S8(fraction_results, count_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for n_agents in N_VALUES:
        df = fraction_results[fraction_results['N'] == n_agents].sort_values('exploratory_fraction').copy()
        x = df['exploratory_fraction'].to_numpy()
        y = df['giant_weak_component'].to_numpy()
        ci = compute_ci(df['giant_weak_component_std'].fillna(0.0).to_numpy(), N_RUNS)
        axes[0].plot(x, y, marker='o', label=f'N={n_agents}')
        axes[0].fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), alpha=0.18)
    axes[0].set_title('Population-size dependence')
    axes[0].set_xlabel('Exploratory fraction')
    axes[0].set_ylabel('$C_w$')
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for n_agents in N_VALUES:
        df = count_results[count_results['N'] == n_agents].sort_values('expected_explorers').copy()
        x = df['expected_explorers'].to_numpy()
        y = df['giant_weak_component'].to_numpy()
        ci = compute_ci(df['giant_weak_component_std'].fillna(0.0).to_numpy(), N_RUNS)
        axes[1].plot(x, y, marker='o', label=f'N={n_agents}')
        axes[1].fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), alpha=0.18)
    axes[1].set_title('Explorer-number scaling')
    axes[1].set_xlabel('Target number of exploratory agents $N_e$')
    axes[1].set_ylabel('$C_w$')
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'Supplementary_Figure_population_scaling.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nFigure saved to:\n{path}')

def plot_realized_explorer_diagnostic(count_results):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for n_agents in N_VALUES:
        df = count_results[count_results['N'] == n_agents].sort_values('expected_explorers').copy()
        ax.plot(df['expected_explorers'], df['realized_explorers_mean'], marker='o', label=f'N={n_agents}')
    max_target = max(EXPLORER_COUNTS)
    ax.plot([0, max_target], [0, max_target], linestyle='--', label='expected = realized')
    ax.set_xlabel('Expected exploratory-agent count')
    ax.set_ylabel('Mean realized exploratory-agent count')
    ax.set_title('Bernoulli-assignment diagnostic')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'population_realized_explorer_diagnostic.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    set_full_model()
    fraction_results = run_fraction_scaling()
    count_results = run_explorer_count_scaling()
    plot_S8(fraction_results, count_results)
    plot_realized_explorer_diagnostic(count_results)
    print('\nS8 population-scaling analysis completed.')