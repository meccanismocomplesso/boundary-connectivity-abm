import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import config
from model import ExploratorySystemModel
OUTPUT_DIR = 'supplementary_K_robustness_outputs'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
P_VALUES = np.round(np.arange(0.0, 0.051, 0.005), 3)
N_RUNS = 20
N_STEPS = 300
CI_Z = 1.96
BASE_WIDTH = 50.0
BASE_HEIGHT = 50.0
BASE_N_AGENTS = 200
S2_K_VALUES = [3, 5, 10, 20]
S2_BASE_SEED = 12000
S3_K_VALUES = [3, 5, 10, 20, 50]
S3_BASE_SEED = 22000
REFERENCE_K = 10
REFERENCE_WIDTH = 50.0
TARGET_ATTRACTOR_SPACING = 0.6 * REFERENCE_WIDTH * np.sin(np.pi / REFERENCE_K)
AGENTS_PER_ATTRACTOR = 20
S3_TIME_K = 50
S3_TIME_VALUES = [300, 600, 1200]
S3_TIME_P_VALUES = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
S3_TIME_N_RUNS = 20
S3_TIME_BASE_SEED = 27000
S4_K_VALUES = [3, 5, 10, 15, 20, 25, 28, 30, 32, 34, 36, 38, 40, 42, 45, 50]
S4_BASE_SEED = 32000
S4_P = 0.0
EMBEDDED_NOISE = config.EMBEDDED_MOVE_SCALE

def set_full_model():
    """
    Configure the same full structured exploratory regime
    used in the manuscript.
    """
    config.NOVELTY_BONUS = 1.0
    config.INTERFACE_BONUS_MODE = 'real'
    config.EXPLORATORY_RULE = 'interface_novelty'
    config.LANDSCAPE_MODE = 'structured'

def clear_unused_logs(model):
    """
    Remove large logs that are not required for the K
    robustness analyses.

    The cumulative transition graph and dynamical state
    are preserved.
    """
    if hasattr(model, 'step_observation_log'):
        model.step_observation_log.clear()
    if hasattr(model, 'agent_history_rows'):
        model.agent_history_rows.clear()
    if hasattr(model, 'transition_event_log'):
        model.transition_event_log.clear()

def prepare_lightweight_model(model):
    """
    Disable DataCollector agent reporters.

    Model-level reporters can remain active because their
    memory cost is small compared with agent-level storage.
    """
    if hasattr(model, 'datacollector'):
        try:
            model.datacollector.agent_reporters = {}
        except Exception:
            pass

def compute_ci(std, n):
    """
    Approximate 95% confidence interval using 1.96 SE.
    """
    return CI_Z * std / np.sqrt(n)

def aggregate_runs(df):
    """
    Aggregate independent simulation realizations.
    """
    output = {}
    metrics = ['giant_weak_component', 'giant_strong_component', 'graph_density', 'n_edges', 'mean_degree', 'normalized_mean_degree', 'transition_rate', 'embedded_boundary_distance', 'embedded_interface_ratio']
    for metric in metrics:
        output[metric] = df[metric].mean()
        output[f'{metric}_std'] = df[metric].std(ddof=1)
    output['n_runs'] = len(df)
    return output

def native_nearest_neighbour_spacing(k, width, height=None):
    """
    Minimum nearest-neighbour attractor spacing implied by
    model.py.

    K = 3:
        special triangular geometry.

    K != 3:
        equally spaced attractors on a ring of radius
        0.30 * min(width, height).
    """
    if height is None:
        height = width
    if k == 3:
        centers = np.array([[0.25 * width, 0.3 * height], [0.75 * width, 0.3 * height], [0.5 * width, 0.75 * height]])
        distances = []
        for i in range(3):
            for j in range(i + 1, 3):
                d = np.linalg.norm(centers[i] - centers[j])
                distances.append(d)
        return float(min(distances))
    radius = 0.3 * min(width, height)
    return float(2.0 * radius * np.sin(np.pi / k))

def scaled_domain_size(k):
    """
    Domain size used in S3.

    The domain is scaled so that the nearest-neighbour
    attractor spacing remains equal to
    TARGET_ATTRACTOR_SPACING.

    For K != 3:

        d_nn = 0.60 * W * sin(pi/K)

    For K = 3, the minimum spacing of the special triangular
    geometry is approximately:

        d_nn = 0.50 * W
    """
    if k == 3:
        return float(TARGET_ATTRACTOR_SPACING / 0.5)
    return float(TARGET_ATTRACTOR_SPACING / (0.6 * np.sin(np.pi / k)))

def scaled_agent_number(k):
    """
    Preserve a constant number of agents per attractor
    in S3.
    """
    return int(AGENTS_PER_ATTRACTOR * k)

def geometric_crowding_ratio(k, width, height):
    """
    Approximate separation between an attractor centre and
    its nearest Voronoi boundary, expressed in units of the
    embedded stochastic displacement scale.

        chi = (d_nn / 2) / delta_b

    Large chi:
        well-separated basins.

    Small chi:
        boundaries become dynamically accessible even to
        embedded agents.
    """
    spacing = native_nearest_neighbour_spacing(k, width, height)
    half_spacing = spacing / 2.0
    return float(half_spacing / max(EMBEDDED_NOISE, 1e-12))

def extract_metrics(model, n_steps=N_STEPS):
    """
    Extract final network- and agent-level observables.

    Parameters
    ----------
    model : ExploratorySystemModel
        Completed simulation.

    n_steps : int
        Number of simulation steps actually executed.
        Required for correct transition-rate normalization.

    Returns
    -------
    dict
        Final connectivity, network, transition and
        embedded-agent metrics.
    """
    G = model.transition_graph
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        giant_weak = 0.0
        giant_strong = 0.0
        density = 0.0
        mean_degree = 0.0
        normalized_mean_degree = 0.0
    else:
        weak_components = list(nx.weakly_connected_components(G))
        strong_components = list(nx.strongly_connected_components(G))
        giant_weak = max((len(component) for component in weak_components)) / n_nodes if weak_components else 0.0
        giant_strong = max((len(component) for component in strong_components)) / n_nodes if strong_components else 0.0
        density = nx.density(G) if n_nodes > 1 else 0.0
        degrees = [degree for _, degree in G.degree()]
        mean_degree = float(np.mean(degrees)) if degrees else 0.0
        normalized_mean_degree = mean_degree / (2.0 * (n_nodes - 1)) if n_nodes > 1 else 0.0
    transition_rate = model.total_transitions / max(1, model.n_agents * n_steps)
    embedded_boundary_distance = model.mean_boundary_distance_by_type('embedded')
    embedded_interface_ratio = model.interface_ratio_by_type('embedded')
    return {'giant_weak_component': float(giant_weak), 'giant_strong_component': float(giant_strong), 'graph_density': float(density), 'n_edges': int(n_edges), 'mean_degree': float(mean_degree), 'normalized_mean_degree': float(normalized_mean_degree), 'transition_rate': float(transition_rate), 'embedded_boundary_distance': float(embedded_boundary_distance), 'embedded_interface_ratio': float(embedded_interface_ratio)}

def run_single_simulation(k, p, n_agents, width, height, seed, n_steps=N_STEPS):
    """
    Run one independent simulation realization.
    """
    model = ExploratorySystemModel(n_agents=n_agents, n_attractors=k, exploratory_fraction=float(p), width=float(width), height=float(height), seed=seed)
    prepare_lightweight_model(model)
    for _ in range(n_steps):
        model.step()
        clear_unused_logs(model)
    if len(model.attractors) != k:
        raise RuntimeError(f'Fixed-landscape invariant violated: requested K={k}, observed K={len(model.attractors)}.')
    if getattr(model, 'attractor_creation_log', []):
        raise RuntimeError('Attractor creation occurred during a fixed-landscape robustness run.')
    metrics = extract_metrics(model, n_steps=n_steps)
    del model
    gc.collect()
    return metrics

def run_S2():
    print('\n')
    print('=' * 70)
    print('S2 - Robustness within native separated-basin regime')
    print('=' * 70)
    rows = []
    total_jobs = len(S2_K_VALUES) * len(P_VALUES) * N_RUNS
    job = 0
    for k in S2_K_VALUES:
        spacing = native_nearest_neighbour_spacing(k, BASE_WIDTH, BASE_HEIGHT)
        print(f'\nK={k} | spacing={spacing:.3f}')
        for p in P_VALUES:
            run_metrics = []
            for r in range(N_RUNS):
                job += 1
                seed = S2_BASE_SEED + 10000 * k + int(p * 100000) + r
                metrics = run_single_simulation(k=k, p=p, n_agents=BASE_N_AGENTS, width=BASE_WIDTH, height=BASE_HEIGHT, seed=seed, n_steps=N_STEPS)
                run_metrics.append(metrics)
            df = pd.DataFrame(run_metrics)
            aggregate = aggregate_runs(df)
            row = {'K': k, 'exploratory_fraction': p, 'width': BASE_WIDTH, 'height': BASE_HEIGHT, 'n_agents': BASE_N_AGENTS, 'agents_per_attractor': BASE_N_AGENTS / k, 'attractor_spacing': spacing, **aggregate}
            rows.append(row)
            print(f"K={k:2d} p={p:.3f} | Cw={row['giant_weak_component']:.3f} | D={row['graph_density']:.4f} | E={row['n_edges']:.1f} | {job}/{total_jobs}")
    results = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'S2_native_K_robustness.csv')
    results.to_csv(path, index=False)
    return results

def run_S3():
    print('\n')
    print('=' * 70)
    print('S3 - Constant-spacing / constant-agents-per-attractor scaling')
    print('=' * 70)
    print(f'\nTarget spacing = {TARGET_ATTRACTOR_SPACING:.4f}')
    rows = []
    total_jobs = len(S3_K_VALUES) * len(P_VALUES) * N_RUNS
    job = 0
    for k in S3_K_VALUES:
        width = scaled_domain_size(k)
        height = width
        n_agents = scaled_agent_number(k)
        spacing = native_nearest_neighbour_spacing(k, width, height)
        print(f'\nK={k:2d} | N={n_agents:4d} | W={width:.2f} | spacing={spacing:.3f}')
        for p in P_VALUES:
            run_metrics = []
            for r in range(N_RUNS):
                job += 1
                seed = S3_BASE_SEED + 10000 * k + int(p * 100000) + r
                metrics = run_single_simulation(k=k, p=p, n_agents=n_agents, width=width, height=height, seed=seed, n_steps=N_STEPS)
                run_metrics.append(metrics)
            df = pd.DataFrame(run_metrics)
            aggregate = aggregate_runs(df)
            row = {'K': k, 'exploratory_fraction': p, 'width': width, 'height': height, 'n_agents': n_agents, 'agents_per_attractor': n_agents / k, 'attractor_spacing': spacing, **aggregate}
            rows.append(row)
            print(f"K={k:2d} p={p:.3f} | Cw={row['giant_weak_component']:.3f} | D={row['graph_density']:.4f} | E={row['n_edges']:.1f} | {job}/{total_jobs}")
    results = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'S3_constant_spacing_scaling.csv')
    results.to_csv(path, index=False)
    return results

def run_S3_time_control():
    print('\n')
    print('=' * 70)
    print('S3c - Finite-time control for K=50')
    print('=' * 70)
    k = S3_TIME_K
    width = scaled_domain_size(k)
    height = width
    n_agents = scaled_agent_number(k)
    spacing = native_nearest_neighbour_spacing(k, width, height)
    rows = []
    total_jobs = len(S3_TIME_VALUES) * len(S3_TIME_P_VALUES) * S3_TIME_N_RUNS
    job = 0
    for n_steps in S3_TIME_VALUES:
        print(f'\nT={n_steps} | K={k} | N={n_agents} | W={width:.2f}')
        for p in S3_TIME_P_VALUES:
            run_metrics = []
            for r in range(S3_TIME_N_RUNS):
                job += 1
                seed = S3_TIME_BASE_SEED + 100000 * n_steps + int(p * 100000) + r
                metrics = run_single_simulation(k=k, p=float(p), n_agents=n_agents, width=width, height=height, seed=seed, n_steps=n_steps)
                run_metrics.append(metrics)
            df = pd.DataFrame(run_metrics)
            aggregate = aggregate_runs(df)
            row = {'K': k, 'n_steps': n_steps, 'exploratory_fraction': float(p), 'width': width, 'height': height, 'n_agents': n_agents, 'agents_per_attractor': n_agents / k, 'attractor_spacing': spacing, **aggregate}
            rows.append(row)
            print(f"T={n_steps:4d} | p={p:.3f} | Cw={row['giant_weak_component']:.3f} | {job}/{total_jobs}")
    results = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'S3c_K50_finite_time.csv')
    results.to_csv(path, index=False)
    return results

def run_S4():
    print('\n')
    print('=' * 70)
    print('S4 - Crowding-induced breakdown at fixed domain')
    print('=' * 70)
    rows = []
    total_jobs = len(S4_K_VALUES) * N_RUNS
    job = 0
    for k in S4_K_VALUES:
        spacing = native_nearest_neighbour_spacing(k, BASE_WIDTH, BASE_HEIGHT)
        crowding_ratio = geometric_crowding_ratio(k, BASE_WIDTH, BASE_HEIGHT)
        print(f'\nK={k:2d} | spacing={spacing:.3f} | chi={crowding_ratio:.3f}')
        run_metrics = []
        for r in range(N_RUNS):
            job += 1
            seed = S4_BASE_SEED + 10000 * k + r
            metrics = run_single_simulation(k=k, p=S4_P, n_agents=BASE_N_AGENTS, width=BASE_WIDTH, height=BASE_HEIGHT, seed=seed, n_steps=N_STEPS)
            run_metrics.append(metrics)
        df = pd.DataFrame(run_metrics)
        aggregate = aggregate_runs(df)
        row = {'K': k, 'exploratory_fraction': S4_P, 'width': BASE_WIDTH, 'height': BASE_HEIGHT, 'n_agents': BASE_N_AGENTS, 'agents_per_attractor': BASE_N_AGENTS / k, 'attractor_spacing': spacing, 'crowding_ratio': crowding_ratio, **aggregate}
        rows.append(row)
        print(f"K={k:2d} | Cw={row['giant_weak_component']:.3f} | D={row['graph_density']:.4f} | rate={row['transition_rate']:.5f} | chi={crowding_ratio:.2f} | {job}/{total_jobs}")
    results = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'S4_fixed_domain_crowding.csv')
    results.to_csv(path, index=False)
    return results

def plot_K_curves(results, K_values, filename, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in K_values:
        df = results[results['K'] == k].sort_values('exploratory_fraction').copy()
        x = df['exploratory_fraction'].to_numpy()
        y = df['giant_weak_component'].to_numpy()
        ci = compute_ci(df['giant_weak_component_std'].fillna(0.0).to_numpy(), N_RUNS)
        axes[0].plot(x, y, marker='o', label=f'K={k}')
        axes[0].fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), alpha=0.18)
        y = df['graph_density'].to_numpy()
        ci = compute_ci(df['graph_density_std'].fillna(0.0).to_numpy(), N_RUNS)
        axes[1].plot(x, y, marker='o', label=f'K={k}')
        axes[1].fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), alpha=0.18)
    axes[0].set_title(f'{title_prefix}: giant weak component')
    axes[0].set_xlabel('Exploratory fraction')
    axes[0].set_ylabel('$C_w$')
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_title(f'{title_prefix}: network density')
    axes[1].set_xlabel('Exploratory fraction')
    axes[1].set_ylabel('Density')
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved to:\n{path}')

def plot_S3_network_scaling(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in S3_K_VALUES:
        df = results[results['K'] == k].sort_values('exploratory_fraction').copy()
        x = df['exploratory_fraction'].to_numpy()
        y = df['n_edges'].to_numpy()
        ci = compute_ci(df['n_edges_std'].fillna(0.0).to_numpy(), N_RUNS)
        axes[0].plot(x, y, marker='o', label=f'K={k}')
        axes[0].fill_between(x, np.maximum(y - ci, 0.0), y + ci, alpha=0.18)
        y = df['mean_degree'].to_numpy()
        ci = compute_ci(df['mean_degree_std'].fillna(0.0).to_numpy(), N_RUNS)
        axes[1].plot(x, y, marker='o', label=f'K={k}')
        axes[1].fill_between(x, np.maximum(y - ci, 0.0), y + ci, alpha=0.18)
    axes[0].set_title('Constant-spacing scaling: transition edges')
    axes[0].set_xlabel('Exploratory fraction')
    axes[0].set_ylabel('$|E|$')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_title('Constant-spacing scaling: mean degree')
    axes[1].set_xlabel('Exploratory fraction')
    axes[1].set_ylabel('Mean total degree')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'Supplementary_Figure_S3b_network_scaling.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved to:\n{path}')

def plot_S3_time_control(results):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for n_steps in S3_TIME_VALUES:
        df = results[results['n_steps'] == n_steps].sort_values('exploratory_fraction').copy()
        x = df['exploratory_fraction'].to_numpy()
        y = df['giant_weak_component'].to_numpy()
        ci = compute_ci(df['giant_weak_component_std'].fillna(0.0).to_numpy(), S3_TIME_N_RUNS)
        ax.plot(x, y, marker='o', label=f'T={n_steps}')
        ax.fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), alpha=0.18)
    ax.set_title('Finite-time control for $K=50$')
    ax.set_xlabel('Exploratory fraction')
    ax.set_ylabel('$C_w$')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'Supplementary_Figure_S3c_K50_finite_time.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved to:\n{path}')

def plot_S4(results):
    results = results.sort_values('K').copy()
    x = results['K'].to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    y = results['giant_weak_component'].to_numpy()
    ci = compute_ci(results['giant_weak_component_std'].fillna(0.0).to_numpy(), N_RUNS)
    axes[0].plot(x, y, marker='o')
    axes[0].fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), alpha=0.18)
    axes[0].set_xlabel('Number of attractors K')
    axes[0].set_ylabel('$C_w$ at $p=0$')
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title('Spontaneous connectivity')
    axes[0].grid(True, alpha=0.3)
    y = results['graph_density'].to_numpy()
    ci = compute_ci(results['graph_density_std'].fillna(0.0).to_numpy(), N_RUNS)
    axes[1].plot(x, y, marker='o')
    axes[1].fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), alpha=0.18)
    axes[1].set_xlabel('Number of attractors K')
    axes[1].set_ylabel('Network density at p=0')
    axes[1].set_title('Transition-network density')
    axes[1].grid(True, alpha=0.3)
    y = results['transition_rate'].to_numpy()
    ci = compute_ci(results['transition_rate_std'].fillna(0.0).to_numpy(), N_RUNS)
    axes[2].plot(x, y, marker='o')
    axes[2].fill_between(x, np.maximum(y - ci, 0.0), y + ci, alpha=0.18)
    axes[2].set_xlabel('Number of attractors K')
    axes[2].set_ylabel('Transitions per agent-step')
    axes[2].set_title('Embedded transition activity')
    axes[2].grid(True, alpha=0.3)
    y = results['crowding_ratio'].to_numpy()
    axes[3].plot(x, y, marker='o')
    axes[3].set_xlabel('Number of attractors K')
    axes[3].set_ylabel('$(d_{nn}/2)/\\delta_b$')
    axes[3].set_title('Geometric basin separation')
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'Supplementary_Figure_S4_crowding_breakdown.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved to:\n{path}')

def save_geometry_summary():
    rows = []
    all_k = sorted(set(S2_K_VALUES + S3_K_VALUES + S4_K_VALUES))
    for k in all_k:
        native_spacing = native_nearest_neighbour_spacing(k, BASE_WIDTH, BASE_HEIGHT)
        native_crowding = geometric_crowding_ratio(k, BASE_WIDTH, BASE_HEIGHT)
        scaled_width = scaled_domain_size(k)
        scaled_spacing = native_nearest_neighbour_spacing(k, scaled_width, scaled_width)
        rows.append({'K': k, 'fixed_domain_width': BASE_WIDTH, 'fixed_domain_spacing': native_spacing, 'fixed_domain_crowding_ratio': native_crowding, 'S3_scaled_width': scaled_width, 'S3_scaled_spacing': scaled_spacing, 'S3_n_agents': scaled_agent_number(k), 'S3_agents_per_attractor': AGENTS_PER_ATTRACTOR})
    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'K_geometry_summary.csv')
    df.to_csv(path, index=False)
    print('\nGeometry summary')
    print(df.to_string(index=False))
if __name__ == '__main__':
    set_full_model()
    save_geometry_summary()
    S2_results = run_S2()
    plot_K_curves(results=S2_results, K_values=S2_K_VALUES, filename='Supplementary_Figure_S2_native_K_robustness.png', title_prefix='Native geometry')
    S3_results = run_S3()
    plot_K_curves(results=S3_results, K_values=S3_K_VALUES, filename='Supplementary_Figure_S3_constant_spacing_scaling.png', title_prefix='Constant-spacing scaling')
    plot_S3_network_scaling(S3_results)
    S3_time_results = run_S3_time_control()
    plot_S3_time_control(S3_time_results)
    S4_results = run_S4()
    plot_S4(S4_results)
    print('\nAll K-robustness analyses completed.')
