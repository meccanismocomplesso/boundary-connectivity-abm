from pathlib import Path
import config
from model import ExploratorySystemModel

def run_model(exploratory_fraction=0.1, n_agents=200, n_attractors=3, steps=300, seed=42, width=None, height=None, output_dir=None, collect_transition_events=False, collect_step_observations=False, collect_agent_history=False, collect_datacollector=False):
    """
    Run one fixed-landscape simulation.

    Parameters
    ----------
    exploratory_fraction : float
        Bernoulli probability that an initialized agent is exploratory.
    n_agents : int
        Population size.
    n_attractors : int
        Fixed number of attractor regions.
    steps : int
        Number of discrete simulation steps.
    seed : int
        Random seed.
    width, height : float or None
        Spatial-domain dimensions.  None uses config defaults.
        Explicit values are required by constant-spacing scaling runs.
    output_dir : str/path or None
        If supplied, export all model, agent, event, observation and
        trajectory tables.

    Returns
    -------
    ExploratorySystemModel
        The completed model instance.
    """
    _validate_run_arguments(exploratory_fraction=exploratory_fraction, n_agents=n_agents, n_attractors=n_attractors, steps=steps, width=width, height=height)
    if width is None:
        width = config.GRID_WIDTH
    if height is None:
        height = config.GRID_HEIGHT
    model = ExploratorySystemModel(n_agents=n_agents, n_attractors=n_attractors, exploratory_fraction=exploratory_fraction, width=width, height=height, seed=seed, collect_transition_events=collect_transition_events, collect_step_observations=collect_step_observations, collect_agent_history=collect_agent_history, collect_datacollector=collect_datacollector)
    for _ in range(steps):
        model.step()
    if len(model.attractors) != n_attractors:
        raise RuntimeError(f'Fixed-landscape invariant violated at run completion: requested K={n_attractors}, observed K={len(model.attractors)}.')
    if model.attractor_creation_log:
        raise RuntimeError('Attractor creation log is non-empty in a fixed-landscape experiment.')
    if output_dir is not None:
        _export_run_outputs(model=model, output_dir=output_dir)
    return model

def _validate_run_arguments(*, exploratory_fraction, n_agents, n_attractors, steps, width, height):
    if not 0.0 <= float(exploratory_fraction) <= 1.0:
        raise ValueError(f'exploratory_fraction must be in [0, 1], got {exploratory_fraction}.')
    if int(n_agents) <= 0:
        raise ValueError(f'n_agents must be > 0, got {n_agents}.')
    if int(n_attractors) < 2:
        raise ValueError(f'n_attractors must be >= 2, got {n_attractors}.')
    if int(steps) <= 0:
        raise ValueError(f'steps must be > 0, got {steps}.')
    if width is not None and float(width) <= 1.0:
        raise ValueError(f'width must be > 1, got {width}.')
    if height is not None and float(height) <= 1.0:
        raise ValueError(f'height must be > 1, got {height}.')

def _export_run_outputs(*, model, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if model.datacollector is not None:
        model_df = model.datacollector.get_model_vars_dataframe()
        agent_df = model.datacollector.get_agent_vars_dataframe()
    else:
        model_df = None
        agent_df = None
    history_df = model.export_agent_history_dataframe()
    transition_events_df = model.export_transition_event_dataframe()
    step_observations_df = model.export_step_observation_dataframe()
    creation_df = model.export_attractor_creation_dataframe()
    if model_df is not None:
        model_df.to_csv(out / 'model_vars.csv', index=True)
    if agent_df is not None:
        agent_df.to_csv(out / 'agent_vars.csv', index=True)
    history_df.to_csv(out / 'agent_trajectories.csv', index=False)
    transition_events_df.to_csv(out / 'transition_events.csv', index=False)
    step_observations_df.to_csv(out / 'step_observations.csv', index=False)
    creation_df.to_csv(out / 'attractor_creation_log.csv', index=False)
if __name__ == '__main__':
    run_model(exploratory_fraction=config.EXPLORATORY_RATIO, n_agents=config.N_AGENTS, n_attractors=3, steps=config.MAX_STEPS, seed=config.RANDOM_SEED, width=config.GRID_WIDTH, height=config.GRID_HEIGHT, output_dir='output')