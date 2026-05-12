from pathlib import Path

from model import ExploratorySystemModel


def run_model(
    exploratory_fraction=0.1,
    n_agents=200,
    n_attractors=3,
    steps=300,
    seed=42,
    output_dir=None,
):
    """
    Run a single simulation of the ExploratorySystemModel.

    If output_dir is provided, export the main simulation tables there.
    """
    if not (0.0 <= exploratory_fraction <= 1.0):
        raise ValueError(f"exploratory_fraction must be in [0, 1], got {exploratory_fraction}")

    model = ExploratorySystemModel(
        n_agents=n_agents,
        n_attractors=n_attractors,
        exploratory_fraction=exploratory_fraction,
        seed=seed,
    )

    for _ in range(steps):
        model.step()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_df = model.datacollector.get_model_vars_dataframe()
        agent_df = model.datacollector.get_agent_vars_dataframe()
        history_df = model.export_agent_history_dataframe()
        creation_df = model.export_attractor_creation_dataframe()

        model_df.to_csv(out / "model_vars.csv", index=True)
        agent_df.to_csv(out / "agent_vars.csv", index=True)
        history_df.to_csv(out / "agent_trajectories.csv", index=False)
        creation_df.to_csv(out / "attractor_creation_log.csv", index=False)

    return model



