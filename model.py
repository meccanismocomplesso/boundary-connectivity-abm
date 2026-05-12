import math
import pandas as pd
import networkx as nx

from mesa import Model
from mesa.datacollection import DataCollector

import config
from agents import SocialAgent
from utils import Attractor, get_attractor_by_id


class ExploratorySystemModel(Model):
    def __init__(
        self,
        n_agents=40,
        n_attractors=3,
        exploratory_fraction=0.1,
        width=config.GRID_WIDTH,
        height=config.GRID_HEIGHT,
        seed=config.RANDOM_SEED,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height

        self.n_agents = n_agents
        self.n_attractors = n_attractors
        self.exploratory_fraction = exploratory_fraction

        self.interface_threshold = config.INTERFACE_THRESHOLD
        self.exploratory_move_scale = config.EXPLORATORY_MOVE_SCALE

        self.total_transitions = 0
        self.interface_occupancy = 0
        self.transition_burst = 0
        self.previous_total_transitions = 0

        self.current_step = 0

        self.attractors = self._create_attractors(n_attractors)
        self.reachable_configurations = set(attr.id for attr in self.attractors)

        self.attractor_tension = {attr.id: 0.0 for attr in self.attractors}

        # Local counters populated by agents during each step
        self.local_transition_counter = {attr.id: 0 for attr in self.attractors}
        self.local_interface_counter = {attr.id: 0 for attr in self.attractors}
        self.local_explorer_counter = {attr.id: 0 for attr in self.attractors}
        self.local_explorer_interface_counter = {attr.id: 0 for attr in self.attractors}

        # Logs for paper figures
        self.attractor_creation_log = []
        self.agent_history_rows = []

        # Transition event 
        self.transition_event_log = []
        self.step_observation_log = []


        # Transition network
        self.transition_edges = {}
        self.transition_graph = nx.DiGraph()
        for attr in self.attractors:
            self.transition_graph.add_node(attr.id)

        self.agents_list = []
        self._create_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "total_transitions": lambda m: m.total_transitions,
                "interface_occupancy": lambda m: m.interface_occupancy,
                "reachable_configurations": lambda m: len(m.reachable_configurations),
                "num_attractors": lambda m: len(m.attractors),
                "avg_unique_attractors_visited": lambda m: m.avg_unique_attractors_visited(),
                "max_local_tension": lambda m: m.max_local_tension(),
                "exploration_entropy": lambda m: m.exploration_entropy(),
                "transition_burst": lambda m: m.compute_transition_burst(),
                "embedded_mean_boundary_distance": lambda m: m.mean_boundary_distance_by_type("embedded"),
                "exploratory_mean_boundary_distance": lambda m: m.mean_boundary_distance_by_type("exploratory"),
                "embedded_interface_ratio": lambda m: m.interface_ratio_by_type("embedded"),
                "exploratory_interface_ratio": lambda m: m.interface_ratio_by_type("exploratory"),
                "transition_graph_density": lambda m: m.transition_graph_density(),
                "transition_graph_n_nodes": lambda m: m.transition_graph.number_of_nodes(),
                "transition_graph_n_edges": lambda m: m.transition_graph.number_of_edges(),
                "transition_graph_mean_degree": lambda m: m.transition_graph_mean_degree(),
                "giant_weak_component_ratio": lambda m: m.giant_weak_component_ratio(),
                "giant_strong_component_ratio": lambda m: m.giant_strong_component_ratio(),
                "n_weak_components": lambda m: m.n_weak_components(),
                "n_strong_components": lambda m: m.n_strong_components(),
                "graph_betweenness_max": lambda m: m.graph_betweenness_max(),
                "graph_betweenness_mean": lambda m: m.graph_betweenness_mean(),
            },
            agent_reporters={
                "agent_type": lambda a: a.agent_type,
                "current_attractor": lambda a: a.current_attractor,
                "transition_count": lambda a: a.transition_count,
                "interface_visits": lambda a: a.interface_visits,
                "visited_attractors_count": lambda a: len(a.visited_attractors),
                "boundary_distance": lambda a: a.boundary_distance,
                "step_distance": lambda a: getattr(a, "step_distance", 0.0),
                "attractor_switches": lambda a: getattr(a, "attractor_switches", 0),
                "total_moves": lambda a: a.total_moves,
            },
        )





    # ---------------------------------------------------
    # ATTRACTORS
    # ---------------------------------------------------

    def _create_attractors(self, n_attractors):
        if n_attractors == 3:
            centers = [
                (self.width * 0.25, self.height * 0.30),
                (self.width * 0.75, self.height * 0.30),
                (self.width * 0.50, self.height * 0.75),
            ]
        else:
            centers = []
            cx, cy = self.width * 0.5, self.height * 0.5
            radius = min(self.width, self.height) * 0.30
            for i in range(n_attractors):
                angle = 2 * math.pi * i / n_attractors
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                centers.append((x, y))

        attractors = []
        for i, center in enumerate(centers):
            attractors.append(
                Attractor(
                    id=i,
                    center=center,
                    radius=config.ATTRACTOR_RADIUS,
                    strength=config.ATTRACTOR_STRENGTH,
                    emergent=False,
                )
            )

        return attractors

    # ---------------------------------------------------
    # AGENTS
    # ---------------------------------------------------

    def _create_agents(self):
        if getattr(config, "BERNOULLI_EXPLORER_ASSIGNMENT", False):
            for _ in range(self.n_agents):
                is_exploratory = self.random.random() < self.exploratory_fraction
                agent_type = "exploratory" if is_exploratory else "embedded"

                attr = self.random.choice(self.attractors)
                x = self.random.uniform(attr.center[0] - attr.radius, attr.center[0] + attr.radius)
                y = self.random.uniform(attr.center[1] - attr.radius, attr.center[1] + attr.radius)

                agent = SocialAgent(self, agent_type, (x, y))
                agent.assign_current_attractor()
                self.agents_list.append(agent)
        else:
            n_exploratory = int(self.n_agents * self.exploratory_fraction)

            for i in range(self.n_agents):
                agent_type = "exploratory" if i < n_exploratory else "embedded"

                attr = self.random.choice(self.attractors)
                x = self.random.uniform(attr.center[0] - attr.radius, attr.center[0] + attr.radius)
                y = self.random.uniform(attr.center[1] - attr.radius, attr.center[1] + attr.radius)

                agent = SocialAgent(self, agent_type, (x, y))
                agent.assign_current_attractor()
                self.agents_list.append(agent)

    # ---------------------------------------------------
    # STEP
    # ---------------------------------------------------

    def step(self):
        self.current_step += 1

        self.interface_occupancy = 0

        # Reset local counters at each step
        self.local_transition_counter = {attr.id: 0 for attr in self.attractors}
        self.local_interface_counter = {attr.id: 0 for attr in self.attractors}
        self.local_explorer_counter = {attr.id: 0 for attr in self.attractors}
        self.local_explorer_interface_counter = {attr.id: 0 for attr in self.attractors}

        prev_total = self.total_transitions

        self.random.shuffle(self.agents_list)

        for agent in self.agents_list:
            agent.step()
            self.reachable_configurations.update(agent.visited_attractors)

        self._update_attractor_tension()
        self._check_local_transitions()

        self.transition_burst = self.total_transitions - prev_total
        self.previous_total_transitions = self.total_transitions

        self.datacollector.collect(self)

        # Persist last step of each agent for export
        for agent in self.agents_list:
            if hasattr(agent, "history") and agent.history:
                self.agent_history_rows.append(agent.history[-1])

    # ---------------------------------------------------
    # TENSION DYNAMICS
    # ---------------------------------------------------

    def _update_attractor_tension(self):
        base_tension = getattr(config, "LOCAL_TENSION_BASE", 0.0)
        density_weight = getattr(config, "LOCAL_DENSITY_WEIGHT", 0.0)
        explorer_weight = getattr(config, "LOCAL_EXPLORER_WEIGHT", 0.0)
        interface_weight = getattr(config, "LOCAL_INTERFACE_WEIGHT", 0.0)
        transition_weight = getattr(config, "LOCAL_TRANSITION_WEIGHT", 0.0)
        tension_decay = getattr(config, "TENSION_DECAY", 0.0)

        # Agent counts per attractor
        counts = {attr.id: 0 for attr in self.attractors}
        for agent in self.agents_list:
            aid = agent.current_attractor
            if aid in counts:
                counts[aid] += 1

        for attr in self.attractors:
            aid = attr.id

            local_density = counts[aid] / max(1, self.n_agents)
            local_explorer_ratio = self.local_explorer_counter[aid] / max(1, self.n_agents)
            local_interface_ratio = self.local_interface_counter[aid] / max(1, self.n_agents)
            local_transitions = self.local_transition_counter[aid]

            # Decay existing tension
            self.attractor_tension[aid] *= max(0.0, 1.0 - tension_decay)

            # Add new tension contributions
            self.attractor_tension[aid] += (
                base_tension
                + density_weight * local_density
                + explorer_weight * local_explorer_ratio
                + interface_weight * local_interface_ratio
                + transition_weight * local_transitions
            )

    # ---------------------------------------------------
    # EMERGENCE CHECK
    # ---------------------------------------------------

    def _check_local_transitions(self):
        require_explorers = getattr(config, "REQUIRE_EXPLORERS_FOR_EMERGENCE", False)

        triggered = []
        for aid, tension in self.attractor_tension.items():
            if tension < config.TRANSITION_THRESHOLD:
                continue

            if require_explorers:
                has_explorer = self.local_explorer_counter.get(aid, 0) > 0
                if not has_explorer:
                    continue

            triggered.append(aid)

        for aid in triggered:
            creator_agent = self._infer_creator_agent(aid)
            self._create_emergent_attractor(parent_id=aid, creator_agent=creator_agent)
            self.attractor_tension[aid] = 0.0

    def _infer_creator_agent(self, attractor_id):
        """
        Heuristic creator attribution for logging:
        prefer exploratory agents in the triggering attractor;
        otherwise fallback to any agent in that attractor.
        """
        local_agents = [a for a in self.agents_list if a.current_attractor == attractor_id]

        if not local_agents:
            return None

        exploratory_agents = [a for a in local_agents if a.agent_type == "exploratory"]
        if exploratory_agents:
            return exploratory_agents[0]

        return local_agents[0]

    # ---------------------------------------------------
    # CREATE EMERGENT ATTRACTOR
    # ---------------------------------------------------

    def _create_emergent_attractor(self, parent_id, creator_agent=None):
        if len(self.attractors) >= config.MAX_ATTRACTORS:
            return

        parent = get_attractor_by_id(self.attractors, parent_id)
        if parent is None:
            return

        new_id = max(attr.id for attr in self.attractors) + 1

        x = parent.center[0] + self.random.uniform(-12, 12)
        y = parent.center[1] + self.random.uniform(-12, 12)

        x = min(max(x, 0), self.width - 1)
        y = min(max(y, 0), self.height - 1)

        new_attr = Attractor(
            id=new_id,
            center=(x, y),
            radius=config.EMERGENT_ATTRACTOR_RADIUS,
            strength=config.EMERGENT_ATTRACTOR_STRENGTH,
            emergent=True,
        )

        self.attractors.append(new_attr)
        self.attractor_tension[new_id] = 0.0

        # Extend local counters for the new attractor
        self.local_transition_counter[new_id] = 0
        self.local_interface_counter[new_id] = 0
        self.local_explorer_counter[new_id] = 0
        self.local_explorer_interface_counter[new_id] = 0

        self.reachable_configurations.add(new_id)
        self.transition_graph.add_node(new_id)

        self.attractor_creation_log.append(
            {
                "step": self.current_step,
                "parent_id": parent_id,
                "new_attractor_id": new_id,
                "creator_agent_id": creator_agent.unique_id if creator_agent is not None else None,
                "creator_agent_type": creator_agent.agent_type if creator_agent is not None else None,
            }
        )

    # ---------------------------------------------------
    # TRANSITION GRAPH
    # ---------------------------------------------------

    def register_transition(self, source_attr, target_attr, agent=None):
        edge = (source_attr, target_attr)
        self.transition_edges[edge] = self.transition_edges.get(edge, 0) + 1

        if self.transition_graph.has_edge(source_attr, target_attr):
            self.transition_graph[source_attr][target_attr]["weight"] += 1
        else:
            self.transition_graph.add_edge(source_attr, target_attr, weight=1)

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    def avg_unique_attractors_visited(self):
        if not self.agents_list:
            return 0.0
        return sum(len(a.visited_attractors) for a in self.agents_list) / len(self.agents_list)

    def max_local_tension(self):
        return max(self.attractor_tension.values()) if self.attractor_tension else 0.0

    def exploration_entropy(self):
        counts = {}
        total = 0

        for agent in self.agents_list:
            if agent.agent_type == "exploratory" and agent.current_attractor is not None:
                aid = agent.current_attractor
                counts[aid] = counts.get(aid, 0) + 1
                total += 1

        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p + 1e-12)

        return entropy

    def compute_transition_burst(self):
        return self.transition_burst

    def giant_weak_component_ratio(self):
        G = self.transition_graph
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
        comps = list(nx.weakly_connected_components(G))
        if not comps:
            return 0.0
        largest = max(len(c) for c in comps)
        return float(largest / n)


    def giant_strong_component_ratio(self):
        G = self.transition_graph
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
        comps = list(nx.strongly_connected_components(G))
        if not comps:
            return 0.0
        largest = max(len(c) for c in comps)
        return float(largest / n)


    def n_weak_components(self):
        G = self.transition_graph
        if G.number_of_nodes() == 0:
            return 0
        return int(nx.number_weakly_connected_components(G))


    def n_strong_components(self):
        G = self.transition_graph
        if G.number_of_nodes() == 0:
            return 0
        return int(nx.number_strongly_connected_components(G))


    def graph_betweenness_max(self):
        G = self.transition_graph
        if G.number_of_nodes() <= 2:
            return 0.0
        try:
            bc = nx.betweenness_centrality(G)
            if not bc:
                return 0.0
            return float(max(bc.values()))
        except Exception:
            return 0.0


    def graph_betweenness_mean(self):
        G = self.transition_graph
        if G.number_of_nodes() <= 2:
            return 0.0
        try:
            bc = nx.betweenness_centrality(G)
            if not bc:
                return 0.0
            return float(sum(bc.values()) / len(bc))
        except Exception:
            return 0.0



    # ---------------------------------------------------
    # EXPORT HELPERS
    # ---------------------------------------------------

    def mean_boundary_distance_by_type(self, agent_type):
        vals = [
            a.boundary_distance
            for a in self.agents_list
            if a.agent_type == agent_type and a.boundary_distance is not None
        ]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))


    def interface_ratio_by_type(self, agent_type):
        agents = [a for a in self.agents_list if a.agent_type == agent_type]
        if not agents:
            return 0.0

        ratios = []
        for a in agents:
            if a.total_moves > 0:
                ratios.append(a.interface_visits / a.total_moves)

        if not ratios:
            return 0.0
        return float(sum(ratios) / len(ratios))


    def transition_graph_density(self):
        n = self.transition_graph.number_of_nodes()
        if n <= 1:
            return 0.0
        return float(nx.density(self.transition_graph))


    def transition_graph_mean_degree(self):
        n = self.transition_graph.number_of_nodes()
        if n == 0:
            return 0.0
        degs = [d for _, d in self.transition_graph.degree()]
        if not degs:
            return 0.0
        return float(sum(degs) / len(degs))


    def export_agent_history_dataframe(self):
        if not self.agent_history_rows:
            return pd.DataFrame(
                columns=[
                    "step",
                    "agent_id",
                    "agent_type",
                    "x",
                    "y",
                    "current_attractor",
                    "boundary_distance",
                    "transition_count",
                    "step_distance",
                    "attractor_switches",
                    "interface_visits",
                    "total_moves",
                ]
            )
        return pd.DataFrame(self.agent_history_rows)

    def export_attractor_creation_dataframe(self):
        if not self.attractor_creation_log:
            return pd.DataFrame(
                columns=[
                    "step",
                    "parent_id",
                    "new_attractor_id",
                    "creator_agent_id",
                    "creator_agent_type",
                ]
            )
        return pd.DataFrame(self.attractor_creation_log)


    def export_transition_event_dataframe(self):
        if not self.transition_event_log:
            return pd.DataFrame(
                columns=[
                    "step",
                    "agent_id",
                    "agent_type",
                    "prev_attractor",
                    "new_attractor",
                    "pre_x",
                    "pre_y",
                    "post_x",
                    "post_y",
                    "pre_boundary_distance",
                    "post_boundary_distance",
                    "event_boundary_distance",
                    "transition_occurred",
                ]
            )
        return pd.DataFrame(self.transition_event_log)


    def export_step_observation_dataframe(self):
        if not self.step_observation_log:
            return pd.DataFrame(
                columns=[
                    "step",
                    "agent_id",
                    "agent_type",
                    "pre_x",
                    "pre_y",
                    "post_x",
                    "post_y",
                    "pre_boundary_distance",
                    "post_boundary_distance",
                    "event_boundary_distance",
                    "transition_occurred",
                ]

            )
        return pd.DataFrame(self.step_observation_log)



