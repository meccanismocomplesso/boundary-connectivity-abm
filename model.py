import math
import networkx as nx
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
import config
from agents import SocialAgent
from utils import Attractor

class ExploratorySystemModel(Model):
    """
    Fixed-landscape agent-based model used in the minimal-mechanism paper.

    The attractor set is created once at initialization and remains unchanged
    for the entire run.  The only endogenous network structure is the directed
    transition graph generated from observed movements between attractor
    regions.

    Attractor centres remain present in the flat control solely to preserve the
    same nearest-attractor state assignment and transition-measurement
    procedure.  In that condition they exert no force on agent motion; this is
    implemented in SocialAgent.
    """

    def __init__(self, n_agents=config.N_AGENTS, n_attractors=3, exploratory_fraction=config.EXPLORATORY_RATIO, width=config.GRID_WIDTH, height=config.GRID_HEIGHT, seed=config.RANDOM_SEED, collect_transition_events=False, collect_step_observations=False, collect_agent_history=False, collect_datacollector=False):
        self._validate_initial_parameters(n_agents=n_agents, n_attractors=n_attractors, exploratory_fraction=exploratory_fraction, width=width, height=height)
        super().__init__(seed=seed)
        self.width = float(width)
        self.height = float(height)
        self.n_agents = int(n_agents)
        self.n_attractors = int(n_attractors)
        self.initial_n_attractors = int(n_attractors)
        self.exploratory_fraction = float(exploratory_fraction)
        self.interface_threshold = float(config.INTERFACE_THRESHOLD)
        self.exploratory_move_scale = float(config.EXPLORATORY_MOVE_SCALE)
        self.total_transitions = 0
        self.interface_occupancy = 0
        self.transition_burst = 0
        self.previous_total_transitions = 0
        self.current_step = 0
        self.attractors = self._create_attractors(self.n_attractors)
        self.reachable_configurations = set()
        self.local_transition_counter = self._zero_attr_counter()
        self.local_interface_counter = self._zero_attr_counter()
        self.local_explorer_counter = self._zero_attr_counter()
        self.local_explorer_interface_counter = self._zero_attr_counter()
        self.collect_transition_events = bool(collect_transition_events)
        self.collect_step_observations = bool(collect_step_observations)
        self.collect_agent_history = bool(collect_agent_history)
        self.collect_datacollector = bool(collect_datacollector)
        self.transition_event_log = []
        self.step_observation_log = []
        self.agent_history_rows = []
        self.attractor_creation_log = []
        self.transition_edges = {}
        self.transition_graph = nx.DiGraph()
        self.transition_graph.add_nodes_from((attr.id for attr in self.attractors))
        self.agents_list = []
        self._create_agents()
        for agent in self.agents_list:
            if agent.current_attractor is not None:
                self.reachable_configurations.add(agent.current_attractor)
        self.datacollector = None
        if self.collect_datacollector:
            self.datacollector = DataCollector(model_reporters={'total_transitions': lambda m: m.total_transitions, 'interface_occupancy': lambda m: m.interface_occupancy, 'reachable_configurations': lambda m: len(m.reachable_configurations), 'num_attractors': lambda m: len(m.attractors), 'avg_unique_attractors_visited': lambda m: m.avg_unique_attractors_visited(), 'exploration_entropy': lambda m: m.exploration_entropy(), 'transition_burst': lambda m: m.compute_transition_burst(), 'embedded_mean_boundary_distance': lambda m: m.mean_boundary_distance_by_type('embedded'), 'exploratory_mean_boundary_distance': lambda m: m.mean_boundary_distance_by_type('exploratory'), 'embedded_interface_ratio': lambda m: m.interface_ratio_by_type('embedded'), 'exploratory_interface_ratio': lambda m: m.interface_ratio_by_type('exploratory'), 'transition_graph_density': lambda m: m.transition_graph_density(), 'transition_graph_n_nodes': lambda m: m.transition_graph.number_of_nodes(), 'transition_graph_n_edges': lambda m: m.transition_graph.number_of_edges(), 'transition_graph_mean_degree': lambda m: m.transition_graph_mean_degree(), 'giant_weak_component_ratio': lambda m: m.giant_weak_component_ratio(), 'giant_strong_component_ratio': lambda m: m.giant_strong_component_ratio(), 'n_weak_components': lambda m: m.n_weak_components(), 'n_strong_components': lambda m: m.n_strong_components(), 'graph_betweenness_max': lambda m: m.graph_betweenness_max(), 'graph_betweenness_mean': lambda m: m.graph_betweenness_mean()}, agent_reporters={'agent_type': lambda a: a.agent_type, 'current_attractor': lambda a: a.current_attractor, 'transition_count': lambda a: a.transition_count, 'interface_visits': lambda a: a.interface_visits, 'visited_attractors_count': lambda a: len(a.visited_attractors), 'boundary_distance': lambda a: a.boundary_distance, 'step_distance': lambda a: getattr(a, 'step_distance', 0.0), 'attractor_switches': lambda a: getattr(a, 'attractor_switches', 0), 'total_moves': lambda a: a.total_moves})

    @staticmethod
    def _validate_initial_parameters(*, n_agents, n_attractors, exploratory_fraction, width, height):
        if int(n_agents) <= 0:
            raise ValueError('n_agents must be > 0.')
        if int(n_attractors) < 2:
            raise ValueError('n_attractors must be >= 2.')
        if not 0.0 <= float(exploratory_fraction) <= 1.0:
            raise ValueError('exploratory_fraction must be in [0, 1].')
        if float(width) <= 1.0 or float(height) <= 1.0:
            raise ValueError('width and height must both be > 1.')
        if float(config.ATTRACTOR_RADIUS) <= 0.0:
            raise ValueError('ATTRACTOR_RADIUS must be > 0.')
        if float(config.INTERFACE_THRESHOLD) < 0.0:
            raise ValueError('INTERFACE_THRESHOLD must be >= 0.')
        if float(config.EMBEDDED_MOVE_SCALE) < 0.0:
            raise ValueError('EMBEDDED_MOVE_SCALE must be >= 0.')
        if float(config.EXPLORATORY_MOVE_SCALE) < 0.0:
            raise ValueError('EXPLORATORY_MOVE_SCALE must be >= 0.')

    def _create_attractors(self, n_attractors):
        """
        Create the complete attractor set once.

        K = 3 uses the paper's explicit triangular layout.
        Other K values use equally spaced centres on a circle.
        """
        if n_attractors == 3:
            centers = [(self.width * 0.25, self.height * 0.3), (self.width * 0.75, self.height * 0.3), (self.width * 0.5, self.height * 0.75)]
        else:
            centers = []
            cx = self.width * 0.5
            cy = self.height * 0.5
            layout_radius = min(self.width, self.height) * 0.3
            for i in range(n_attractors):
                angle = 2.0 * math.pi * i / n_attractors
                centers.append((cx + layout_radius * math.cos(angle), cy + layout_radius * math.sin(angle)))
        return [Attractor(id=i, center=center, radius=float(config.ATTRACTOR_RADIUS), strength=float(config.ATTRACTOR_STRENGTH), emergent=False) for i, center in enumerate(centers)]

    def _zero_attr_counter(self):
        return {attr.id: 0 for attr in self.attractors}

    def _create_agents(self):
        """
        Place agents uniformly inside attractor-centred bounding
        squares and assign exploratory type independently as
        Bernoulli(p), matching the paper protocol.
        """
        bernoulli_assignment = bool(getattr(config, 'BERNOULLI_EXPLORER_ASSIGNMENT', True))
        if not bernoulli_assignment:
            raise ValueError('The fixed-landscape paper model requires BERNOULLI_EXPLORER_ASSIGNMENT = True.')
        for _ in range(self.n_agents):
            is_exploratory = self.random.random() < self.exploratory_fraction
            agent_type = 'exploratory' if is_exploratory else 'embedded'
            attr = self.random.choice(self.attractors)
            x = self.random.uniform(attr.center[0] - attr.radius, attr.center[0] + attr.radius)
            y = self.random.uniform(attr.center[1] - attr.radius, attr.center[1] + attr.radius)
            x = min(max(x, 0.0), self.width - 1.0)
            y = min(max(y, 0.0), self.height - 1.0)
            agent = SocialAgent(self, agent_type, (x, y))
            agent.assign_current_attractor()
            self.agents_list.append(agent)

    def step(self):
        self.current_step += 1
        self.interface_occupancy = 0
        self.local_transition_counter = self._zero_attr_counter()
        self.local_interface_counter = self._zero_attr_counter()
        self.local_explorer_counter = self._zero_attr_counter()
        self.local_explorer_interface_counter = self._zero_attr_counter()
        prev_total = self.total_transitions
        self.random.shuffle(self.agents_list)
        for agent in self.agents_list:
            agent.step()
            if agent.current_attractor is not None:
                self.reachable_configurations.add(agent.current_attractor)
        if len(self.attractors) != self.initial_n_attractors:
            raise RuntimeError(f'Fixed-landscape invariant violated: expected {self.initial_n_attractors} attractors, found {len(self.attractors)}.')
        self.transition_burst = self.total_transitions - prev_total
        self.previous_total_transitions = self.total_transitions
        if self.collect_agent_history:
            self._record_agent_history()
        if self.datacollector is not None:
            self.datacollector.collect(self)

    def _record_agent_history(self):
        """
        Record one compact trajectory row per agent per step.

        This replaces the old dependency on agent.history and makes
        run.py's agent_trajectories.csv export informative again.
        """
        for agent in self.agents_list:
            self.agent_history_rows.append({'step': self.current_step, 'agent_id': agent.unique_id, 'agent_type': agent.agent_type, 'x': agent.pos[0], 'y': agent.pos[1], 'current_attractor': agent.current_attractor, 'boundary_distance': agent.boundary_distance, 'transition_count': agent.transition_count, 'step_distance': getattr(agent, 'step_distance', 0.0), 'attractor_switches': getattr(agent, 'attractor_switches', 0), 'interface_visits': agent.interface_visits, 'total_moves': agent.total_moves})

    def register_transition(self, source_attr, target_attr, agent=None):
        """
        Register one observed transition between two distinct
        fixed attractor states.
        """
        if source_attr is None or target_attr is None:
            return
        if source_attr == target_attr:
            return
        valid_ids = {attr.id for attr in self.attractors}
        if source_attr not in valid_ids or target_attr not in valid_ids:
            raise RuntimeError('Transition references an unknown attractor ID.')
        edge = (source_attr, target_attr)
        self.transition_edges[edge] = self.transition_edges.get(edge, 0) + 1
        if self.transition_graph.has_edge(source_attr, target_attr):
            self.transition_graph[source_attr][target_attr]['weight'] += 1
        else:
            self.transition_graph.add_edge(source_attr, target_attr, weight=1)

    def avg_unique_attractors_visited(self):
        if not self.agents_list:
            return 0.0
        return float(sum((len(a.visited_attractors) for a in self.agents_list)) / len(self.agents_list))

    def exploration_entropy(self):
        counts = {}
        total = 0
        for agent in self.agents_list:
            if agent.agent_type == 'exploratory' and agent.current_attractor is not None:
                aid = agent.current_attractor
                counts[aid] = counts.get(aid, 0) + 1
                total += 1
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p + 1e-12)
        return float(entropy)

    def compute_transition_burst(self):
        return float(self.transition_burst)

    def mean_boundary_distance_by_type(self, agent_type):
        vals = [a.boundary_distance for a in self.agents_list if a.agent_type == agent_type and a.boundary_distance is not None]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def interface_ratio_by_type(self, agent_type):
        agents = [a for a in self.agents_list if a.agent_type == agent_type]
        if not agents:
            return 0.0
        ratios = [a.interface_visits / a.total_moves for a in agents if a.total_moves > 0]
        if not ratios:
            return 0.0
        return float(sum(ratios) / len(ratios))

    def transition_graph_density(self):
        n = self.transition_graph.number_of_nodes()
        if n <= 1:
            return 0.0
        return float(nx.density(self.transition_graph))

    def transition_graph_mean_degree(self):
        """
        Mean total degree of the directed graph:
        mean(in-degree + out-degree) = 2E / |V|.
        """
        n = self.transition_graph.number_of_nodes()
        if n == 0:
            return 0.0
        return float(sum((degree for _, degree in self.transition_graph.degree())) / n)

    def giant_weak_component_ratio(self):
        G = self.transition_graph
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
        components = list(nx.weakly_connected_components(G))
        if not components:
            return 0.0
        return float(max((len(c) for c in components)) / n)

    def giant_strong_component_ratio(self):
        G = self.transition_graph
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
        components = list(nx.strongly_connected_components(G))
        if not components:
            return 0.0
        return float(max((len(c) for c in components)) / n)

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
        bc = nx.betweenness_centrality(G)
        if not bc:
            return 0.0
        return float(max(bc.values()))

    def graph_betweenness_mean(self):
        G = self.transition_graph
        if G.number_of_nodes() <= 2:
            return 0.0
        bc = nx.betweenness_centrality(G)
        if not bc:
            return 0.0
        return float(sum(bc.values()) / len(bc))

    def export_agent_history_dataframe(self):
        columns = ['step', 'agent_id', 'agent_type', 'x', 'y', 'current_attractor', 'boundary_distance', 'transition_count', 'step_distance', 'attractor_switches', 'interface_visits', 'total_moves']
        if not self.agent_history_rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.agent_history_rows, columns=columns)

    def export_attractor_creation_dataframe(self):
        """
        Backward-compatible export.

        The fixed-landscape model never creates attractors, so
        this table is always empty by design.
        """
        return pd.DataFrame(columns=['step', 'parent_id', 'new_attractor_id', 'creator_agent_id', 'creator_agent_type'])

    def export_transition_event_dataframe(self):
        columns = ['step', 'agent_id', 'agent_type', 'prev_attractor', 'new_attractor', 'pre_x', 'pre_y', 'post_x', 'post_y', 'pre_boundary_distance', 'post_boundary_distance', 'event_boundary_distance', 'transition_occurred']
        if not self.transition_event_log:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.transition_event_log, columns=columns)

    def export_step_observation_dataframe(self):
        columns = ['step', 'agent_id', 'agent_type', 'pre_x', 'pre_y', 'post_x', 'post_y', 'pre_boundary_distance', 'post_boundary_distance', 'event_boundary_distance', 'transition_occurred']
        if not self.step_observation_log:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.step_observation_log, columns=columns)