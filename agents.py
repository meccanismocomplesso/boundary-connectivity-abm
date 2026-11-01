from mesa import Agent
from utils import nearest_attractor, is_interface, get_attractor_by_id, clip_position, basin_boundary_distance
import config


class SocialAgent(Agent):
    """
    Agent used in the fixed-landscape minimal-mechanism experiments.

    Movement hierarchy
    ------------------
    embedded:
        Attractor-following motion + low-amplitude stochastic noise.

    exploratory / random_walk:
        Pure high-mobility random walk.

    exploratory / interface_only:
        Candidate selection without drift; candidates are ranked only by
        interface proximity.

    exploratory / interface_novelty:
        Full minimal model. Candidate selection without drift; candidates are
        ranked by interface proximity and novelty.

    flat_control:
        Attractor-derived forces and bonuses are disabled for movement.
        Embedded and exploratory agents retain only their respective stochastic
        displacement scales. Attractor labels remain available operationally
        for state assignment and transition measurement.
    """

    def __init__(self, model, agent_type, pos):
        super().__init__(model)
        self.agent_type = agent_type
        self.pos = pos
        self.current_attractor = None
        self.visited_attractors = set()
        self.interface_visits = 0
        self.total_moves = 0
        self.transition_count = 0
        self.boundary_distance = None
        self.history = []
        self.step_distance = 0.0
        self.attractor_switches = 0
        self.prev_pos = pos

    def step(self):
        prev_attr = self.assign_current_attractor()
        pre_boundary_distance = basin_boundary_distance(self.pos, self.model.attractors) if prev_attr is not None else None
        pre_pos = self.pos

        self.move()

        new_attr = self.assign_current_attractor()
        post_boundary_distance = basin_boundary_distance(self.pos, self.model.attractors) if new_attr is not None else None
        transition_occurred = prev_attr is not None and new_attr is not None and prev_attr != new_attr

        if pre_boundary_distance is None and post_boundary_distance is None:
            event_boundary_distance = None
        elif pre_boundary_distance is None:
            event_boundary_distance = post_boundary_distance
        elif post_boundary_distance is None:
            event_boundary_distance = pre_boundary_distance
        else:
            event_boundary_distance = min(pre_boundary_distance, post_boundary_distance)

        if transition_occurred:
            self.transition_count += 1
            self.attractor_switches += 1
            self.model.total_transitions += 1

            if prev_attr in self.model.local_transition_counter:
                self.model.local_transition_counter[prev_attr] += 1
            if new_attr in self.model.local_transition_counter:
                self.model.local_transition_counter[new_attr] += 1

            self.model.register_transition(prev_attr, new_attr)

            if self.model.collect_transition_events:
                self.model.transition_event_log.append({
                    "step": self.model.current_step,
                    "agent_id": self.unique_id,
                    "agent_type": self.agent_type,
                    "prev_attractor": prev_attr,
                    "new_attractor": new_attr,
                    "pre_x": pre_pos[0],
                    "pre_y": pre_pos[1],
                    "post_x": self.pos[0],
                    "post_y": self.pos[1],
                    "pre_boundary_distance": pre_boundary_distance,
                    "post_boundary_distance": post_boundary_distance,
                    "event_boundary_distance": event_boundary_distance,
                    "transition_occurred": 1,
                })

        if self.model.collect_step_observations:
            self.model.step_observation_log.append({
                "step": self.model.current_step,
                "agent_id": self.unique_id,
                "agent_type": self.agent_type,
                "pre_x": pre_pos[0],
                "pre_y": pre_pos[1],
                "post_x": self.pos[0],
                "post_y": self.pos[1],
                "pre_boundary_distance": pre_boundary_distance,
                "post_boundary_distance": post_boundary_distance,
                "event_boundary_distance": event_boundary_distance,
                "transition_occurred": int(transition_occurred),
            })

        in_interface = is_interface(self.pos, self.model.attractors, self.model.interface_threshold)

        if in_interface:
            self.interface_visits += 1
            self.model.interface_occupancy += 1

            if self.current_attractor in self.model.local_interface_counter:
                self.model.local_interface_counter[self.current_attractor] += 1

            if self.agent_type == "exploratory" and self.current_attractor in self.model.local_explorer_interface_counter:
                self.model.local_explorer_interface_counter[self.current_attractor] += 1

        if self.agent_type == "exploratory" and self.current_attractor in self.model.local_explorer_counter:
            self.model.local_explorer_counter[self.current_attractor] += 1

        dx = self.pos[0] - pre_pos[0]
        dy = self.pos[1] - pre_pos[1]
        self.step_distance = (dx * dx + dy * dy) ** 0.5
        self.prev_pos = pre_pos
        self.boundary_distance = post_boundary_distance
        self.total_moves += 1

    def assign_current_attractor(self):
        attr_id, _ = nearest_attractor(self.pos, self.model.attractors)
        self.current_attractor = attr_id

        if attr_id is not None:
            self.visited_attractors.add(attr_id)

        return attr_id

    def move(self):
        landscape_mode = getattr(config, "LANDSCAPE_MODE", "structured")

        if self.agent_type == "embedded":
            self.exploit_move()
        elif landscape_mode == "flat_control":
            self.explore_move_random_walk()
        else:
            exploratory_rule = getattr(config, "EXPLORATORY_RULE", "random_walk")

            if exploratory_rule == "random_walk":
                self.explore_move_random_walk()
            elif exploratory_rule == "interface_only":
                self.explore_move_candidate_selection(use_interface=True, use_novelty=False)
            elif exploratory_rule == "interface_novelty":
                self.explore_move_candidate_selection(use_interface=True, use_novelty=True)
            else:
                raise ValueError(
                    f"Unknown EXPLORATORY_RULE: {exploratory_rule!r}. "
                    "Expected 'random_walk', 'interface_only', or 'interface_novelty'."
                )

        self.pos = clip_position(self.pos, self.model.width, self.model.height)

    def exploit_move(self):
        landscape_mode = getattr(config, "LANDSCAPE_MODE", "structured")
        noise_scale = float(getattr(config, "EMBEDDED_MOVE_SCALE", 0.5))

        if landscape_mode == "flat_control":
            self.pos = (
                self.pos[0] + self.model.random.uniform(-noise_scale, noise_scale),
                self.pos[1] + self.model.random.uniform(-noise_scale, noise_scale),
            )
            return

        target = get_attractor_by_id(self.model.attractors, self.current_attractor)

        if target is None:
            target = self.model.random.choice(self.model.attractors)

        dx = target.center[0] - self.pos[0]
        dy = target.center[1] - self.pos[1]

        self.pos = (
            self.pos[0] + target.strength * dx + self.model.random.uniform(-noise_scale, noise_scale),
            self.pos[1] + target.strength * dy + self.model.random.uniform(-noise_scale, noise_scale),
        )

    def explore_move_random_walk(self):
        """Pure high-mobility random walk used by the baseline."""
        scale = float(self.model.exploratory_move_scale)

        self.pos = (
            self.pos[0] + self.model.random.uniform(-scale, scale),
            self.pos[1] + self.model.random.uniform(-scale, scale),
        )

    def explore_move_candidate_selection(self, *, use_interface, use_novelty):
        """
        Generate N_CANDIDATES unbiased stochastic moves and select the
        highest-scoring candidate.
        interface_only: score = interface bonus
        interface_novelty: score = interface bonus + novelty bonus
        Candidate generation contains no attractor-directed drift.
        """
        n_candidates = int(getattr(config, "N_CANDIDATES", 5))

        if n_candidates < 1:
            raise ValueError("N_CANDIDATES must be >= 1.")

        novelty_bonus_value = float(getattr(config, "NOVELTY_BONUS", 1.0))
        interface_bonus_value = float(getattr(config, "INTERFACE_BONUS_VALUE", 0.5))
        interface_bonus_mode = getattr(config, "INTERFACE_BONUS_MODE", "real")
        tie_epsilon = float(getattr(config, "CANDIDATE_TIE_EPSILON", 1e-9))

        candidates = []

        for _ in range(n_candidates):
            candidate = self._generate_exploratory_candidate()
            attr_id, _ = nearest_attractor(candidate, self.model.attractors)
            score = 0.0

            if use_interface:
                if interface_bonus_mode == "real":
                    if is_interface(candidate, self.model.attractors, self.model.interface_threshold):
                        score += interface_bonus_value
                elif interface_bonus_mode == "none":
                    pass
                else:
                    raise ValueError(
                        f"Unknown INTERFACE_BONUS_MODE: {interface_bonus_mode!r}. "
                        "Expected 'real' or 'none'."
                    )

            if use_novelty and attr_id is not None and attr_id not in self.visited_attractors:
                score += novelty_bonus_value

            score += tie_epsilon * self.model.random.random()
            candidates.append((score, candidate))

        self.pos = max(candidates, key=lambda item: item[0])[1]

    def _generate_exploratory_candidate(self):
        """Generate one unbiased stochastic exploratory candidate."""
        scale = float(self.model.exploratory_move_scale)
        candidate = (
            self.pos[0] + self.model.random.uniform(-scale, scale),
            self.pos[1] + self.model.random.uniform(-scale, scale),
        )
        return clip_position(candidate, self.model.width, self.model.height)
