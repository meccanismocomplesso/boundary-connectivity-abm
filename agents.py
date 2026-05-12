from mesa import Agent
from utils import (
    nearest_attractor,
    is_interface,
    get_attractor_by_id,
    clip_position,
    boundary_distance,
)
import config


class SocialAgent(Agent):
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

        # Added for paper logging
        self.history = []
        self.step_distance = 0.0
        self.attractor_switches = 0
        self.prev_pos = pos

    def step(self):
        prev_attr = self.assign_current_attractor()

        pre_attr_obj = get_attractor_by_id(self.model.attractors, prev_attr)
        if pre_attr_obj is not None:
            pre_boundary_distance = boundary_distance(self.pos, pre_attr_obj)
        else:
            pre_boundary_distance = None

        pre_pos = self.pos

        self.move()

        new_attr = self.assign_current_attractor()

        post_attr_obj = get_attractor_by_id(self.model.attractors, new_attr)
        if post_attr_obj is not None:
            post_boundary_distance = boundary_distance(self.pos, post_attr_obj)
        else:
            post_boundary_distance = None

        transition_occurred = (
            prev_attr is not None and new_attr is not None and prev_attr != new_attr
        )

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
            self.model.total_transitions += 1

            if prev_attr in self.model.local_transition_counter:
                self.model.local_transition_counter[prev_attr] += 1
            if new_attr in self.model.local_transition_counter:
                self.model.local_transition_counter[new_attr] += 1

            self.model.register_transition(prev_attr, new_attr)

            self.model.transition_event_log.append(
                {
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
                }
            )


        self.model.step_observation_log.append(
            {
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
            }
        )


        in_interface = is_interface(
            self.pos,
            self.model.attractors,
            self.model.interface_threshold,
        )

        if in_interface:
            self.interface_visits += 1
            self.model.interface_occupancy += 1
            if self.current_attractor in self.model.local_interface_counter:
                self.model.local_interface_counter[self.current_attractor] += 1
            if self.agent_type == "exploratory":
                if self.current_attractor in self.model.local_explorer_interface_counter:
                    self.model.local_explorer_interface_counter[self.current_attractor] += 1

        if self.agent_type == "exploratory":
            if self.current_attractor in self.model.local_explorer_counter:
                self.model.local_explorer_counter[self.current_attractor] += 1

        self.boundary_distance = post_boundary_distance
        self.total_moves += 1



    def assign_current_attractor(self):
        attr_id, _ = nearest_attractor(self.pos, self.model.attractors)
        self.current_attractor = attr_id
        self.visited_attractors.add(attr_id)
        return attr_id

    def move(self):
        if self.agent_type == "embedded":
            self.exploit_move()
        else:
            exploratory_rule = getattr(config, "EXPLORATORY_RULE", "structured")
            if exploratory_rule == "random_walk":
                self.explore_move_random_walk()
            else:
                self.explore_move()

        self.pos = clip_position(self.pos, self.model.width, self.model.height)


    def exploit_move(self):
        landscape_mode = getattr(config, "LANDSCAPE_MODE", "structured")

        if landscape_mode == "flat_control":
            # embedded agents lose the attractor-following gradient
            self.pos = (
                self.pos[0] + self.model.random.uniform(-0.5, 0.5),
                self.pos[1] + self.model.random.uniform(-0.5, 0.5),
            )
            return

        target = get_attractor_by_id(self.model.attractors, self.current_attractor)
        if target is None:
            target = self.model.random.choice(self.model.attractors)

        dx = target.center[0] - self.pos[0]
        dy = target.center[1] - self.pos[1]

        self.pos = (
            self.pos[0] + target.strength * dx + self.model.random.uniform(-0.5, 0.5),
            self.pos[1] + target.strength * dy + self.model.random.uniform(-0.5, 0.5),
        )





    def explore_move(self):
        current = get_attractor_by_id(self.model.attractors, self.current_attractor)
        if current is None:
            current = self.model.random.choice(self.model.attractors)

        novelty_bonus_value = getattr(config, "NOVELTY_BONUS", 1.0)
        interface_bonus_value = getattr(config, "INTERFACE_BONUS_VALUE", 0.5)
        interface_bonus_mode = getattr(config, "INTERFACE_BONUS_MODE", "real")

        candidates = []
        for _ in range(5):
            candidate = (
                self.pos[0]
                + 0.10 * (current.center[0] - self.pos[0])
                + self.model.random.uniform(
                    -self.model.exploratory_move_scale,
                    self.model.exploratory_move_scale,
                ),
                self.pos[1]
                + 0.10 * (current.center[1] - self.pos[1])
                + self.model.random.uniform(
                    -self.model.exploratory_move_scale,
                    self.model.exploratory_move_scale,
                ),
            )

            candidate = clip_position(candidate, self.model.width, self.model.height)
            attr_id, _ = nearest_attractor(candidate, self.model.attractors)

            novelty_bonus = novelty_bonus_value if attr_id not in self.visited_attractors else 0.0

            if interface_bonus_mode == "real":
                interface_bonus = interface_bonus_value if is_interface(
                    candidate,
                    self.model.attractors,
                    self.model.interface_threshold,
                ) else 0.0
            elif interface_bonus_mode == "random":
                interface_bonus = interface_bonus_value if self.model.random.random() < 0.5 else 0.0
            elif interface_bonus_mode == "none":
                interface_bonus = 0.0
            else:
                interface_bonus = 0.0

            random_bonus = self.model.random.random()

            score = novelty_bonus + interface_bonus + random_bonus
            candidates.append((score, candidate))

        self.pos = max(candidates, key=lambda x: x[0])[1]



    def explore_move_random_walk(self):
        self.pos = (
            self.pos[0] + self.model.random.uniform(
                -self.model.exploratory_move_scale,
                self.model.exploratory_move_scale,
            ),
            self.pos[1] + self.model.random.uniform(
                -self.model.exploratory_move_scale,
                self.model.exploratory_move_scale,
            ),
        )

