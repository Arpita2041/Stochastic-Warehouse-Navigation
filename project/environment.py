"""A finite stochastic MDP with terminal goal and hazard cells."""
import hashlib
import json
import numpy as np

DEFAULT = {
    "rows": 8, "columns": 8, "goal": [7, 7],
    "walls": [[1, 1], [1, 2], [1, 5], [2, 5], [3, 1], [3, 3],
              [3, 4], [4, 3], [5, 1], [5, 5], [6, 5]],
    "hazards": [[2, 2], [4, 5], [6, 2]],
    "slip_probability": .12, "step_reward": -.1,
    "collision_reward": -.3, "goal_reward": 10., "hazard_reward": -10.
}
ACTION_NAMES = ["up", "right", "down", "left"]
OFFSETS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

class Warehouse:
    def __init__(self, config=None):
        self.config = dict(DEFAULT if config is None else config)
        self.rows, self.columns = int(self.config["rows"]), int(self.config["columns"])
        if min(self.rows, self.columns) < 2 or self.rows * self.columns > 2500:
            raise ValueError("Use a grid of at least 2 by 2 and at most 2500 cells.")
        self.walls = {tuple(cell) for cell in self.config["walls"]}
        self.hazards = {tuple(cell) for cell in self.config["hazards"]}
        self.goal = tuple(self.config["goal"])
        if (self.walls & self.hazards or self.goal in self.walls or self.goal in self.hazards):
            raise ValueError("Walls, hazards, and goal must be disjoint.")
        for cell in self.walls | self.hazards | {self.goal}:
            if not (0 <= cell[0] < self.rows and 0 <= cell[1] < self.columns):
                raise ValueError("A configured cell is outside the grid.")
        self.slip = float(self.config["slip_probability"])
        if not 0 <= self.slip < 1:
            raise ValueError("slip_probability must lie in [0, 1).")
        self.n_states = self.rows * self.columns
        self.safe_states = np.array([self.encode((r, c)) for r in range(self.rows)
                                    for c in range(self.columns)
                                    if (r, c) not in self.walls | self.hazards | {self.goal}])
        if len(self.safe_states) == 0:
            raise ValueError("No nonterminal safe cells remain.")
        reachable, frontier = {self.goal}, [self.goal]
        while frontier:
            r, c = frontier.pop()
            for dr, dc in OFFSETS:
                cell = (r + dr, c + dc)
                if (0 <= cell[0] < self.rows and 0 <= cell[1] < self.columns
                    and cell not in self.walls | self.hazards and cell not in reachable):
                    reachable.add(cell)
                    frontier.append(cell)
        if any(self.decode(s) not in reachable for s in self.safe_states):
            raise ValueError("Every safe starting cell must have a route to the goal.")

    def encode(self, cell):
        return int(cell[0] * self.columns + cell[1])

    def decode(self, state):
        return int(state) // self.columns, int(state) % self.columns

    def terminal(self, state):
        return self.decode(state) in self.hazards | {self.goal}

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    def move(self, state, action):
        if self.terminal(state):
            return state, 0., True, False
        r, c = self.decode(state)
        dr, dc = OFFSETS[action]
        cell = (r + dr, c + dc)
        collision = (not (0 <= cell[0] < self.rows and 0 <= cell[1] < self.columns)
                     or cell in self.walls)
        if collision:
            return state, float(self.config["collision_reward"]), False, True
        next_state = self.encode(cell)
        if cell == self.goal:
            return next_state, float(self.config["goal_reward"]), True, False
        if cell in self.hazards:
            return next_state, float(self.config["hazard_reward"]), True, False
        return next_state, float(self.config["step_reward"]), False, False

    def transitions(self, state, action):
        if not 0 <= action < 4:
            raise ValueError("Action must be an integer from 0 to 3.")
        return [(probability, *self.move(state, actual))
                for actual, probability in ((action, 1 - self.slip),
                                             ((action - 1) % 4, self.slip / 2),
                                             ((action + 1) % 4, self.slip / 2))]

    def step(self, state, action, rng):
        transitions = self.transitions(state, action)
        choice = rng.choice(3, p=[item[0] for item in transitions])
        return transitions[choice][1:]
