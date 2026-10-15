import unittest
import numpy as np
from project.environment import Warehouse
from project.algorithms import action_values, value_iteration, learn, evaluate, td_target

class RLTests(unittest.TestCase):
    def test_transition_probabilities_and_absorbing_terminals(self):
        env = Warehouse()
        for state in range(env.n_states):
            for action in range(4):
                transitions = env.transitions(state, action)
                self.assertAlmostEqual(sum(x[0] for x in transitions), 1.)
        goal = env.encode(env.goal)
        for item in env.transitions(goal, 0):
            self.assertEqual(item[1:4], (goal, 0., True))

    def test_wall_collision_and_hazard_termination(self):
        env = Warehouse()
        state = env.encode((0, 0))
        nxt, reward, terminal, collision = env.move(state, 0)
        self.assertEqual(nxt, state)
        self.assertTrue(collision)
        self.assertFalse(terminal)
        nxt, reward, terminal, collision = env.move(env.encode((2, 1)), 1)
        self.assertTrue(terminal)
        self.assertEqual(reward, -10.)
        self.assertEqual(env.decode(nxt), (2, 2))
        self.assertEqual(td_target(-10, True, .95, 1000), -10)

    def test_value_iteration_bellman_residual(self):
        env = Warehouse()
        q, _, residual = value_iteration(env)
        self.assertLess(residual, 1e-9)
        recomputed = action_values(env, q.max(axis=1), .95)
        self.assertLess(np.max(np.abs(q - recomputed)), 1e-8)

    def test_both_learners_solve_small_deterministic_mdp(self):
        config = {"rows":2, "columns":2, "goal":[1,1], "walls":[], "hazards":[],
                  "slip_probability":0, "step_reward":-.1, "collision_reward":-.3,
                  "goal_reward":10., "hazard_reward":-10.}
        env = Warehouse(config)
        for algorithm in ("q_learning", "sarsa"):
            q, _ = learn(env, algorithm, seed=5, episodes=800)
            result = evaluate(env, q, seed=100001, episodes=100)
            self.assertGreaterEqual(np.mean([row["success"] for row in result]), .98)
            self.assertTrue(np.isfinite(q).all())
            np.testing.assert_array_equal(q[env.encode(env.goal)], np.zeros(4))

    def test_unreachable_map_rejected(self):
        config = {"rows":2, "columns":2, "goal":[1,1], "walls":[[0,1],[1,0]],
                  "hazards":[], "slip_probability":0, "step_reward":-.1,
                  "collision_reward":-.3, "goal_reward":10., "hazard_reward":-10.}
        with self.assertRaises(ValueError):
            Warehouse(config)

if __name__ == "__main__":
    unittest.main()
