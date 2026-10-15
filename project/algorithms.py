"""Model-based value iteration and model-free Q-learning / SARSA."""
import numpy as np

def action_values(env, values, discount):
    q = np.zeros((env.n_states, 4))
    for state in env.safe_states:
        for action in range(4):
            q[state, action] = sum(prob * (reward + (0 if terminal else discount * values[nxt]))
                                   for prob, nxt, reward, terminal, _ in env.transitions(state, action))
    return q

def value_iteration(env, discount=.95, tolerance=1e-9, max_iterations=10000):
    values = np.zeros(env.n_states)
    for iteration in range(max_iterations):
        q = action_values(env, values, discount)
        updated = q.max(axis=1)
        residual = float(np.max(np.abs(updated - values)))
        values = updated
        if residual < tolerance:
            return action_values(env, values, discount), iteration + 1, residual
    raise RuntimeError("Value iteration failed to reach the specified Bellman residual.")

def epsilon_greedy(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(4))
    best = np.flatnonzero(np.isclose(q[state], q[state].max()))
    return int(rng.choice(best))

def td_target(reward, terminal, discount, bootstrap):
    return reward if terminal else reward + discount * bootstrap

def learn(env, algorithm, seed, episodes=15000, discount=.95, max_steps=120):
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError("Unknown learning algorithm.")
    rng = np.random.default_rng(seed)
    q, visits = np.zeros((env.n_states, 4)), np.zeros((env.n_states, 4))
    history = []
    for episode in range(episodes):
        epsilon = .05 + .95 * np.exp(-episode / (episodes / 5))
        state = int(rng.choice(env.safe_states))
        action = epsilon_greedy(q, state, epsilon, rng)
        total, discounted, success, hazards = 0., 0., False, False
        for step in range(max_steps):
            nxt, reward, terminal, _ = env.step(state, action, rng)
            next_action = epsilon_greedy(q, nxt, epsilon, rng) if not terminal else 0
            bootstrap = q[nxt].max() if algorithm == "q_learning" else q[nxt, next_action]
            target = td_target(reward, terminal, discount, bootstrap)
            visits[state, action] += 1
            alpha = .6 / (1 + .02 * visits[state, action]) ** .6
            q[state, action] += alpha * (target - q[state, action])
            total += reward
            discounted += discount ** step * reward
            state, action = nxt, next_action
            if terminal:
                success = env.decode(state) == env.goal
                hazards = env.decode(state) in env.hazards
                break
        history.append({"episode": episode + 1, "return": total,
                        "discounted_return": discounted, "steps": step + 1,
                        "success": int(success), "hazard": int(hazards), "epsilon": epsilon})
    return q, history

def evaluate(env, q, seed=100001, episodes=1000, discount=.95, max_steps=120):
    movement = np.random.default_rng(seed)
    starting = np.random.default_rng(seed + 1)
    actions = np.random.default_rng(seed + 2)
    outcomes = []
    for episode in range(episodes):
        state = int(starting.choice(env.safe_states))
        total, discounted, collisions, success, hazard = 0., 0., 0, False, False
        for step in range(max_steps):
            action = int(actions.integers(4)) if q is None else int(q[state].argmax())
            state, reward, terminal, collision = env.step(state, action, movement)
            total += reward
            discounted += discount ** step * reward
            collisions += int(collision)
            if terminal:
                success = env.decode(state) == env.goal
                hazard = env.decode(state) in env.hazards
                break
        outcomes.append({"episode": episode + 1, "return": total, "discounted_return": discounted,
                         "steps": step + 1, "collisions": collisions,
                         "success": int(success), "hazard": int(hazard),
                         "timeout": int(not success and not hazard)})
    return outcomes
