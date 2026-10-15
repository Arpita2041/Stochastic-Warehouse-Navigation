# Methodology

The environment is an original finite simulated MDP, not recorded warehouse
robot data. The default grid is 8 by 8 with explicitly listed walls and hazards.
The state is the agent's cell. Actions intend up, right, down, or left.
With probability 0.88 the intended move executes, and with probability 0.06
each the agent moves in the adjacent perpendicular direction.

An ordinary move returns -0.1, a wall or boundary collision returns -0.3 and
leaves the agent in place, the goal returns +10 and terminates, and a hazard
returns -10 and terminates. Terminal transitions thereafter have zero reward.
All nonterminal safe starting cells must have a route to the goal without
entering hazards. The map validator checks this condition.

The discounted objective uses gamma 0.95. Value iteration uses the complete
transition and reward model and stops at Bellman residual below 1e-9.
It is a model-based reference, not a model-free learner with the same access.
Q-learning and SARSA use only sampled transitions.

The temporal-difference target for Q-learning is reward plus gamma times the
largest next-state Q value. For SARSA it uses the next action selected by the
current epsilon-greedy policy. Terminal targets contain reward only.
The learning rate is 0.6 divided by (1 + 0.02 times the state-action visit count)
to the power 0.6. Epsilon is 0.05 + 0.95 times exp(-episode / (episodes / 5)).
Greedy action ties during training are chosen randomly.

Three independent training seeds are used for each learner. Full mode trains
15,000 episodes per learner per seed and evaluates 1,000 episodes per seed.
Quick mode uses 2,500 and 250 respectively. Hyperparameters are fixed rather
than chosen from evaluation success. Training starts are sampled from safe
nonterminal cells. An episode is capped at 120 steps.

The cap is a truncation for the underlying discounted MDP, so a nonterminal
last training transition still bootstraps. Evaluation reports timeouts
separately. If the time cap were part of the task definition, remaining time
would belong in the state and its terminal handling would need to change.

Evaluation uses greedy policies, separate seed values, and distinct random
generators for starts, movement, and the random-policy actions. All models
receive the same evaluation seed for a given repetition, but their trajectories
can differ. Success, hazard rate, timeout rate, collisions, steps, undiscounted
return, and discounted return are measured. The planner optimizes discounted
return, not success alone. Standard deviations across the three repetitions
include training variation for the learners and finite-episode randomness.

Policy arrows indicate intended actions; slip means actual trajectories can
deviate. Replay produces the full transition trace and an optional figure.
The environment configuration and its fingerprint are stored with policies.
A learned tabular policy applies to that map only. A changed map requires
retraining; there is no claim of transfer to unseen layouts or physical robots.
