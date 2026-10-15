# Environment card

Data source: an original, fully specified simulated environment in
project/environment.py and examples/warehouse.json. No external download is
required. The map, transitions, rewards, and seeds are reproducible.
Original code and environment definitions use the repository's MIT license.

Reference algorithms:
Sutton, R. S., and Barto, A. G. (2018). Reinforcement Learning: An Introduction,
second edition. Sections on dynamic programming and on-policy/off-policy
temporal-difference control.

Publisher page:
https://mitpress.mit.edu/9780262039246/reinforcement-learning/

Configuration includes rows, columns, goal, walls, hazards, slip probability,
and rewards. Safe starting states exclude walls, hazards, and goal.
Evaluation episode records retain return, discounted return, success,
hazard, timeout, step count, and collision count.

This environment contains no real robot observations. Performance is specific
to its state abstraction, reward specification, and simulated transitions.
