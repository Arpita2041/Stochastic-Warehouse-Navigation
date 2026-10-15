# Validation

The delivery environment used Python 3.12.14 with the versions
in requirements.txt. All 5 tests in this repository passed.
The complete delivery contains 26 passing tests across the five projects.
Every project's command-line prediction or replay path also completed.

Verified properties include the project's mathematical and data-protocol checks,
model serialization, and the example inference path. External ZIP member selection,
nested archives, download caching, and downloaded-file hashes were tested with
mocked responses in projects that download a dataset.

Notebook files were checked as JSON notebooks, and every code cell was checked
for Python syntax. The underlying experiments were executed from the command line.
The Jupyter interface was not executed in the delivery environment.

Both quick and full navigation experiments completed. The full
experiment trained Q-learning and SARSA for 15,000 episodes per method for each
of three seeds, then evaluated every policy for 1,000 episodes per repetition.
reference_run contains saved policies, episode records, measured results, plots,
and an example trajectory.

Tests verified transition probability mass, absorbing terminal behavior,
collision/hazard handling, Bellman residual, learning on a deterministic test
MDP, and rejection of an unreachable map.
