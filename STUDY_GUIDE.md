# Study guide

Read project/environment.py and enumerate the possible outcomes of one action.
Then derive both update rules in project/algorithms.py.

Be prepared to explain:

1. The full MDP definition and why the state is Markov for this simulator.
2. How boundary and wall collisions affect transition probabilities.
3. Why the transition probabilities sum to one, even when outcomes repeat.
4. Why goal and hazard transitions must not bootstrap.
5. What information value iteration has that the learners do not.
6. The difference between Q-learning and SARSA targets.
7. How exploration affects SARSA's learned risk preferences.
8. Why epsilon-greedy action ties are randomized during training.
9. The difference between truncation and terminal completion.
10. Why success rate and discounted return need not rank policies identically.
11. What sources of variation appear in the reported standard deviations.
12. Why a tabular policy cannot be transferred by changing the map file alone.

For the practical exam, calculate a Bellman backup at a cell adjacent to a
hazard. Replay a saved policy and identify an intended move that slipped.
Change a reward or slip probability, retrain with the fixed protocol, and
explain the resulting behavior without choosing a configuration from test scores.
