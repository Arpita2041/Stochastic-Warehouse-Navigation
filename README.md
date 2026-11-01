# Stochastic Warehouse Navigation with Reinforcement Learning

The main objective is to develop a Reinforcement Learning (RL) system that teaches an agent to navigate safely through a warehouse, avoid obstacles and hazardous locations, and reach a destination efficiently.

## Workflow

The environment is original and fully specified in project/environment.py and examples/warehouse.json. It is created an 8 × 8 grid-based warehouse environment containing walls, hazards, and a goal location.
The proposed idea compares four methods: Random Policy, Value Iteration, Q-Learning, SARSA.

## Results

  - SARSA achieved the highest observed success rate, approximately 98.7%, along with the lowest hazard rate.
  - Value Iteration achieved the lowest average number of steps and the highest discounted return among the informed methods.
  - Q-Learning also learned a successful navigation strategy, substantially outperforming the Random Policy.

The comparison demonstrates that agents can learn effective routes through interactions with the simulated environment.

**Development date:** November 2025
