# Travel & Hospitality Dynamic Pricing using Reinforcement Learning

This project develops an autonomous pricing agent using Reinforcement Learning to maximize revenue from finite inventory over a limited selling horizon.

## Project Structure

(To be updated)

### Week 1 Progress

- Implemented stochastic customer demand model
- Simulated customer purchase decisions
- Completed Gymnasium environment step function
- Simulated one booking season
- Implemented custom Gymnasium environment
- Built stochastic customer demand model
- Developed baseline pricing strategies
- Created evaluation pipeline for heuristic agents

### Code Refactoring

- Centralized configuration values in `config.py`
- Improved project maintainability by removing hard-coded constants
- Standardized evaluation settings

### Week 2 Progress

- Implemented the Q-Learning agent
- Initialized the Q-table
- Added epsilon-greedy action selection
- Implemented Bellman update equation
- Added epsilon decay
- Trained Q-Learning agent over multiple episodes
- Trained Q-Learning agent
- Saved the learned Q-table
- Evaluated the learned policy against baseline strategies
- Visualized Q-Learning training rewards
- Compared heuristic baselines with the learned policy
- Generated evaluation plots for analysis

## Week 2 Deliverables

✔ Baseline Pricing Strategies

✔ Tabular Q-Learning Agent

✔ Bellman Update

✔ Epsilon-Greedy Exploration

✔ Model Evaluation

✔ Training Curve Visualization

✔ Revenue Comparison

✔ Evaluation Metrics

## Neural Network Architecture

The DQN model is a fully connected feed-forward neural network.

Architecture:

Input Layer

- Remaining Inventory
- Days Until Departure

↓

Hidden Layer (64 neurons)

↓

ReLU Activation

↓

Hidden Layer (64 neurons)

↓

ReLU Activation

↓

Output Layer (5 neurons)

Each output neuron estimates the expected future reward (Q-value) for one pricing action.

## Week 3 Progress

- Implemented Deep Q-Network architecture
- Added PyTorch neural network
- Validated forward pass
- Added Experience Replay Buffer
- Implemented random mini-batch sampling
- Prepared DQN training pipeline
- Added DQN Agent class
- Integrated neural network and optimizer
- Implemented epsilon-greedy action selection
- Added epsilon decay
- Implemented DQN training step
- Added mini-batch learning
- Computed Bellman targets
- Updated neural network using gradient descent
- Added Target Network
- Implemented weight synchronization
- Updated DQN training to use stable target Q-values
- Completed DQN training pipeline
- Added target network synchronization
- Saved trained DQN model
- Logged episode rewards