import numpy as np

from src.q_learning import QLearningAgent
from src.evaluate import evaluate_qlearning

agent = QLearningAgent()

agent.q_table = np.load("models/q_table.npy")

results = evaluate_qlearning(agent)

print(results)