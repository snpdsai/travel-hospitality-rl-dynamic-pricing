import os
import pandas as pd
import numpy as np

from src.baseline_agents import FixedPriceAgent, TimeDiscountAgent
from src.q_learning import QLearningAgent
from src.evaluate import evaluate_agent, evaluate_qlearning

os.makedirs("outputs/metrics", exist_ok=True)

fixed = FixedPriceAgent()
discount = TimeDiscountAgent()

fixed_metrics = evaluate_agent(fixed)
discount_metrics = evaluate_agent(discount)

q_agent = QLearningAgent()
q_agent.q_table = np.load("models/q_table.npy")

q_metrics = evaluate_qlearning(q_agent)

results = pd.DataFrame(
    [
        ["Fixed Price", *fixed_metrics.values()],
        ["Time Discount", *discount_metrics.values()],
        ["Q-Learning", *q_metrics.values()]
    ],
    columns=[
        "Agent",
        "Average Revenue",
        "Std Revenue",
        "Min Revenue",
        "Max Revenue"
    ]
)

results.to_csv(
    "outputs/metrics/evaluation_metrics.csv",
    index=False
)

print(results)