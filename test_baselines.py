from src.baseline_agents import (
    FixedPriceAgent,
    TimeDiscountAgent
)

from src.evaluate import evaluate_agent

fixed = FixedPriceAgent()
discount = TimeDiscountAgent()

print("Fixed Price Strategy")
print(evaluate_agent(fixed))

print()

print("Time Discount Strategy")
print(evaluate_agent(discount))