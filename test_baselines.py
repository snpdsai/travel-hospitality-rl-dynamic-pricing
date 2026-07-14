from src.baseline_agents import (
    FixedPriceAgent,
    TimeDiscountAgent
)

from src.evaluate import run_episode


fixed_agent = FixedPriceAgent()

discount_agent = TimeDiscountAgent()

fixed_reward = run_episode(fixed_agent)

discount_reward = run_episode(discount_agent)

print(f"Fixed Price Revenue: {fixed_reward}")
print(f"Time Discount Revenue: {discount_reward}")