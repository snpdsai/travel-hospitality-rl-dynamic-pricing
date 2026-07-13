from src.environment import DynamicPricingEnv

env = DynamicPricingEnv()

state, _ = env.reset()

done = False

total_reward = 0

while not done:

    action = env.action_space.sample()

    state, reward, done, truncated, info = env.step(action)

    print(
        f"State={state}, "
        f"Reward={reward}"
    )

    total_reward += reward

print("\nEpisode Finished")
print("Total Revenue:", total_reward)