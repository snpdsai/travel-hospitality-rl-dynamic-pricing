from src.dqn_agent import DQNAgent

agent = DQNAgent()

state = [50, 30]

print("Initial epsilon:", agent.epsilon)

print()

for i in range(10):

    action = agent.choose_action(state)

    print(
        f"Action {i+1}:",
        action
    )

agent.decay_epsilon()

print()

print("New epsilon:", agent.epsilon)