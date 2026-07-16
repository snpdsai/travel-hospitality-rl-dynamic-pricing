from src.q_learning import QLearningAgent

agent = QLearningAgent()

print("Q-table Shape:", agent.q_table.shape)

state = (50, 30)

for i in range(10):

    action = agent.choose_action(state)

    print(f"Action {i+1}: {action}")