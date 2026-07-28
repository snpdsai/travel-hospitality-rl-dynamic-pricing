"""
Configuration file for the Travel & Hospitality
Dynamic Pricing Reinforcement Learning Project.
"""

# ==========================================================
# Environment Configuration
# ==========================================================

# Initial inventory available at the start of each episode
INITIAL_INVENTORY = 20

# Number of booking days before departure
BOOKING_HORIZON = 15

# Available pricing actions
PRICE_LEVELS = [
    80,
    100,
    120,
    140,
    160,
]

# ==========================================================
# Q-Learning Hyperparameters
# ==========================================================

QL_ALPHA = 0.1
QL_GAMMA = 0.95

QL_EPSILON_START = 1.0
QL_EPSILON_DECAY = 0.995
QL_EPSILON_MIN = 0.05

QL_EPISODES = 500

# ==========================================================
# DQN Hyperparameters
# ==========================================================

DQN_GAMMA = 0.95

LEARNING_RATE = 0.001

BATCH_SIZE = 64

REPLAY_CAPACITY = 10_000

DQN_EPSILON_START = 1.0
DQN_EPSILON_DECAY = 0.995
DQN_EPSILON_MIN = 0.05

TAU = 0.005

DQN_EPISODES = 500

# ==========================================================
# Evaluation
# ==========================================================

EVALUATION_EPISODES = 100

MOVING_AVERAGE_WINDOW = 20

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42