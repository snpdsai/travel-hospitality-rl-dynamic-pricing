# Environment Settings

MAX_INVENTORY = 50
MAX_DAYS = 30
BASE_PRICE = 100

# Pricing Actions

PRICE_LEVELS = {
    0: 80,
    1: 90,
    2: 100,
    3: 110,
    4: 120
}

PRICE_MULTIPLIERS = {
    0: 0.8,
    1: 0.9,
    2: 1.0,
    3: 1.1,
    4: 1.2
}

# Evaluation

DEFAULT_EPISODES = 100

# Random Seed

RANDOM_SEED = 42

# ==========================
# DQN Hyperparameters
# ==========================

STATE_SIZE = 2          # Inventory, Days Left
ACTION_SIZE = 5         # Five pricing levels

HIDDEN_SIZE = 64

LEARNING_RATE = 1e-3

GAMMA = 0.95