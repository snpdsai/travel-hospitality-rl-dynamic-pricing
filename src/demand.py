import numpy as np


def purchase_probability(price, days_left):
    """
    Returns the probability that a customer purchases
    at the given price and remaining days.
    """

    base_probability = {
        80: 0.90,
        100: 0.75,
        120: 0.60,
        140: 0.40,
        160: 0.25,
    }

    probability = base_probability[price]

    # Increase urgency as departure approaches
    urgency_bonus = (15 - days_left) * 0.02

    probability += urgency_bonus

    # Keep probability between 5% and 99%
    probability = np.clip(probability, 0.05, 0.99)

    return probability