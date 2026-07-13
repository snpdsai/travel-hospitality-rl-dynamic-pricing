import numpy as np


class DemandModel:
    """
    Simulates customer purchase probability.
    """

    def __init__(self):

        self.price_multipliers = {
            0: 0.8,
            1: 0.9,
            2: 1.0,
            3: 1.1,
            4: 1.2
        }

    def purchase_probability(self, action, days_left):
        """
        Returns probability of purchase.
        """

        price_factor = self.price_multipliers[action]

        # Lower price -> higher probability
        price_effect = 1.2 - price_factor

        # Demand increases near departure
        time_effect = (30 - days_left) / 30

        probability = 0.30 + 0.45 * price_effect + 0.25 * time_effect

        return np.clip(probability, 0.05, 0.95)

    def customer_buys(self, action, days_left):
        """
        Returns True if a purchase occurs.
        """

        p = self.purchase_probability(action, days_left)

        return np.random.rand() < p