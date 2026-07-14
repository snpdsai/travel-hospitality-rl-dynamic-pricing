class FixedPriceAgent:
    """
    Always selects the same price.
    """

    def __init__(self, action=2):
        self.action = action

    def choose_action(self, state):
        return self.action


class TimeDiscountAgent:
    """
    Gradually reduces prices as departure approaches.
    """

    def choose_action(self, state):

        inventory, days_left = state

        if days_left > 20:
            return 4      # Highest price

        elif days_left > 10:
            return 2      # Base price

        else:
            return 0      # Discounted price