"""Combine strategies to produce signals."""
from .smc_strategy import SMCStrategy
from .price_action import PriceActionStrategy


class SignalGenerator:
    def __init__(self):
        self.smc = SMCStrategy()
        self.price_action = PriceActionStrategy()

    def generate(self, data):
        """Generate a consolidated signal."""
        return self.smc.generate_signal(data) or self.price_action.analyze(data)
