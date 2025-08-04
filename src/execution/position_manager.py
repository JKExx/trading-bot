"""Track open positions."""

class PositionManager:
    def __init__(self):
        self.positions = []

    def add(self, position):
        self.positions.append(position)
