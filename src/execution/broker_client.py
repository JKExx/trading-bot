"""Abstract broker client interface."""

class BrokerClient:
    def place_order(self, order):
        """Place an order with the broker."""
        raise NotImplementedError
