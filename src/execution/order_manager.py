"""Manage order lifecycle."""
from .broker_client import BrokerClient


class OrderManager:
    def __init__(self, broker: BrokerClient):
        self.broker = broker

    def submit(self, order):
        """Submit an order via the broker."""
        self.broker.place_order(order)
