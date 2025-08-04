"""Risk management utilities."""

class RiskManager:
    def position_size(self, account_balance: float, risk_per_trade: float, stop_loss: float) -> float:
        """Calculate position size based on risk."""
        return account_balance * risk_per_trade / max(stop_loss, 1e-8)
