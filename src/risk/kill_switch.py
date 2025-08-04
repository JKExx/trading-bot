"""Emergency stop mechanism."""

class KillSwitch:
    def should_stop(self, drawdown: float, limit: float) -> bool:
        """Return True if drawdown exceeds limit."""
        return drawdown <= -abs(limit)
