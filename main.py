"""Entry point for the trading bot."""
from src.utils.logger import create_logger


def main() -> None:
    logger = create_logger("main")
    logger.info("Trading bot started")


if __name__ == "__main__":
    main()
