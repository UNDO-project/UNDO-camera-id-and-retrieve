import random

from src.config import MIN_REQUEST_DELAY, MAX_REQUEST_DELAY


def get_random_delay() -> float:
    r"""
    Get a random delay between MIN and MAX to mimic human behavior.

    :return: Random delay in seconds
    """
    return random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY)
