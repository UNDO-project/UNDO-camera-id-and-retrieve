import random

from src.config import scraper


def get_random_delay() -> float:
    r"""
    Get a random delay between MIN and MAX to mimic human behavior.

    :return: Random delay in seconds
    """
    return random.uniform(scraper.min_request_delay, scraper.max_request_delay)
