"""Base scraper class."""

from abc import ABC, abstractmethod
from src.models.camera import Camera


class BaseScraper(ABC):
    """
    Abstract base class for vendor-specific scrapers.

    All scraper implementations should inherit from this class
    and implement the ``scrape`` method.
    """

    @abstractmethod
    def scrape(self) -> list[Camera]:
        """
        Scrape CCTV camera data from vendor source.

        :returns: List of Camera objects
        """
        pass
