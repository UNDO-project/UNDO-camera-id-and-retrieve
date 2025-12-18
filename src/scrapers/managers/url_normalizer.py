"""URL normalization strategies for different vendors."""

from abc import ABC, abstractmethod


class URLNormalizer(ABC):
    """Abstract base class for URL normalization strategies."""

    @abstractmethod
    def normalize(self, url: str | None, base_url: str) -> str | None:
        """
        Normalize a URL to absolute format.

        :param url: URL that may be relative or malformed
        :param base_url: Base URL for resolving relative URLs
        :return: Normalized absolute URL, or None if invalid
        """
        pass


class RelativeURLNormalizer(URLNormalizer):
    """Handles relative URLs starting with '/' (e.g., Axis)."""

    def normalize(self, url: str | None, base_url: str) -> str | None:
        """
        Normalize relative URLs starting with '/'.

        :param url: URL that may be relative
        :param base_url: Base URL for resolving relative URLs
        :return: Normalized absolute URL, or None if invalid
        """
        if not url:
            return None
        if url.startswith("/"):
            return f"{base_url}{url}"
        return url


class ProtocolRelativeURLNormalizer(URLNormalizer):
    """Handles protocol-relative URLs starting with '//' (e.g., Hikvision)."""

    def normalize(self, url: str | None, base_url: str) -> str | None:
        """
        Normalize protocol-relative URLs starting with '//'.

        :param url: URL that may be protocol-relative
        :param base_url: Base URL (unused but kept for interface consistency)
        :return: Normalized absolute URL with https: prefix, or None if invalid
        """
        if not url:
            return None
        if url.startswith("//"):
            return f"https:{url}"
        return url
