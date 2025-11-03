"""Camera data model."""

from pydantic import BaseModel


class Camera(BaseModel):
    """
    Represents a CCTV camera product.

    :ivar name: Camera model name
    :ivar vendor: Manufacturer name
    :ivar url: Product URL
    """

    name: str
    vendor: str
    url: str
