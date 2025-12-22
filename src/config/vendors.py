from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AxisSettings(BaseSettings):
    """Configuration for Axis Communications scraping."""

    model_config = SettingsConfigDict(
        env_prefix="CIDAR_AXIS_",
        case_sensitive=False,
    )

    base_url: str = "https://www.axis.com"

    @computed_field
    @property
    def products_url(self) -> str:
        """Full products URL."""
        return f"{self.base_url}/products/network-cameras"


class HikVisionSettings(BaseSettings):
    """Configuration for HikVision scraping."""

    model_config = SettingsConfigDict(
        env_prefix="CIDAR_HIKVISION_",
        case_sensitive=False,
    )

    base_url: str = "https://www.hikvision.com"
    region: str = "europe"
    page_load_timeout: int = 30000
    products_per_page: int = 12

    @computed_field
    @property
    def ip_products_url(self) -> str:
        """IP products URL."""
        return f"{self.base_url}/{self.region}/products/IP-Products/"

    @computed_field
    @property
    def its_products_url(self) -> str:
        """ITS products URL."""
        return f"{self.base_url}/{self.region}/products/ITS-Products/traffic-cameras/"

    @computed_field
    @property
    def thermal_products_url(self) -> str:
        """Thermal products URL."""
        return f"{self.base_url}/{self.region}/products/Thermal-Products"
