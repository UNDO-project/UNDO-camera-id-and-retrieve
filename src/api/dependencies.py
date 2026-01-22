"""API dependencies and state management."""

from typing import Optional

from loguru import logger

from src.identification.service import IdentificationService
from src.api.services.catalog import CatalogService


class ServiceState:
    """Global state manager for the API service.

    Implements a singleton pattern to ensure services are initialized
    only once and reused across requests.
    """

    def __init__(self):
        """Initialize the service state."""
        self._service: Optional[IdentificationService] = None
        self._catalog_service: Optional[CatalogService] = None

    @property
    def service(self) -> IdentificationService:
        """Get or initialize the identification service.

        Returns:
            IdentificationService: The identification service instance.
        """
        if self._service is None:
            logger.info("Initializing IdentificationService...")
            self._service = IdentificationService()
            logger.success("IdentificationService initialized successfully")
        return self._service

    @property
    def catalog_service(self) -> CatalogService:
        """Get or initialize the catalog service.

        Returns:
            CatalogService: The catalog service instance.
        """
        if self._catalog_service is None:
            logger.info("Initializing CatalogService...")
            self._catalog_service = CatalogService()
            logger.success("CatalogService initialized successfully")
        return self._catalog_service

    def reload_catalog(self) -> None:
        """Reload the catalog and embeddings.

        This forces reinitialization of both the IdentificationService
        and CatalogService, useful for admin operations or when catalog
        data changes.
        """
        logger.info("Reloading catalog and embeddings...")

        self._service = None
        _ = self.service

        if self._catalog_service is not None:
            self._catalog_service.reload_catalog()

        logger.success("Catalog reloaded successfully")

    def is_ready(self) -> bool:
        """Check if the service is ready.

        Returns:
            bool: True if service is initialized and ready.
        """
        return self._service is not None


# Global state instance
state = ServiceState()


def get_identification_service() -> IdentificationService:
    """FastAPI dependency to inject the identification service.

    Returns:
        IdentificationService: The shared identification service instance.

    Example:
        ```python
        @app.post("/identify")
        async def identify(service: IdentificationService = Depends(get_identification_service)):
            # Use service
            results = service.identify_from_image(...)
        ```
    """
    return state.service


def get_catalog_service() -> CatalogService:
    """FastAPI dependency to inject the catalog service.

    Returns:
        CatalogService: The shared catalog service instance.

    Example:
        ```python
        @app.get("/catalog/cameras")
        async def list_cameras(
            service: CatalogService = Depends(get_catalog_service)
        ):
            cameras, total = service.filter_cameras(...)
        ```
    """
    return state.catalog_service
