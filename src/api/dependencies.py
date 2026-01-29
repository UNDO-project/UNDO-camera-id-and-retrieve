"""API dependencies and state management."""

from typing import Optional

from loguru import logger

from src.identification.service import IdentificationService
from src.identification.async_service import AsyncIdentificationService
from src.api.services.catalog import CatalogService
from src.api.websocket.manager import ConnectionManager


class ServiceState:
    """Global state manager for the API service.

    Implements a singleton pattern to ensure services are initialized
    only once and reused across requests.
    """

    def __init__(self):
        """Initialize the service state."""
        self._service: Optional[IdentificationService] = None
        self._async_service: Optional[AsyncIdentificationService] = None
        self._catalog_service: Optional[CatalogService] = None
        self._connection_manager: Optional[ConnectionManager] = None

    @property
    def service(self) -> IdentificationService:
        r"""Get or initialize the identification service.

        :return: The identification service instance
        """
        if self._service is None:
            logger.info("Initializing IdentificationService...")
            self._service = IdentificationService()
            logger.success("IdentificationService initialized successfully")
        return self._service

    @property
    def async_service(self) -> AsyncIdentificationService:
        r"""Get or initialize the async identification service.

        :return: The async identification service instance
        """
        if self._async_service is None:
            logger.info("Initializing AsyncIdentificationService...")
            self._async_service = AsyncIdentificationService()
            logger.success("AsyncIdentificationService initialized successfully")
        return self._async_service

    @property
    def catalog_service(self) -> CatalogService:
        r"""Get or initialize the catalog service.

        :return: The catalog service instance
        """
        if self._catalog_service is None:
            logger.info("Initializing CatalogService...")
            self._catalog_service = CatalogService()
            logger.success("CatalogService initialized successfully")
        return self._catalog_service

    @property
    def connection_manager(self) -> ConnectionManager:
        r"""Get or initialize the connection manager.

        :return: The connection manager instance
        """
        if self._connection_manager is None:
            logger.info("Initializing ConnectionManager...")
            self._connection_manager = ConnectionManager()
            logger.success("ConnectionManager initialized successfully")
        return self._connection_manager

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
    r"""FastAPI dependency to inject the identification service.

    :return: The shared identification service instance

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
    r"""FastAPI dependency to inject the catalog service.

    :return: The shared catalog service instance

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


def get_async_identification_service() -> AsyncIdentificationService:
    r"""FastAPI dependency to inject the async identification service.

    :return: The shared async identification service instance

    Example:
        ```python
        @app.websocket("/ws/video-stream")
        async def video_stream(
            websocket: WebSocket,
            service: AsyncIdentificationService = Depends(get_async_identification_service)
        ):
            detections = await service.detect_frame(frame_bytes)
        ```
    """
    return state.async_service


def get_connection_manager() -> ConnectionManager:
    r"""FastAPI dependency to inject the connection manager.

    :return: The shared connection manager instance

    Example:
        ```python
        @app.websocket("/ws/video-stream")
        async def video_stream(
            websocket: WebSocket,
            manager: ConnectionManager = Depends(get_connection_manager)
        ):
            await manager.connect(websocket, client_id, config)
        ```
    """
    return state.connection_manager
