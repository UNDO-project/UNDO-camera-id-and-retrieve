"""Catalog management endpoints."""

from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from src.api.dependencies import get_identification_service, state
from src.api.models.responses import CatalogStatsResponse
from src.identification.service import IdentificationService
from src.config import paths

router = APIRouter()


@router.get("/stats", response_model=CatalogStatsResponse)
async def get_catalog_stats(
    service: IdentificationService = Depends(get_identification_service),
) -> CatalogStatsResponse:
    """Get catalog statistics.

    Returns information about the loaded catalog including:
    - Total number of cameras
    - Embeddings status
    - Catalog file path

    Args:
        service: Injected identification service

    Returns:
        CatalogStatsResponse: Catalog statistics

    Example:
        ```bash
        curl http://localhost:8000/api/v1/catalog/stats
        ```
    """
    try:
        # Check if catalog index exists
        catalog_loaded = (
            hasattr(service, "catalog_index") and service.catalog_index is not None
        )

        if not catalog_loaded:
            return CatalogStatsResponse(
                total_cameras=0,
                embeddings_loaded=False,
                embedding_count=0,
                catalog_path="Not loaded",
            )

        catalog_index = service.catalog_index

        # Get embedding count
        embedding_count = 0
        embeddings_loaded = False
        if (
            hasattr(catalog_index, "embeddings")
            and catalog_index.embeddings is not None
        ):
            embeddings_loaded = True
            embedding_count = len(catalog_index.embeddings)

        # Get total cameras from catalog
        total_cameras = 0
        if hasattr(catalog_index, "catalog") and catalog_index.catalog is not None:
            total_cameras = len(catalog_index.catalog)

        # Get catalog path
        catalog_path = str(paths.output_dir / "products.parquet")

        return CatalogStatsResponse(
            total_cameras=total_cameras,
            embeddings_loaded=embeddings_loaded,
            embedding_count=embedding_count,
            catalog_path=catalog_path,
        )

    except Exception as e:
        logger.error(f"Error getting catalog stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting catalog stats: {str(e)}"
        )


@router.post("/reload")
async def reload_catalog() -> Dict[str, Any]:
    """Reload the catalog and embeddings.

    This is an admin endpoint that forces reinitialization of the
    IdentificationService, reloading the catalog from disk.

    Returns:
        Dict[str, Any]: Success status and message

    Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/catalog/reload
        ```

    Note:
        This operation may take several seconds depending on catalog size.
        Consider adding authentication to this endpoint in production.
    """
    try:
        logger.info("Catalog reload requested via API")
        state.reload_catalog()
        return {"success": True, "message": "Catalog reloaded successfully"}

    except Exception as e:
        logger.error(f"Error reloading catalog: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error reloading catalog: {str(e)}"
        )
