"""Catalog management endpoints."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from src.api.dependencies import get_identification_service, get_catalog_service, state
from src.api.models.responses import (
    CatalogStatsResponse,
    IndexInfoResponse,
    MatchingFlagsInfo,
)
from src.api.models.catalog import (
    CatalogListResponse,
    CatalogFacetsResponse,
    CameraDetailResponse,
    PaginationInfo,
)
from src.identification.service import IdentificationService
from src.api.services.catalog import CatalogService
from src.config import paths

router = APIRouter()


@router.get("/stats", response_model=CatalogStatsResponse)
async def get_catalog_stats(
    service: IdentificationService = Depends(get_identification_service),
) -> CatalogStatsResponse:
    r"""Get catalog statistics.

    Returns information about the loaded catalog including:
    - Total number of cameras
    - Embeddings status
    - Catalog file path

    :param service: Injected identification service
    :return: Catalog statistics

    Example:
        ```bash
        curl http://localhost:8000/api/v1/catalog/stats
        ```
    """
    try:
        index_loaded = hasattr(service, "index") and service.index is not None

        if not index_loaded:
            return CatalogStatsResponse(
                total_cameras=0,
                vendor_count=0,
                embeddings_loaded=False,
                embedding_count=0,
                catalog_path="Not loaded",
            )

        catalog_index = service.index

        embedding_count = 0
        embeddings_loaded = False
        if (
            hasattr(catalog_index, "embeddings")
            and catalog_index.embeddings is not None
        ):
            embeddings_loaded = True
            embedding_count = len(catalog_index.embeddings)

        total_cameras = 0
        vendor_count = 0
        if hasattr(service, "catalog") and service.catalog is not None:
            total_cameras = len(service.catalog)

            unique_vendors = set()
            for camera_record in service.catalog.values():
                if hasattr(camera_record, "source") and camera_record.source:
                    unique_vendors.add(camera_record.source)
            vendor_count = len(unique_vendors)

        catalog_path = str(paths.output_dir / "products.parquet")

        return CatalogStatsResponse(
            total_cameras=total_cameras,
            vendor_count=vendor_count,
            embeddings_loaded=embeddings_loaded,
            embedding_count=embedding_count,
            catalog_path=catalog_path,
        )

    except Exception as e:
        logger.error(f"Error getting catalog stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting catalog stats: {str(e)}"
        )


@router.get("/index-info", response_model=IndexInfoResponse)
async def get_index_info(
    service: IdentificationService = Depends(get_identification_service),
) -> IndexInfoResponse:
    r"""Get read-only metadata about the loaded embeddings index.

    Once retrieval depends on matching flags and index build parameters,
    "why did results change?" becomes a real debugging question. This
    endpoint reports which index configuration is serving requests:
    build timestamp, row/product counts, augmentation status, and the
    active ``CIDAR_MATCH_*`` settings.

    :param service: Injected identification service
    :return: Index metadata and active matching flags

    Example:
        ```bash
        curl http://localhost:8000/api/v1/catalog/index-info
        ```

    .. note::
        The index is loaded once at service initialization. After
        rebuilding embeddings (e.g. enabling augmentation), restart the
        API — ``/catalog/reload`` reloads the parquet catalog, not the
        embeddings index.
    """
    try:
        from src.config import matching

        index = service.index

        embeddings = getattr(index, "embeddings", None)
        total_rows = int(embeddings.shape[0]) if embeddings is not None else 0
        embedding_dim = (
            int(embeddings.shape[1])
            if embeddings is not None and embeddings.ndim == 2
            else 0
        )

        camera_ids = getattr(index, "camera_ids", None) or []
        total_products = len(set(camera_ids))
        variant_tags = getattr(index, "variant_tags", None)

        embeddings_path = getattr(index, "embeddings_path", None)
        built_at = None
        if embeddings_path is not None and Path(embeddings_path).exists():
            built_at = datetime.fromtimestamp(
                Path(embeddings_path).stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")

        return IndexInfoResponse(
            embeddings_path=str(embeddings_path) if embeddings_path else "unknown",
            built_at=built_at,
            total_rows=total_rows,
            total_products=total_products,
            embedding_dim=embedding_dim,
            augmented=variant_tags is not None,
            variants_per_product=(
                round(total_rows / total_products, 2) if total_products else 0.0
            ),
            mean_vector_present=getattr(index, "mean_vector", None) is not None,
            matching=MatchingFlagsInfo(
                crop_margin=matching.crop_margin,
                augment_enabled=matching.augment_enabled,
                augment_k=matching.augment_k,
                mean_center=matching.mean_center,
                mean_center_active=bool(getattr(index, "_center_active", False)),
            ),
        )

    except Exception as e:
        logger.error(f"Error getting index info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting index info: {str(e)}"
        )


@router.post("/reload")
async def reload_catalog() -> Dict[str, Any]:
    r"""Reload the catalog and embeddings.

    This is an admin endpoint that forces reinitialization of the
    IdentificationService, reloading the catalog from disk.

    :return: Success status and message

    Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/catalog/reload
        ```

    .. note::
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


@router.get("/cameras", response_model=CatalogListResponse)
async def list_cameras(
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    vendor: Optional[str] = Query(default=None, description="Filter by vendor"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    series: Optional[str] = Query(default=None, description="Filter by product series"),
    search: Optional[str] = Query(default=None, description="Search query"),
    catalog_service: CatalogService = Depends(get_catalog_service),
) -> CatalogListResponse:
    r"""List cameras with pagination and filtering.

    Returns a paginated list of cameras with optional filters for vendor,
    category, series, and full-text search.

    :param page: Page number (1-indexed)
    :param limit: Number of items per page (max 100)
    :param vendor: Filter by vendor/source name
    :param category: Filter by camera category
    :param series: Filter by product series
    :param search: Search query for model_name and display_name
    :param catalog_service: Injected catalog service
    :return: CatalogListResponse with cameras and pagination info

    Example:
        ```bash
        curl "http://localhost:8000/api/v1/catalog/cameras?page=1&limit=20&vendor=Axis"
        ```
    """
    try:
        cameras, total = catalog_service.filter_cameras(
            vendor=vendor,
            category=category,
            series=series,
            search=search,
            page=page,
            limit=limit,
        )

        total_pages = (total + limit - 1) // limit if total > 0 else 0

        facets = catalog_service.get_facets()

        return CatalogListResponse(
            data=cameras,
            pagination=PaginationInfo(
                page=page,
                limit=limit,
                total=total,
                total_pages=total_pages,
            ),
            facets=facets,
        )

    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing cameras: {str(e)}")


@router.get("/cameras/{camera_id}", response_model=CameraDetailResponse)
async def get_camera(
    camera_id: str,
    catalog_service: CatalogService = Depends(get_catalog_service),
) -> CameraDetailResponse:
    r"""Get full camera details by ID.

    Returns complete information about a single camera including
    specifications, images, and datasheet link.

    :param camera_id: Unique camera identifier
    :param catalog_service: Injected catalog service
    :return: CameraDetailResponse with full camera details

    Example:
        ```bash
        curl "http://localhost:8000/api/v1/catalog/cameras/axis-m3057-plr-mk-ii"
        ```
    """
    try:
        camera = catalog_service.get_camera_detail(camera_id)

        if camera is None:
            raise HTTPException(
                status_code=404, detail=f"Camera not found: {camera_id}"
            )

        return camera

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting camera: {str(e)}")


@router.get("/facets", response_model=CatalogFacetsResponse)
async def get_catalog_facets(
    catalog_service: CatalogService = Depends(get_catalog_service),
) -> CatalogFacetsResponse:
    r"""Get available filter options with counts.

    Returns facet data for vendors, categories, and product series
    to support UI filter rendering.

    :param catalog_service: Injected catalog service
    :return: CatalogFacetsResponse with filter options and counts

    Example:
        ```bash
        curl "http://localhost:8000/api/v1/catalog/facets"
        ```
    """
    try:
        facets = catalog_service.get_facets()

        return CatalogFacetsResponse(
            vendors=facets.get("vendors", []),
            categories=facets.get("categories", []),
            series=facets.get("series", []),
        )

    except Exception as e:
        logger.error(f"Error getting facets: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting facets: {str(e)}")
