"""Catalog service for loading and querying camera data."""

import json
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
from loguru import logger

from src.config import paths
from src.models.camera import CameraRecord
from src.api.models.catalog import (
    CameraSummaryResponse,
    CameraDetailResponse,
    CameraSpecsSummary,
    FacetItem,
)


class CatalogService:
    """Service for loading and querying camera catalog data.

    Provides lazy loading of the catalog with explicit lifecycle control.
    """

    def __init__(self) -> None:
        """Initialize catalog service with lazy loading."""
        self._catalog: Optional[pd.DataFrame] = None

    @property
    def catalog(self) -> pd.DataFrame:
        r"""Get or load catalog (lazy initialization).

        :return: DataFrame with camera records
        :raises FileNotFoundError: If dataset parquet file does not exist
        """
        if self._catalog is None:
            self._catalog = self._load_catalog()
        return self._catalog

    @staticmethod
    def _load_catalog() -> pd.DataFrame:
        r"""Load catalog from parquet file.

        :return: DataFrame with camera records
        :raises FileNotFoundError: If dataset parquet file does not exist
        """
        dataset_path = paths.output_dir / "products.parquet"

        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_parquet(dataset_path)
        logger.info(f"Loaded {len(df)} camera records from catalog")
        return df

    def reload_catalog(self) -> None:
        """Force reload of catalog from disk.

        Clears the cached catalog so the next access will reload from disk.
        """
        self._catalog = None
        logger.info("Catalog cache cleared")

    @staticmethod
    def _row_to_record(row: pd.Series) -> CameraRecord:
        r"""Convert a DataFrame row to a CameraRecord.

        :param row: DataFrame row containing camera data
        :return: CameraRecord instance
        """
        specs = {}
        if row.get("specifications"):
            try:
                specs = json.loads(row["specifications"])
            except (json.JSONDecodeError, TypeError):
                specs = {}

        images = []
        if row.get("image_urls"):
            try:
                images = json.loads(row["image_urls"])
            except (json.JSONDecodeError, TypeError):
                images = []

        image_files = []
        if row.get("image_files"):
            try:
                image_files = json.loads(row["image_files"])
            except (json.JSONDecodeError, TypeError):
                image_files = []

        return CameraRecord(
            camera_id=row["camera_id"],
            model_name=row["model_name"],
            display_name=row["display_name"],
            description=row.get("description"),
            specifications=specs,
            image_url=images[0] if images else None,
            images=images,
            image_files=image_files,
            datasheet_url=row.get("datasheet_url"),
            datasheet_file=row.get("datasheet_file"),
            source=row["source"],
            category=row["category"],
            product_category=row.get("product_category", ""),
            product_series=row.get("product_series", ""),
        )

    @staticmethod
    def _flatten_specs(specs: Dict[str, Any]) -> Optional[CameraSpecsSummary]:
        r"""Extract commonly-used fields from specifications dict.

        :param specs: Full specifications dictionary
        :return: CameraSpecsSummary with flattened fields, None if specs is empty
        """
        if not specs:
            return None

        spec_lower = {k.lower(): v for k, v in specs.items()}

        def get_value(*keys: str) -> Optional[str]:
            for key in keys:
                key_lower = key.lower()
                if key_lower in spec_lower:
                    value = spec_lower[key_lower]

                    if isinstance(value, dict):
                        for subkey in [
                            "Focal length",
                            "focal length",
                            "value",
                            "Value",
                        ]:
                            if subkey in value:
                                subvalue = value[subkey]
                                return (
                                    str(subvalue)
                                    if not isinstance(subvalue, dict)
                                    else None
                                )
                        continue

                    if isinstance(value, str):
                        return value
                    return str(value)
            return None

        return CameraSpecsSummary(
            max_resolution=get_value("max resolution", "resolution"),
            lens=get_value("lens", "focal length"),
            horizontal_fov=get_value("horizontal fov", "fov"),
        )

    @staticmethod
    def _build_thumbnail_url(image_file: str) -> str:
        r"""Build thumbnail URL for an image file.

        :param image_file: Local file path, may include data/images/ prefix
        :return: URL path for the image endpoint
        """
        clean_path = image_file.removeprefix("data/images/")
        return f"/api/v1/images/{clean_path}"

    @staticmethod
    def _build_datasheet_url(datasheet_file: Optional[str]) -> Optional[str]:
        r"""Build datasheet URL if file exists.

        :param datasheet_file: Local PDF file path, may include data/pdfs/ prefix
        :return: URL path for the datasheet endpoint, or None if no file
        """
        if not datasheet_file:
            return None
        clean_path = datasheet_file.removeprefix("data/pdfs/")
        return f"/api/v1/datasheets/{clean_path}"

    def filter_cameras(
        self,
        vendor: Optional[str] = None,
        category: Optional[str] = None,
        series: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> Tuple[List[CameraSummaryResponse], int]:
        r"""Filter cameras with pagination.

        :param vendor: Filter by source/vendor
        :param category: Filter by camera category
        :param series: Filter by product series
        :param search: Full-text search in model_name, display_name
        :param page: Page number (1-indexed)
        :param limit: Items per page
        :return: Tuple of (list of CameraSummaryResponse, total count)
        """
        df = self.catalog

        filtered_df = df.copy()

        if vendor:
            filtered_df = filtered_df[filtered_df["source"] == vendor]

        if category:
            filtered_df = filtered_df[filtered_df["category"] == category]

        if series:
            filtered_df = filtered_df[filtered_df["product_series"] == series]

        if search:
            search_lower = search.lower()
            mask = filtered_df["model_name"].str.lower().str.contains(
                search_lower, na=False
            ) | filtered_df["display_name"].str.lower().str.contains(
                search_lower, na=False
            )
            filtered_df = filtered_df[mask]

        total = len(filtered_df)

        offset = (page - 1) * limit
        paginated_df = filtered_df.iloc[offset : offset + limit]

        cameras = []
        for _, row in paginated_df.iterrows():
            specs = {}
            if row.get("specifications"):
                try:
                    specs = json.loads(row["specifications"])
                except (json.JSONDecodeError, TypeError):
                    specs = {}

            image_files = []
            if row.get("image_files"):
                try:
                    image_files = json.loads(row["image_files"])
                except (json.JSONDecodeError, TypeError):
                    image_files = []

            thumbnail_url = ""
            if image_files:
                thumbnail_url = self._build_thumbnail_url(image_files[0])

            cameras.append(
                CameraSummaryResponse(
                    camera_id=row["camera_id"],
                    model_name=row["model_name"],
                    display_name=row["display_name"],
                    source=row["source"],
                    category=row["category"],
                    product_series=row.get("product_series", ""),
                    thumbnail_url=thumbnail_url,
                    specs=self._flatten_specs(specs),
                )
            )

        return cameras, total

    def get_camera_detail(self, camera_id: str) -> Optional[CameraDetailResponse]:
        r"""Get full camera details by ID.

        :param camera_id: Unique camera identifier
        :return: CameraDetailResponse if found, None otherwise
        """
        df = self.catalog
        row = df[df["camera_id"] == camera_id]

        if row.empty:
            return None

        record = self._row_to_record(row.iloc[0])

        image_urls = [url for url in record.images if url and url.startswith("http")]
        image_files = [self._build_thumbnail_url(f) for f in record.image_files]

        return CameraDetailResponse(
            camera_id=record.camera_id,
            model_name=record.model_name,
            display_name=record.display_name,
            description=record.description,
            source=record.source,
            category=record.category,
            product_series=record.product_series,
            image_urls=image_urls,
            image_files=image_files,
            datasheet_url=self._build_datasheet_url(record.datasheet_file),
            specs=self._flatten_specs(record.specifications),
        )

    def get_facets(self) -> Dict[str, Any]:
        r"""Get available filter options with counts.

        :return: Dictionary with vendors, categories, and series facets
        """
        df = self.catalog

        vendors = df.groupby("source").size().reset_index(name="count")
        vendor_facets = [
            FacetItem(value=row["source"], count=row["count"])
            for _, row in vendors.iterrows()
        ]

        categories = df.groupby("category").size().reset_index(name="count")
        category_facets = [
            FacetItem(value=row["category"], count=row["count"])
            for _, row in categories.iterrows()
        ]

        series_df = df.dropna(subset=["product_series"])
        series = series_df.groupby("product_series").size().reset_index(name="count")
        series_facets = [
            FacetItem(value=row["product_series"], count=row["count"])
            for _, row in series.iterrows()
        ]

        return {
            "vendors": vendor_facets,
            "categories": category_facets,
            "series": series_facets,
        }

    @staticmethod
    def load_catalog(self) -> pd.DataFrame:
        r"""Load catalog from parquet file.

        .. deprecated::
            Use :meth:`CatalogService.catalog` instead for proper lifecycle control.

        :return: DataFrame with camera records
        :raises FileNotFoundError: If dataset parquet file does not exist
        """
        service = CatalogService()
        return service.catalog

    @staticmethod
    def get_camera_by_id(camera_id: str) -> Optional[CameraRecord]:
        r"""Get a single camera record by ID.

        .. deprecated::
            Use :meth:`CatalogService.get_camera_detail` instead.

        :param camera_id: Unique camera identifier
        :return: CameraRecord if found, None otherwise
        """
        service = CatalogService()
        return service._row_to_record(
            service.catalog[service.catalog["camera_id"] == camera_id].iloc[0]
        )

    @staticmethod
    def flatten_specs(specs: Dict[str, Any]) -> Optional[CameraSpecsSummary]:
        r"""Extract commonly-used fields from specifications dict.

        .. deprecated::
            Use :meth:`CatalogService._flatten_specs` instead.

        :param specs: Full specifications dictionary
        :return: CameraSpecsSummary with flattened fields, None if specs is empty
        """
        service = CatalogService()
        return service._flatten_specs(specs)
