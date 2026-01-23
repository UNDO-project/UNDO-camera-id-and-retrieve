"""Catalog service for loading and querying camera data."""

import json
import re
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
    def _find_value_recursive(data: Any, *key_patterns: str) -> Optional[str]:
        r"""Recursively search for value matching any key pattern.

        :param data: Dict or value to search
        :param key_patterns: List of key patterns to match (case-insensitive)
        :return: First matching value as string, or None
        """
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()
                for pattern in key_patterns:
                    pattern_lower = pattern.lower()
                    if pattern_lower in key_lower:
                        if isinstance(value, dict):
                            result = CatalogService._find_value_recursive(
                                value, *key_patterns
                            )
                            if result:
                                return result
                        elif isinstance(value, str) and value not in ["–", "-"]:
                            return value
                        elif not isinstance(value, dict):
                            return str(value)
                        return None

            for value in data.values():
                result = CatalogService._find_value_recursive(value, *key_patterns)
                if result:
                    return result
        return None

    @staticmethod
    def _extract_focal_length_from_combined(value: str) -> Optional[str]:
        r"""Extract focal length from combined 'Focal Length & FOV' string.

        :param value: Combined string like '2.8 mm, horizontal FOV 100.2°'
        :return: Focal length part only, or None if not found
        """
        if not value:
            return None

        match = re.search(
            r"[\d.,]+\s*mm(?:\s*to\s*[\d.,]+\s*mm)?", value, re.IGNORECASE
        )
        if match:
            return match.group(0).strip()
        return None

    @staticmethod
    def _extract_horizontal_fov_from_combined(value: str) -> Optional[str]:
        r"""Extract horizontal FOV from combined 'Focal Length & FOV' string.

        :param value: Combined string like '2.8 mm, horizontal FOV 100.2°'
        :return: Horizontal FOV part only, or None if not found
        """
        if not value:
            return None

        match = re.search(
            r"horizontal\s+(?:field\s+of\s+view\s*:?\s*)?([\d.,]+\s*°?\s*(?:to\s*[\d.,]+\s*°?)?)",
            value,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _flatten_specs(specs: Dict[str, Any]) -> Optional[CameraSpecsSummary]:
        r"""Extract commonly-used fields from specifications dict.

        :param specs: Full specifications dictionary
        :return: CameraSpecsSummary with flattened fields, None if specs is empty
        """
        if not specs:
            return None

        max_resolution = CatalogService._find_value_recursive(
            specs, "max resolution", "max. resolution", "resolution", "video resolution"
        )

        lens_value = CatalogService._find_value_recursive(
            specs, "focal length", "lens", "focal length & fov"
        )

        if lens_value and "fov" in lens_value.lower():
            extracted_focal = CatalogService._extract_focal_length_from_combined(
                lens_value
            )
            lens_value = extracted_focal if extracted_focal else lens_value

        fov_value = CatalogService._find_value_recursive(
            specs,
            "horizontal field of view",
            "horizontal fov",
            "fov",
            "field of view",
            "focal length & fov",
        )

        if fov_value and (
            "focal length" in fov_value.lower()
            or "field of view" in fov_value.lower()
            or "vertical" in fov_value.lower()
            or "diagonal" in fov_value.lower()
        ):
            extracted_fov = CatalogService._extract_horizontal_fov_from_combined(
                fov_value
            )
            fov_value = extracted_fov if extracted_fov else None

        return CameraSpecsSummary(
            max_resolution=max_resolution,
            lens=lens_value,
            horizontal_fov=fov_value,
        )

    @staticmethod
    def _build_thumbnail_url(image_file: str) -> str:
        r"""Build thumbnail URL for an image file.

        :param image_file: Local file path, may include data/images/ prefix
        :return: URL path for the image endpoint
        """
        clean_path = image_file.removeprefix("data/images/")
        return f"/api/v1/images/{clean_path}"

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
            datasheet_url=record.datasheet_url,
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
