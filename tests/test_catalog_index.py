"""Tests for catalog loading and indexing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.identification.catalog import (
    load_catalog,
    iter_reference_image_files,
    _parse_json_field,
    _row_to_camera_record,
)
from src.identification.index import CatalogIndex
from src.models.camera import CameraRecord


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a sample parquet file for testing."""
    data = {
        "camera_id": ["axis-m3027", "axis-m3028", "hik-ds2de"],
        "model_name": ["AXIS M3027-PVE", "AXIS M3028-PVE", "DS-2DE2A404IW-DE3"],
        "display_name": ["AXIS M3027-PVE", "AXIS M3028-PVE", "DS-2DE2A404IW-DE3"],
        "description": ["5 MP camera", "8 MP camera", "4 MP PTZ"],
        "image_urls": [
            json.dumps(["https://example.com/axis-m3027.jpg"]),
            json.dumps(
                [
                    "https://example.com/axis-m3028-1.jpg",
                    "https://example.com/axis-m3028-2.jpg",
                ]
            ),
            json.dumps([]),
        ],
        "image_files": [
            json.dumps(["data/images/axis-m3027/img1.jpg"]),
            json.dumps(
                ["data/images/axis-m3028/img1.jpg", "data/images/axis-m3028/img2.jpg"]
            ),
            json.dumps([]),
        ],
        "datasheet_url": [
            "https://example.com/axis-m3027.pdf",
            None,
            "https://example.com/ds2de.pdf",
        ],
        "datasheet_file": ["data/pdfs/axis-m3027.pdf", None, None],
        "specifications": [
            json.dumps({"Resolution": {"value": "5 MP"}}),
            json.dumps({"Resolution": {"value": "8 MP"}}),
            "{}",
        ],
        "source": ["Axis Communications", "Axis Communications", "HikVision"],
        "category": ["Network Camera", "Network Camera", "PTZ Camera"],
        "product_category": ["Network Cameras", "Network Cameras", "PTZ Cameras"],
        "product_series": ["AXIS M3000", "AXIS M3000", "DS-2DE"],
    }

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_products.parquet"
    df.to_parquet(parquet_path)

    return parquet_path


def test_load_catalog(sample_parquet: Path) -> None:
    """Test loading catalog from parquet file."""
    catalog = load_catalog(sample_parquet)

    assert len(catalog) == 3
    assert "axis-m3027" in catalog
    assert "axis-m3028" in catalog
    assert "hik-ds2de" in catalog

    # Check first camera record
    axis_cam = catalog["axis-m3027"]
    assert axis_cam.model_name == "AXIS M3027-PVE"
    assert axis_cam.source == "Axis Communications"
    assert axis_cam.category == "Network Camera"
    assert len(axis_cam.image_files) == 1
    assert axis_cam.datasheet_url == "https://example.com/axis-m3027.pdf"


def test_load_catalog_missing_file(tmp_path: Path) -> None:
    """Test loading catalog with non-existent parquet file."""
    nonexistent_path = tmp_path / "nonexistent.parquet"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_catalog(nonexistent_path)

    assert "not found" in str(exc_info.value)


def test_load_catalog_with_empty_camera_id(tmp_path: Path) -> None:
    """Test loading catalog skips rows with empty camera_id."""
    data = {
        "camera_id": ["valid-cam", "", None],
        "model_name": ["Valid Camera", "Invalid 1", "Invalid 2"],
        "display_name": ["Valid", "Invalid 1", "Invalid 2"],
        "description": ["desc", "desc", "desc"],
        "image_urls": ["[]", "[]", "[]"],
        "image_files": ["[]", "[]", "[]"],
        "datasheet_url": [None, None, None],
        "datasheet_file": [None, None, None],
        "specifications": ["{}", "{}", "{}"],
        "source": ["Test", "Test", "Test"],
        "category": ["Test", "Test", "Test"],
        "product_category": ["Test", "Test", "Test"],
        "product_series": ["Test", "Test", "Test"],
    }

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_invalid.parquet"
    df.to_parquet(parquet_path)

    catalog = load_catalog(parquet_path)

    # Should only load the valid camera
    assert len(catalog) == 1
    assert "valid-cam" in catalog


def test_parse_json_field_valid_json() -> None:
    """Test parsing valid JSON field."""
    result = _parse_json_field('["a", "b", "c"]', default=[])
    assert result == ["a", "b", "c"]

    result = _parse_json_field('{"key": "value"}', default={})
    assert result == {"key": "value"}


def test_parse_json_field_invalid_json() -> None:
    """Test parsing invalid JSON returns default."""
    result = _parse_json_field("not json", default=[])
    assert result == []

    result = _parse_json_field("{invalid}", default={})
    assert result == {}


def test_parse_json_field_none_value() -> None:
    """Test parsing None value returns default."""
    result = _parse_json_field(None, default=[])
    assert result == []

    result = _parse_json_field(pd.NA, default={})
    assert result == {}


def test_parse_json_field_empty_string() -> None:
    """Test parsing empty string returns default."""
    result = _parse_json_field("", default=[])
    assert result == []

    result = _parse_json_field("   ", default={})
    assert result == {}


def test_row_to_camera_record() -> None:
    """Test converting dataframe row to CameraRecord."""
    row = pd.Series(
        {
            "camera_id": "test-cam-123",
            "model_name": "Test Camera Model",
            "display_name": "Test Cam",
            "description": "A test camera",
            "image_urls": json.dumps(["https://example.com/img1.jpg"]),
            "image_files": json.dumps(["data/images/test.jpg"]),
            "datasheet_url": "https://example.com/datasheet.pdf",
            "datasheet_file": "data/pdfs/datasheet.pdf",
            "specifications": json.dumps({"key": {"subkey": "value"}}),
            "source": "Test Source",
            "category": "Test Category",
            "product_category": "Test Product Category",
            "product_series": "Test Series",
        }
    )

    record = _row_to_camera_record(row)

    assert isinstance(record, CameraRecord)
    assert record.camera_id == "test-cam-123"
    assert record.model_name == "Test Camera Model"
    assert record.source == "Test Source"
    assert len(record.images) == 1
    assert len(record.image_files) == 1


def test_iter_reference_image_files(sample_parquet: Path, tmp_path: Path) -> None:
    """Test iterating over reference image files."""
    # Create actual image files referenced in catalog
    image_dir = tmp_path / "data" / "images"

    axis_m3027_dir = image_dir / "axis-m3027"
    axis_m3027_dir.mkdir(parents=True)
    (axis_m3027_dir / "img1.jpg").write_text("fake image")

    axis_m3028_dir = image_dir / "axis-m3028"
    axis_m3028_dir.mkdir(parents=True)
    (axis_m3028_dir / "img1.jpg").write_text("fake image")
    (axis_m3028_dir / "img2.jpg").write_text("fake image")

    # Load catalog and update image paths to use tmp_path
    catalog = load_catalog(sample_parquet)

    # Update paths to point to tmp_path
    for camera_id in catalog:
        catalog[camera_id].image_files = [
            str(tmp_path / f) for f in catalog[camera_id].image_files
        ]

    # Iterate with max_images_per_camera=1
    results = list(iter_reference_image_files(catalog, max_images_per_camera=1))

    # Should get 1 image per camera (2 cameras have images)
    assert len(results) == 2

    camera_ids = [camera_id for camera_id, _ in results]
    assert "axis-m3027" in camera_ids
    assert "axis-m3028" in camera_ids

    # Iterate with max_images_per_camera=2
    results = list(iter_reference_image_files(catalog, max_images_per_camera=2))

    # axis-m3027 has 1 image, axis-m3028 has 2 images
    assert len(results) == 3


def test_iter_reference_image_files_missing_files(sample_parquet: Path) -> None:
    """Test iterating skips missing image files."""
    catalog = load_catalog(sample_parquet)

    # Image files don't exist, should skip them
    results = list(iter_reference_image_files(catalog, max_images_per_camera=5))

    # Should return empty since no files actually exist
    assert len(results) == 0


def test_catalog_index_initialization(tmp_path: Path) -> None:
    """Test CatalogIndex initialization."""
    # Create fake embeddings file
    embeddings = np.random.rand(10, 512).astype("float32")
    camera_ids = np.array([f"cam-{i}" for i in range(10)], dtype="U256")
    image_paths = np.array([f"/fake/path/img{i}.jpg" for i in range(10)], dtype="U1024")
    sources = np.array(["Test Source"] * 10, dtype="U128")

    embeddings_path = tmp_path / "test_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        camera_ids=camera_ids,
        image_paths=image_paths,
        sources=sources,
    )

    # Initialize index
    index = CatalogIndex(embeddings_path)

    assert index.embeddings is not None
    assert index.embeddings.shape == (10, 512)
    assert len(index.camera_ids) == 10
    assert len(index.image_paths) == 10
    assert len(index.sources) == 10


def test_catalog_index_search(tmp_path: Path) -> None:
    """Test CatalogIndex search functionality."""
    # Create fake embeddings
    embeddings = np.random.rand(20, 128).astype("float32")

    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    camera_ids = np.array([f"cam-{i}" for i in range(20)], dtype="U256")
    image_paths = np.array([f"/fake/img{i}.jpg" for i in range(20)], dtype="U1024")
    sources = np.array(["Source A"] * 10 + ["Source B"] * 10, dtype="U128")

    embeddings_path = tmp_path / "test_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        camera_ids=camera_ids,
        image_paths=image_paths,
        sources=sources,
    )

    index = CatalogIndex(embeddings_path)

    # Create a query vector (normalized)
    query = np.random.rand(128).astype("float32")
    query = query / np.linalg.norm(query)

    # Search with top_k=5
    matches = index.search(query, top_k=5)

    assert len(matches) == 5
    assert all(hasattr(m, "camera_id") for m in matches)
    assert all(hasattr(m, "score") for m in matches)
    assert all(hasattr(m, "catalog_image_path") for m in matches)
    assert all(hasattr(m, "source") for m in matches)

    # Scores should be in descending order
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_catalog_index_search_top_k_larger_than_catalog(tmp_path: Path) -> None:
    """Test searching with top_k larger than catalog size."""
    embeddings = np.random.rand(5, 64).astype("float32")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    camera_ids = np.array([f"cam-{i}" for i in range(5)], dtype="U256")
    image_paths = np.array([f"/fake/img{i}.jpg" for i in range(5)], dtype="U1024")
    sources = np.array(["Test"] * 5, dtype="U128")

    embeddings_path = tmp_path / "small_catalog.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        camera_ids=camera_ids,
        image_paths=image_paths,
        sources=sources,
    )

    index = CatalogIndex(embeddings_path)

    query = np.random.rand(64).astype("float32")
    query = query / np.linalg.norm(query)

    # Search with top_k=10 but only 5 items in catalog
    matches = index.search(query, top_k=10)

    # Should return all 5 items
    assert len(matches) == 5


def test_build_catalog_embeddings_basic(
    sample_parquet: Path, tmp_path: Path, monkeypatch
) -> None:
    """Test building catalog embeddings from parquet file."""
    # Create actual image files
    image_dir = tmp_path / "data" / "images"

    axis_m3027_dir = image_dir / "axis-m3027"
    axis_m3027_dir.mkdir(parents=True)

    # Use PIL to create a real image
    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    img.save(axis_m3027_dir / "img1.jpg")

    # Update parquet to use tmp_path
    catalog = load_catalog(sample_parquet)
    for camera_id in catalog:
        catalog[camera_id].image_files = [
            str(tmp_path / f) for f in catalog[camera_id].image_files
        ]

    # Mock load_catalog to return our updated catalog
    monkeypatch.setattr("src.identification.index.load_catalog", lambda _: catalog)

    # Mock embed_image to return fake embeddings
    fake_embedding = np.random.rand(512).astype("float32")
    monkeypatch.setattr(
        "src.identification.index.embed_image", lambda _: fake_embedding
    )

    embeddings_path = tmp_path / "output" / "embeddings.npz"

    from src.identification.index import build_catalog_embeddings

    result_path = build_catalog_embeddings(
        parquet_path=sample_parquet,
        embeddings_path=embeddings_path,
        max_images_per_camera=1,
    )

    assert result_path == embeddings_path
    assert embeddings_path.exists()

    # Load and verify embeddings file
    data = np.load(embeddings_path)
    assert "embeddings" in data
    assert "camera_ids" in data
    assert "image_paths" in data
    assert "sources" in data


def _save_index_npz(tmp_path: Path, name: str, **arrays) -> Path:
    """Save an npz index artifact and return its path."""
    path = tmp_path / name
    np.savez_compressed(path, **arrays)
    return path


def test_search_collapses_duplicate_camera_ids(tmp_path: Path) -> None:
    """A product with several rows appears once, with its best score."""
    # cam-a has two rows; the second is the exact query (score 1.0)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # cam-a, variant far from query
            [0.0, 1.0, 0.0, 0.0],  # cam-a, variant equal to query
            [0.0, 0.0, 1.0, 0.0],  # cam-b
        ],
        dtype="float32",
    )
    embeddings_path = _save_index_npz(
        tmp_path,
        "dupes.npz",
        embeddings=embeddings,
        camera_ids=np.array(["cam-a", "cam-a", "cam-b"], dtype="U256"),
        image_paths=np.array(["/a1.jpg", "/a2.jpg", "/b.jpg"], dtype="U1024"),
        sources=np.array(["Test"] * 3, dtype="U128"),
    )

    index = CatalogIndex(embeddings_path)
    query = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")

    matches = index.search(query, top_k=5)

    assert len(matches) == 2  # cam-a collapsed to one match
    assert matches[0].camera_id == "cam-a"
    assert matches[0].score == pytest.approx(1.0)
    # The best variant's row metadata is reported
    assert str(matches[0].catalog_image_path) == "/a2.jpg"
    camera_ids = [m.camera_id for m in matches]
    assert len(camera_ids) == len(set(camera_ids))


def test_search_never_returns_same_camera_twice(tmp_path: Path) -> None:
    """Even with many rows per product, top-k has unique camera_ids."""
    rng = np.random.default_rng(3)
    embeddings = rng.random((30, 16)).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    # 10 products x 3 variants each
    camera_ids = np.array(
        [f"cam-{i}" for i in range(10) for _ in range(3)], dtype="U256"
    )
    embeddings_path = _save_index_npz(
        tmp_path,
        "variants.npz",
        embeddings=embeddings,
        camera_ids=camera_ids,
        image_paths=np.array(["/x.jpg"] * 30, dtype="U1024"),
        sources=np.array(["Test"] * 30, dtype="U128"),
        variant_tags=np.array(["orig", "warp1", "blur1"] * 10, dtype="U64"),
    )

    index = CatalogIndex(embeddings_path)
    query = rng.random(16).astype("float32")

    matches = index.search(query, top_k=10)

    camera_ids_returned = [m.camera_id for m in matches]
    assert len(camera_ids_returned) == 10
    assert len(set(camera_ids_returned)) == 10


def test_index_loads_artifact_without_optional_keys(tmp_path: Path) -> None:
    """Artifacts built before variant_tags/mean_vector must keep loading."""
    embeddings = np.eye(4, dtype="float32")
    embeddings_path = _save_index_npz(
        tmp_path,
        "legacy.npz",
        embeddings=embeddings,
        camera_ids=np.array([f"cam-{i}" for i in range(4)], dtype="U256"),
        image_paths=np.array([f"/img{i}.jpg" for i in range(4)], dtype="U1024"),
        sources=np.array(["Test"] * 4, dtype="U128"),
    )

    index = CatalogIndex(embeddings_path)

    assert index.variant_tags is None
    assert index.mean_vector is None
    matches = index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), top_k=2)
    assert matches[0].camera_id == "cam-0"


def test_build_with_augmentation_produces_variant_rows(
    sample_parquet: Path, tmp_path: Path, monkeypatch
) -> None:
    """Augmented build yields (K+1) rows per image with valid tags."""
    from PIL import Image

    image_dir = tmp_path / "data" / "images" / "axis-m3027"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(100, 150, 200)).save(image_dir / "img1.jpg")

    catalog = load_catalog(sample_parquet)
    for camera_id in catalog:
        catalog[camera_id].image_files = [
            str(tmp_path / f) for f in catalog[camera_id].image_files
        ]
    monkeypatch.setattr("src.identification.index.load_catalog", lambda _: catalog)

    fake_embedding = np.random.rand(512).astype("float32")
    monkeypatch.setattr(
        "src.identification.index.embed_image", lambda _: fake_embedding
    )

    from src.identification.index import build_catalog_embeddings

    embeddings_path = tmp_path / "augmented.npz"
    build_catalog_embeddings(
        parquet_path=sample_parquet,
        embeddings_path=embeddings_path,
        augment=True,
        augment_k=4,
    )

    data = np.load(embeddings_path)
    # 1 camera with an existing image -> 1 orig + 4 variants
    assert data["embeddings"].shape[0] == 5
    assert "variant_tags" in data.files
    tags = data["variant_tags"].astype(str).tolist()
    assert tags[0] == "orig"
    assert len(tags) == 5
    assert all(tag for tag in tags)
    assert data["camera_ids"].astype(str).tolist() == ["axis-m3027"] * 5
    assert "mean_vector" in data.files


def test_build_with_flags_off_matches_legacy_layout(
    sample_parquet: Path, tmp_path: Path, monkeypatch
) -> None:
    """With augmentation off, rows and keys match the pre-augmentation build."""
    from PIL import Image

    image_dir = tmp_path / "data" / "images" / "axis-m3027"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(10, 20, 30)).save(image_dir / "img1.jpg")

    catalog = load_catalog(sample_parquet)
    for camera_id in catalog:
        catalog[camera_id].image_files = [
            str(tmp_path / f) for f in catalog[camera_id].image_files
        ]
    monkeypatch.setattr("src.identification.index.load_catalog", lambda _: catalog)

    fake_embedding = np.random.rand(512).astype("float32")
    monkeypatch.setattr(
        "src.identification.index.embed_image", lambda _: fake_embedding
    )

    from src.identification.index import build_catalog_embeddings

    embeddings_path = tmp_path / "plain.npz"
    build_catalog_embeddings(
        parquet_path=sample_parquet,
        embeddings_path=embeddings_path,
        augment=False,
    )

    data = np.load(embeddings_path)
    assert data["embeddings"].shape[0] == 1
    assert "variant_tags" not in data.files
    assert np.array_equal(data["embeddings"][0], fake_embedding)
    assert data["camera_ids"].astype(str).tolist() == ["axis-m3027"]


class TestMeanCentring:
    """Tests for optional query-time mean-centring (WP4)."""

    @pytest.fixture
    def artifact_with_mean(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(11)
        embeddings = rng.random((6, 8)).astype("float32")
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return _save_index_npz(
            tmp_path,
            "with_mean.npz",
            embeddings=embeddings,
            camera_ids=np.array([f"cam-{i}" for i in range(6)], dtype="U256"),
            image_paths=np.array([f"/img{i}.jpg" for i in range(6)], dtype="U1024"),
            sources=np.array(["Test"] * 6, dtype="U128"),
            mean_vector=embeddings.mean(axis=0).astype("float32"),
        )

    def test_flag_off_scores_identical_to_plain_dot_product(self, artifact_with_mean):
        """With centring off, scores are bit-identical to raw cosine."""
        index = CatalogIndex(artifact_with_mean, mean_center=False)
        query = np.random.default_rng(4).random(8).astype("float32")

        matches = index.search(query, top_k=6)

        expected = index.embeddings @ (query / np.linalg.norm(query))
        for match in matches:
            row = index.camera_ids.index(match.camera_id)
            assert match.score == float(expected[row])

    def test_flag_on_centres_and_renormalizes(self, artifact_with_mean):
        """With centring on, both sides are centred then re-unit-normed."""
        index = CatalogIndex(artifact_with_mean, mean_center=True)

        # Catalogue side: search matrix rows are unit-norm after centring
        norms = np.linalg.norm(index._search_matrix, axis=1)
        assert norms == pytest.approx(np.ones(6), abs=1e-5)

        # Query side: score equals the manually centred cosine
        query = np.random.default_rng(4).random(8).astype("float32")
        matches = index.search(query, top_k=1)

        centred_query = query - index.mean_vector
        centred_query /= np.linalg.norm(centred_query)
        expected = index._search_matrix @ centred_query
        assert matches[0].score == pytest.approx(float(expected.max()), abs=1e-6)

    def test_flag_on_without_mean_vector_degrades_gracefully(self, tmp_path):
        """Old artifacts without mean_vector behave as if centring were off."""
        embeddings = np.eye(3, dtype="float32")
        path = _save_index_npz(
            tmp_path,
            "no_mean.npz",
            embeddings=embeddings,
            camera_ids=np.array(["a", "b", "c"], dtype="U256"),
            image_paths=np.array(["/a", "/b", "/c"], dtype="U1024"),
            sources=np.array(["T"] * 3, dtype="U128"),
        )

        index = CatalogIndex(path, mean_center=True)
        query = np.array([1.0, 0.0, 0.0], dtype="float32")

        matches = index.search(query, top_k=1)

        assert index._center_active is False
        assert matches[0].camera_id == "a"
        assert matches[0].score == pytest.approx(1.0)


def test_build_catalog_embeddings_no_valid_images(
    sample_parquet: Path, tmp_path: Path, monkeypatch
) -> None:
    """Test building embeddings when no valid images exist."""
    # Mock load_catalog to return catalog with no image files
    catalog = load_catalog(sample_parquet)
    for camera_id in catalog:
        catalog[camera_id].image_files = []

    monkeypatch.setattr("src.identification.index.load_catalog", lambda _: catalog)

    embeddings_path = tmp_path / "embeddings.npz"

    from src.identification.index import build_catalog_embeddings

    with pytest.raises(RuntimeError) as exc_info:
        build_catalog_embeddings(
            parquet_path=sample_parquet,
            embeddings_path=embeddings_path,
        )

    assert "No embeddings were produced" in str(exc_info.value)
