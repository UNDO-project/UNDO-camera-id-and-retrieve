from pathlib import Path
from pydantic import field_validator, computed_field
from pydantic_settings import BaseSettings


class PathSettings(BaseSettings):
    """Configuration for file system paths."""

    project_root: Path = Path(__file__).parent.parent.parent
    data_dir_name: str = "data"
    images_dir_name: str = "images"
    pdfs_dir_name: str = "pdfs"
    output_dir_name: str = "output"
    models_dir_name: str = "model_weights"
    yolo_weights_filename: str = "yolov8_camera.pt"

    @computed_field
    @property
    def data_dir(self) -> Path:
        """Data directory path."""
        return self.project_root / self.data_dir_name

    @computed_field
    @property
    def images_dir(self) -> Path:
        """Images directory path."""
        return self.data_dir / self.images_dir_name

    @computed_field
    @property
    def pdfs_dir(self) -> Path:
        """PDFs directory path."""
        return self.data_dir / self.pdfs_dir_name

    @computed_field
    @property
    def output_dir(self) -> Path:
        """Output directory path."""
        return self.project_root / self.output_dir_name

    @computed_field
    @property
    def models_dir(self) -> Path:
        """Model weights directory path."""
        return self.project_root / self.models_dir_name

    @computed_field
    @property
    def yolo_camera_weights_default(self) -> Path:
        """Default YOLO camera weights path."""
        return self.models_dir / self.yolo_weights_filename

    @field_validator("project_root", mode="before")
    @classmethod
    def resolve_project_root(cls, v):
        """Resolve project root to absolute path."""
        if isinstance(v, str):
            v = Path(v)
        return v.resolve()

    def ensure_directories_exist(self) -> None:
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.pdfs_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    class ConfigDict:
        env_prefix = "CIDAR_PATH_"
        env_file = ".env"
        case_sensitive = False
