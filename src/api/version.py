"""Application version resolution.

Kept in its own module so both the FastAPI app and individual routes
can import the version without circular imports.
"""

from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import tomllib


def get_app_version() -> str:
    """
    Resolve the app version from pyproject.toml.

    1) Prefer installed package metadata (works in Docker/prod when installed).
    2) Fallback to reading pyproject.toml when running from source.
    """
    package_name = (
        "camera-identification-and-research"  # matches [project].name in pyproject.toml
    )

    try:
        return pkg_version(package_name)
    except PackageNotFoundError:
        # Running from source without an installed distribution
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            return data["project"]["version"]
        except (
            FileNotFoundError,
            OSError,
            UnicodeDecodeError,
            tomllib.TOMLDecodeError,
            KeyError,
        ):
            return "0.0.0"


APP_VERSION = get_app_version()
