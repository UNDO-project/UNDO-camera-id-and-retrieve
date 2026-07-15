"""Settings for catalog matching and retrieval behaviour.

All flags default to current behaviour except ``crop_margin`` (0.05),
which is the one deliberate default-on change: detection crops gain a
small symmetric margin so CLIP sees the whole object. Setting it to
``0.0`` reproduces the previous zero-margin crops exactly.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MatchingSettings(BaseSettings):
    """Configuration for identification matching and retrieval."""

    model_config = SettingsConfigDict(
        env_prefix="CIDAR_MATCH_",
        case_sensitive=False,
    )

    # Fraction of bbox side added symmetrically around detection crops.
    crop_margin: float = 0.05

    # Catalogue-side augmentation: embed K mildly degraded variants per
    # reference image in addition to the original (WP3).
    augment_enabled: bool = False
    augment_k: int = 4

    # Subtract the catalogue mean from both sides before cosine (WP4).
    mean_center: bool = False

    @field_validator("crop_margin")
    @classmethod
    def _validate_crop_margin(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("crop_margin must be >= 0")
        return value

    @field_validator("augment_k")
    @classmethod
    def _validate_augment_k(cls, value: int) -> int:
        if not 1 <= value <= 8:
            raise ValueError("augment_k must be between 1 and 8")
        return value
