"""Merge strategies for dataset append operations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.building.builder import DatasetBuilder


class MergeStrategy(ABC):
    """Abstract base class for merge strategies."""

    @abstractmethod
    def merge(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame,
        builder: "DatasetBuilder",
    ) -> pd.DataFrame:
        """
        Merge new records with existing dataset.

        :param new_df: DataFrame with new records
        :param existing_df: DataFrame with existing records
        :param builder: DatasetBuilder instance for updating stats
        :return: Merged DataFrame
        :raises ValueError: If merge cannot be completed
        """
        pass


class UpdateMergeStrategy(MergeStrategy):
    """
    Update merge strategy: keep last (new) record for duplicates.

    New records replace existing ones with the same camera_id.
    """

    def merge(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame,
        builder: "DatasetBuilder",
    ) -> pd.DataFrame:
        """
        Merge by updating existing records with new ones.

        :param new_df: DataFrame with new records
        :param existing_df: DataFrame with existing records
        :param builder: DatasetBuilder instance for updating stats
        :return: Merged DataFrame with new records taking precedence
        """
        # Compute statistics
        existing_ids = set(existing_df["camera_id"])
        new_ids = set(new_df["camera_id"])
        duplicate_ids = existing_ids & new_ids

        # Concatenate and keep last (new) record for duplicates
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["camera_id"], keep="last")

        # Update statistics
        builder.merge_stats["records_added"] = len(new_ids - existing_ids)
        builder.merge_stats["records_updated"] = len(duplicate_ids)

        logger.info(
            f"Merge strategy 'update': {builder.merge_stats['records_updated']} "
            f"records updated, {builder.merge_stats['records_added']} records added"
        )

        return merged.reset_index(drop=True)


class SkipMergeStrategy(MergeStrategy):
    """
    Skip merge strategy: keep first (existing) record for duplicates.

    Existing records are preserved, new duplicates are skipped.
    """

    def merge(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame,
        builder: "DatasetBuilder",
    ) -> pd.DataFrame:
        """
        Merge by skipping new records that already exist.

        :param new_df: DataFrame with new records
        :param existing_df: DataFrame with existing records
        :param builder: DatasetBuilder instance for updating stats
        :return: Merged DataFrame with existing records preserved
        """
        # Compute statistics
        existing_ids = set(existing_df["camera_id"])
        new_ids = set(new_df["camera_id"])
        duplicate_ids = existing_ids & new_ids

        # Concatenate and keep first (existing) record for duplicates
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["camera_id"], keep="first")

        # Update statistics
        builder.merge_stats["records_added"] = len(new_ids - existing_ids)
        builder.merge_stats["records_skipped"] = len(duplicate_ids)

        logger.info(
            f"Merge strategy 'skip': {builder.merge_stats['records_skipped']} "
            f"records skipped, {builder.merge_stats['records_added']} records added"
        )

        return merged.reset_index(drop=True)


class ErrorMergeStrategy(MergeStrategy):
    """
    Error merge strategy: raise error if duplicates found.

    This strategy enforces no duplicates between new and existing records.
    """

    def merge(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame,
        builder: "DatasetBuilder",
    ) -> pd.DataFrame:
        """
        Merge by raising error if any duplicates exist.

        :param new_df: DataFrame with new records
        :param existing_df: DataFrame with existing records
        :param builder: DatasetBuilder instance for updating stats
        :return: Merged DataFrame (if no duplicates)
        :raises ValueError: If duplicate camera_ids are found
        """
        # Compute statistics
        existing_ids = set(existing_df["camera_id"])
        new_ids = set(new_df["camera_id"])
        duplicate_ids = existing_ids & new_ids

        # Raise error if duplicates found
        if duplicate_ids:
            logger.error(f"Found {len(duplicate_ids)} duplicate camera_ids")
            logger.error(f"Duplicate IDs: {sorted(list(duplicate_ids)[:10])}")
            if len(duplicate_ids) > 10:
                logger.error(f"... and {len(duplicate_ids) - 10} more")
            raise ValueError(
                f"Duplicate camera_ids found: {len(duplicate_ids)} duplicates. "
                "Use --merge-strategy update or skip to handle duplicates."
            )

        # No duplicates, just concatenate
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        builder.merge_stats["records_added"] = len(new_df)

        logger.info(
            f"Merge strategy 'error': {builder.merge_stats['records_added']} records added"
        )

        return merged.reset_index(drop=True)


class MergeStrategyFactory:
    """Factory for creating merge strategy instances."""

    _strategies: dict[str, type[MergeStrategy]] = {
        "update": UpdateMergeStrategy,
        "skip": SkipMergeStrategy,
        "error": ErrorMergeStrategy,
    }

    @classmethod
    def get_strategy(cls, strategy_name: str) -> MergeStrategy:
        """
        Get merge strategy instance by name.

        :param strategy_name: Name of the strategy (update, skip, error)
        :return: MergeStrategy instance
        :raises ValueError: If strategy name is invalid
        """
        strategy_class = cls._strategies.get(strategy_name)
        if strategy_class is None:
            valid_options = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown merge strategy: {strategy_name}. "
                f"Valid options: {valid_options}"
            )
        return strategy_class()

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type[MergeStrategy]) -> None:
        """
        Register a custom merge strategy.

        :param name: Strategy name
        :param strategy_class: MergeStrategy subclass
        """
        cls._strategies[name] = strategy_class
