"""Tests for merge strategies."""

import pandas as pd
import pytest

from src.building.merge_strategies import (
    ErrorMergeStrategy,
    MergeResult,
    MergeStrategyFactory,
    SkipMergeStrategy,
    UpdateMergeStrategy,
)


@pytest.fixture
def existing_df():
    return pd.DataFrame(
        {
            "camera_id": ["cam-001", "cam-002", "cam-003"],
            "model_name": ["Camera 1", "Camera 2", "Camera 3"],
        }
    )


@pytest.fixture
def new_df_no_duplicates():
    return pd.DataFrame(
        {
            "camera_id": ["cam-004", "cam-005"],
            "model_name": ["Camera 4", "Camera 5"],
        }
    )


@pytest.fixture
def new_df_with_duplicates():
    return pd.DataFrame(
        {
            "camera_id": ["cam-002", "cam-003", "cam-006"],
            "model_name": ["Updated Camera 2", "Updated Camera 3", "Camera 6"],
        }
    )


class TestUpdateMergeStrategy:
    """Tests for UpdateMergeStrategy."""

    def test_merge_no_duplicates(self, existing_df, new_df_no_duplicates):
        """Test merging when no duplicates exist."""
        strategy = UpdateMergeStrategy()
        result = strategy.merge(new_df_no_duplicates, existing_df)

        assert isinstance(result, MergeResult)
        assert len(result.dataframe) == 5
        assert result.records_added == 2
        assert result.records_updated == 0
        assert result.records_skipped == 0

    def test_merge_with_duplicates(self, existing_df, new_df_with_duplicates):
        """Test merging when duplicates exist - new records should replace old."""
        strategy = UpdateMergeStrategy()
        result = strategy.merge(new_df_with_duplicates, existing_df)

        assert len(result.dataframe) == 4
        cam_002 = result.dataframe[result.dataframe["camera_id"] == "cam-002"].iloc[0]
        assert cam_002["model_name"] == "Updated Camera 2"
        assert result.records_added == 1
        assert result.records_updated == 2


class TestSkipMergeStrategy:
    """Tests for SkipMergeStrategy."""

    def test_merge_no_duplicates(self, existing_df, new_df_no_duplicates):
        """Test merging when no duplicates exist."""
        strategy = SkipMergeStrategy()
        result = strategy.merge(new_df_no_duplicates, existing_df)

        assert len(result.dataframe) == 5
        assert result.records_added == 2
        assert result.records_skipped == 0

    def test_merge_with_duplicates(self, existing_df, new_df_with_duplicates):
        """Test merging when duplicates exist - old records should be kept."""
        strategy = SkipMergeStrategy()
        result = strategy.merge(new_df_with_duplicates, existing_df)

        assert len(result.dataframe) == 4
        cam_002 = result.dataframe[result.dataframe["camera_id"] == "cam-002"].iloc[0]
        assert cam_002["model_name"] == "Camera 2"
        assert result.records_added == 1
        assert result.records_skipped == 2


class TestErrorMergeStrategy:
    """Tests for ErrorMergeStrategy."""

    def test_merge_no_duplicates(self, existing_df, new_df_no_duplicates):
        """Test merging when no duplicates exist."""
        strategy = ErrorMergeStrategy()
        result = strategy.merge(new_df_no_duplicates, existing_df)

        assert len(result.dataframe) == 5
        assert result.records_added == 2

    def test_merge_with_duplicates_raises_error(
        self, existing_df, new_df_with_duplicates
    ):
        """Test that ValueError is raised when duplicates exist."""
        strategy = ErrorMergeStrategy()

        with pytest.raises(ValueError, match="Duplicate camera_ids found"):
            strategy.merge(new_df_with_duplicates, existing_df)


class TestMergeStrategyFactory:
    """Tests for MergeStrategyFactory."""

    def test_get_update_strategy(self):
        strategy = MergeStrategyFactory.get_strategy("update")
        assert isinstance(strategy, UpdateMergeStrategy)

    def test_get_skip_strategy(self):
        strategy = MergeStrategyFactory.get_strategy("skip")
        assert isinstance(strategy, SkipMergeStrategy)

    def test_get_error_strategy(self):
        strategy = MergeStrategyFactory.get_strategy("error")
        assert isinstance(strategy, ErrorMergeStrategy)

    def test_get_unknown_strategy_raises_error(self):
        with pytest.raises(ValueError, match="Unknown merge strategy"):
            MergeStrategyFactory.get_strategy("invalid")

    def test_register_custom_strategy(self, existing_df, new_df_no_duplicates):
        """Test registering a custom strategy."""

        class CustomStrategy(UpdateMergeStrategy):
            pass

        MergeStrategyFactory.register_strategy("custom_test", CustomStrategy)
        strategy = MergeStrategyFactory.get_strategy("custom_test")
        assert isinstance(strategy, CustomStrategy)

        result = strategy.merge(new_df_no_duplicates, existing_df)
        assert len(result.dataframe) == 5
