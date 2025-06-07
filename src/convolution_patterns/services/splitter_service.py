import os
import pandas as pd
from sklearn.model_selection import train_test_split
from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class SplitterService:
    """
    Service for stratified splitting of image metadata into train/val/test.
    """

    def __init__(self, split_ratios=(80, 10, 10), seed=42):
        self.train_ratio, self.val_ratio, self.test_ratio = split_ratios
        self.seed = seed
        self._validate_ratios()

    def _validate_ratios(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total != 100:
            raise ValueError(f"Split ratios must sum to 100, got {total}")

    def split(self, records: list[dict]) -> dict[str, pd.DataFrame]:
        """
        Performs stratified splitting based on `pattern_type`.

        Parameters:
        - records: List of metadata dicts with keys like `filename`, `pattern_type`, etc.

        Returns:
        - A dictionary with 'train', 'val', and 'test' DataFrames.
        """
        if not records:
            raise ValueError("No records provided to splitter service.")

        df = pd.DataFrame(records)

        if "pattern_type" not in df.columns:
            raise KeyError("Missing 'pattern_type' column for stratification.")

        train_df, temp_df = train_test_split(
            df,
            test_size=(self.val_ratio + self.test_ratio) / 100.0,
            stratify=df["pattern_type"],
            random_state=self.seed,
        )
        val_rel_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_rel_ratio,
            stratify=temp_df["pattern_type"],
            random_state=self.seed,
        )

        logging.info("Split summary: %d train, %d val, %d test", len(train_df), len(val_df), len(test_df))
        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }
