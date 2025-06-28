import os
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class SplitterService:
    """
    Service for stratified splitting of image metadata into train/validation/test sets.

    This service performs stratified splitting to ensure balanced class distribution
    across all splits, which is crucial for training robust machine learning models.
    It includes safety mechanisms to handle edge cases like classes with very few samples.

    Key Features:
    - Stratified splitting based on pattern_type to maintain class balance
    - Configurable split ratios with validation
    - Fallback to random splitting if stratification fails
    - Comprehensive logging and validation of split results
    - Safety checks for classes with insufficient samples

    Attributes:
        USE_STRATIFIED_SPLIT (bool): Global flag to enable/disable stratified splitting.
                                   Set to False for debugging or when stratification fails.
        train_ratio (int): Percentage of data allocated to training set
        val_ratio (int): Percentage of data allocated to validation set
        test_ratio (int): Percentage of data allocated to test set
        seed (int): Random seed for reproducible splits

    Example:
        >>> records = [
        ...     {'filename': 'img1.png', 'pattern_type': 'Uptrend'},
        ...     {'filename': 'img2.png', 'pattern_type': 'Downtrend'},
        ...     # ... more records
        ... ]
        >>> splitter = SplitterService(split_ratios=(70, 15, 15), seed=42)
        >>> splits = splitter.split(records)
        >>> train_df = splits['train']
        >>> val_df = splits['val']
        >>> test_df = splits['test']
    """

    # Global control for stratified splitting - can be toggled for debugging
    USE_STRATIFIED_SPLIT = True  # Set to False to disable stratification

    def __init__(
        self, split_ratios: Tuple[int, int, int] = (80, 10, 10), seed: int = 42
    ):
        """
        Initialize the SplitterService with specified ratios and random seed.

        Args:
            split_ratios (Tuple[int, int, int]): Tuple of (train%, val%, test%) ratios.
                                               Must sum to 100. Default is (80, 10, 10).
            seed (int): Random seed for reproducible splits. Default is 42.

        Raises:
            ValueError: If split ratios don't sum to 100.

        Example:
            >>> # 70% train, 20% val, 10% test
            >>> splitter = SplitterService(split_ratios=(70, 20, 10), seed=1337)
        """
        self.train_ratio, self.val_ratio, self.test_ratio = split_ratios
        self.seed = seed
        self._validate_ratios()

    def _validate_ratios(self) -> None:
        """
        Validate that split ratios sum to 100%.

        Raises:
            ValueError: If ratios don't sum to exactly 100.
        """
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total != 100:
            raise ValueError(
                f"Split ratios must sum to 100, got {total}. "
                f"Current ratios: train={self.train_ratio}%, "
                f"val={self.val_ratio}%, test={self.test_ratio}%"
            )

    def split(self, records: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Performs stratified splitting of records based on pattern_type column.

        This method ensures that each class (pattern_type) is proportionally represented
        in all three splits (train/val/test). If stratification fails due to insufficient
        samples in some classes, it falls back to random splitting with warnings.

        Args:
            records (List[Dict]): List of metadata dictionaries. Each dict must contain
                                at minimum a 'pattern_type' key for stratification.
                                Common keys include 'filename', 'pattern_type', 'path', etc.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing three DataFrames:
                - 'train': Training set DataFrame
                - 'val': Validation set DataFrame
                - 'test': Test set DataFrame
                Each DataFrame has reset indices and contains the same columns as input.

        Raises:
            ValueError: If no records are provided or records list is empty.
            KeyError: If 'pattern_type' column is missing from the data.

        Example:
            >>> records = [
            ...     {'filename': 'chart1.png', 'pattern_type': 'Uptrend', 'path': '/data/chart1.png'},
            ...     {'filename': 'chart2.png', 'pattern_type': 'Downtrend', 'path': '/data/chart2.png'},
            ... ]
            >>> splits = splitter.split(records)
            >>> print(f"Train: {len(splits['train'])} samples")
            >>> print(f"Val: {len(splits['val'])} samples")
            >>> print(f"Test: {len(splits['test'])} samples")
        """
        # Input validation
        if not records:
            raise ValueError("No records provided to splitter service.")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(records)

        # Ensure required column exists for stratification
        if "pattern_type" not in df.columns:
            raise KeyError(
                "Missing 'pattern_type' column for stratification. "
                "Available columns: " + ", ".join(df.columns.tolist())
            )

        # Analyze class distribution before splitting
        class_counts = df["pattern_type"].value_counts()
        min_samples = class_counts.min()
        total_samples = len(df)

        logging.info("Dataset analysis before splitting:")
        logging.info("  Total samples: %d", total_samples)
        logging.info("  Total classes: %d", len(class_counts))
        logging.info("  Class distribution:")

        for pattern, count in class_counts.items():
            percentage = (count / total_samples) * 100
            logging.info("    %s: %d samples (%.1f%%)", pattern, count, percentage)

        # Warn about potential stratification issues
        if min_samples < 3:
            logging.warning(
                "⚠️  Some classes have very few samples (minimum: %d). "
                "Stratified splitting may fail or produce unbalanced splits. "
                "Consider collecting more data or adjusting split ratios.",
                min_samples,
            )

        # Calculate expected samples per split for smallest class
        expected_val_samples = (min_samples * self.val_ratio) / 100
        expected_test_samples = (min_samples * self.test_ratio) / 100

        if expected_val_samples < 1 or expected_test_samples < 1:
            logging.warning(
                "⚠️  Smallest class may not appear in all splits. "
                "Expected samples in val: %.1f, test: %.1f",
                expected_val_samples,
                expected_test_samples,
            )

        try:
            # First split: separate training from (validation + test)
            train_df, temp_df = train_test_split(
                df,
                test_size=(self.val_ratio + self.test_ratio) / 100.0,
                stratify=df["pattern_type"] if self.USE_STRATIFIED_SPLIT else None,
                random_state=self.seed,
            )

            # Second split: separate validation from test
            # Calculate relative ratio for val within the temp set
            val_rel_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_rel_ratio,  # test_size is what remains after val
                stratify=temp_df["pattern_type"] if self.USE_STRATIFIED_SPLIT else None,
                random_state=self.seed,
            )

            logging.info("✅ Stratified splitting completed successfully")

        except ValueError as e:
            # Handle stratification failures (usually due to insufficient samples)
            if "stratify" in str(e).lower():
                logging.warning("⚠️  Stratified split failed: %s", str(e))
                logging.warning(
                    "🔄 Falling back to random (non-stratified) splitting..."
                )

                # Fallback to non-stratified split
                train_df, temp_df = train_test_split(
                    df,
                    test_size=(self.val_ratio + self.test_ratio) / 100.0,
                    random_state=self.seed,
                )

                val_rel_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=1 - val_rel_ratio,
                    random_state=self.seed,
                )

                logging.info("✅ Random splitting completed as fallback")
            else:
                # Re-raise if it's a different type of ValueError
                raise

        # Post-split validation and analysis
        self._validate_split_results(train_df, val_df, test_df, df)

        # Return splits with reset indices for clean DataFrames
        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

    def _validate_split_results(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        original_df: pd.DataFrame,
    ) -> None:
        """
        Validate and log the results of the splitting operation.

        This method checks for class consistency across splits and provides
        detailed logging about the final distribution.

        Args:
            train_df (pd.DataFrame): Training set DataFrame
            val_df (pd.DataFrame): Validation set DataFrame
            test_df (pd.DataFrame): Test set DataFrame
            original_df (pd.DataFrame): Original dataset before splitting
        """
        # Get unique classes from each split
        train_classes = set(train_df["pattern_type"].unique())
        val_classes = set(val_df["pattern_type"].unique())
        test_classes = set(test_df["pattern_type"].unique())
        all_original_classes = set(original_df["pattern_type"].unique())

        # Check for missing classes in each split
        missing_from_train = all_original_classes - train_classes
        missing_from_val = all_original_classes - val_classes
        missing_from_test = all_original_classes - test_classes

        # Log warnings for missing classes
        if missing_from_train:
            logging.warning(
                "❌ Train set missing classes: %s", sorted(missing_from_train)
            )
        if missing_from_val:
            logging.warning(
                "❌ Validation set missing classes: %s", sorted(missing_from_val)
            )
        if missing_from_test:
            logging.warning(
                "❌ Test set missing classes: %s", sorted(missing_from_test)
            )

        # Log successful split summary
        logging.info("📊 Final split summary:")
        logging.info(
            "  Train: %d samples (%d classes) - %.1f%%",
            len(train_df),
            len(train_classes),
            (len(train_df) / len(original_df)) * 100,
        )
        logging.info(
            "  Val:   %d samples (%d classes) - %.1f%%",
            len(val_df),
            len(val_classes),
            (len(val_df) / len(original_df)) * 100,
        )
        logging.info(
            "  Test:  %d samples (%d classes) - %.1f%%",
            len(test_df),
            len(test_classes),
            (len(test_df) / len(original_df)) * 100,
        )

        # Verify total samples match
        total_split_samples = len(train_df) + len(val_df) + len(test_df)
        if total_split_samples != len(original_df):
            logging.error(
                "❌ Sample count mismatch! Original: %d, Split total: %d",
                len(original_df),
                total_split_samples,
            )
        else:
            logging.info("✅ Sample counts verified - no data loss during splitting")

        # Check if all splits have the same classes (ideal scenario)
        if (
            len(train_classes)
            == len(val_classes)
            == len(test_classes)
            == len(all_original_classes)
        ):
            logging.info("✅ All splits contain all classes - optimal for training")
        else:
            logging.warning(
                "⚠️  Class distribution varies across splits. "
                "This may affect model training and evaluation."
            )
