from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InferenceResult:
    """
    Structured result for a single pattern detection inference.

    Attributes:
        label: Predicted class label (as integer).
        confidence: Model's confidence in the prediction (float, 0.0–1.0).
        window_size: Number of bars in the window that triggered detection.
        raw_probs: List of raw class probabilities output by the model.
        meta: Optional dictionary for extra metadata (e.g., detection time, window indices).
    """

    label: int
    confidence: float
    window_size: int
    raw_probs: List[float] = field(default_factory=list)
    meta: Optional[dict] = field(default_factory=dict)
