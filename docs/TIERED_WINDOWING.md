# 📐 Tiered Sliding Window Inference Strategy

This document outlines the core design of the **Tiered Sliding Window Inference Strategy** used for localized pattern detection in CNN-based chart classifiers.

---

## 🎯 Objective

To classify chart patterns **anchored to the latest bar**, using a CNN trained on fixed-size images (e.g., 128×128), by dynamically sliding the historical context window while preserving inference accuracy.

---

## 📓 Reference Notebook

The logic and visual previews of this strategy are implemented in:

> 📁 [`notebook/notebooks/pattern_plot_preview.ipynb`](../notebook/notebooks/pattern_plot_preview.ipynb)

---

## 🗂️ Example Data

Example CSV with indicator data used to generate tiered image windows:

> 📄 [`examples/AUD_JPY_processed_signals.csv`](../examples/AUD_JPY_processed_signals.csv)

---

## 🪜 Sliding Window Logic

The approach begins with a **full-context window** and incrementally slides to smaller, more localized windows — always anchored to the **rightmost (most recent)** bar.

### ✅ Parameters

| Parameter          | Description                             | Example Value |
| ------------------ | --------------------------------------- | ------------- |
| `DEFAULT_NUM_DAYS` | Starting window size (max bars)         | `21`          |
| `ENDING_NUM_DAYS`  | Minimum window size (tightest view)     | `5`           |
| `WINDOW_STRIDE`    | Step size in bars when shrinking window | `2`           |
| `IMAGE_SIZE`       | Output image resolution for CNN input   | `(128, 128)`  |

---

## 🔁 Inference Workflow

1. **Load signal data** from CSV
2. **Generate tiered windows**: bar counts from `21 → 5` with `stride = 2`
3. For each window:

   * Slice `N` most recent bars
   * Plot and convert to image (128×128)
   * Pass to CNN model
   * Record `label` and `confidence`
   * Stop early if confidence ≥ threshold

---

## 📊 Why Right-Anchored?

By always anchoring to the most recent bar:

* You ensure relevance to current market state
* Older context is removed first
* Local patterns (e.g., spikes, crossovers) are preserved

Each smaller window provides a **zoomed-in perspective**, potentially boosting model confidence where the full context may dilute signal clarity.

---

## 🖼️ Visual Examples

| Bars | Stride | Description            |
| ---- | ------ | ---------------------- |
| 21   | 0      | Full context window    |
| 17   | 4      | Mid-range zoom         |
| 11   | 10     | Highly localized view  |
| 5    | 16     | Final ultra-local view |

Color mapping:

* **LC (Convolution)**: `#4CAF50` (Green)
* **LP (Projection)**: `#FF9800` (Orange)
* **AHMA**: `#880E4F` (Deep Pink)
* **Close (optional)**: `#2196F3` (Blue)

---

## 📦 Output Example

```json
{
  "instrument": "AUD_JPY",
  "best_window_size": 13,
  "stride": 8,
  "label": "Trend_Reversal_Bullish",
  "confidence": 0.92
}
```

---

## 📁 Key Code Entry Points

* `generate_sliding_windows(df)`
* `plot_pattern_image_array(df)`
* `inspect_window((window_size, image))`

