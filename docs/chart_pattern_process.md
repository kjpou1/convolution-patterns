# 📈 `chart_pattern_processor.py` – Sliding Window Pattern Extractor

**BARS-Based Chart Pattern Data Preprocessor**  
Prepares structured CNN input data slices for indicator-based chart rendering.

---

- [📈 `chart_pattern_processor.py` – Sliding Window Pattern Extractor](#-chart_pattern_processorpy--sliding-window-pattern-extractor)
  - [🎯 Objective](#-objective)
  - [🧠 Why Bar-Based?](#-why-bar-based)
  - [📥 Input Requirements](#-input-requirements)
  - [🛠️ Key Features](#️-key-features)
  - [📤 Output Structure](#-output-structure)
  - [🚀 CLI Usage](#-cli-usage)
    - [🔧 Main Options](#-main-options)
  - [✅ Example](#-example)
    - [💡 This will:](#-this-will)
  - [📦 Sample Output Summary](#-sample-output-summary)
  - [🖼 Sample Render Script (Bash)](#-sample-render-script-bash)
  - [🖼 Render Script Details](#-render-script-details)
    - [🧩 Purpose](#-purpose)
    - [🛠️ Anatomy of a Render Call](#️-anatomy-of-a-render-call)
    - [✅ Safety + Logging](#-safety--logging)
    - [📌 Output Example](#-output-example)
    - [🧾 Integration Summary](#-integration-summary)
  - [📎 Integration](#-integration)
  - [🧾 Best Practices](#-best-practices)
  - [📅 Last Updated](#-last-updated)


---

## 🎯 Objective

This script prepares financial indicator data (like LC, LP, AHMA) into **sliding window slices** for visual pattern classification.  
Instead of using fixed date intervals, it slides based on **bars (rows)** for consistency across varying timeframes.

It generates:
- 📄 `.json` files of minimal data for each image window
- 📦 `manifest.json` files to describe metadata per window
- 🖼 `render_batch.sh` script to call `patterncli render-images`
- 🧾 Full `render_calls_summary.json` for reproducibility

---

## 🧠 Why Bar-Based?

- Dates are inconsistent (weekends, holidays)
- Bar counts reflect actual **data availability**
- Aligns with CNN expectations of fixed-length spatial inputs

---

## 📥 Input Requirements

Input: CSV with the following columns:

| Column                | Purpose                    |
| --------------------- | -------------------------- |
| `Date`                | Bar timestamp              |
| `Close`               | Close price                |
| `AHMA`                | Adaptive Hull Moving Avg   |
| `Leavitt_Convolution` | Smoothed LC trend line     |
| `Leavitt_Projection`  | Forward-looking projection |

---

## 🛠️ Key Features

- ✅ **Sliding windows** in bar units (`--window-bars`, `--stride-bars`)
- ✅ Auto instrument detection from filename
- ✅ Filtering via `--start-date` and `--end-date`
- ✅ Compatible with `patterncli render-images`
- ✅ Optional `--dry-run` and `--verbose` modes
- ✅ Summary `.json` and batch `.sh` script generation

---

## 📤 Output Structure

Each window produces:

```

artifacts/data\_windows/{instrument}/{YYYY-MM-DD}/
├── data\_<start>\_<end>.json
├── manifest.json

```

Global outputs:

```

artifacts/scripts/{instrument}\_render.sh
artifacts/scripts/{instrument}\_summary.json

````

---

## 🚀 CLI Usage

```bash
poetry run python scripts/chart_pattern_processor.py <INPUT_CSV> [OPTIONS]
````

### 🔧 Main Options

| Option                  | Description                       |
| ----------------------- | --------------------------------- |
| `--window-bars`         | Bars per window (default: 30)     |
| `--stride-bars`         | Bars between windows (default: 5) |
| `--min-bars`            | Bars required to proceed          |
| `--start-date`          | Earliest date to consider         |
| `--end-date`            | Latest date to consider           |
| `--script-name`         | Where to save render `.sh`        |
| `--summary-file`        | Where to save JSON summary        |
| `--instrument-override` | Skip filename parsing             |
| `--dry-run`             | Preview only, no files written    |
| `--verbose`             | Enable debug logging              |

---

## ✅ Example

```bash
poetry run python scripts/chart_pattern_processor.py \
  ~/Public/projects/python/trading_strategies/convolution_strategy_ml/artifacts/preclose/CHF_JPY/CHF_JPY_processed_signals.csv \
  --script-name ./artifacts/scripts/CHF_JPY/chf_jpy_render.sh \
  --summary-file ./artifacts/scripts/CHF_JPY/chf_jpy_summary.json \
  --start-date 20241130 \
  --stride-bars 3
```

### 💡 This will:

* Use a **30-bar window** (default)
* Step back every **3 bars**
* Filter starting from **Nov 30, 2024**
* Generate **39 image windows**
* Write:

  * 🧾 `chf_jpy_summary.json`
  * 🖼 `chf_jpy_render.sh`
  * 💾 One JSON + manifest per window

---

## 📦 Sample Output Summary

```json
{
  "total_files_processed": 1,
  "total_windows_created": 39,
  "processed_files": [
    {
      "file": "...CHF_JPY_processed_signals.csv",
      "instrument": "CHF_JPY",
      "windows": 39
    }
  ],
  "configuration": {
    "window_bars": 30,
    "stride_bars": 3
  },
  ...
}
```

---

## 🖼 Sample Render Script (Bash)

```bash
poetry run patterncli render-images \
  --config configs/render_config.yaml \
  --input artifacts/data_windows/CHF_JPY/2025-06-23/data_20250513_20250623.json \
  --output-dir ./data/rendered/CHF_JPY/2025-06-23/ \
  --manifest ./data/rendered/CHF_JPY/2025-06-23/manifest.csv \
  --backend pil \
  --no-include-close
```

✔️ All 39 windows have their own CLI call
✔️ `set -e` ensures early exit on failure
✔️ Output directory structure: `data/rendered/{instrument}/{date}`

---

## 🖼 Render Script Details

The processor generates a Bash script (`render.sh`) that automates the rendering of chart pattern images for each sliding window.

---

### 🧩 Purpose

The render script ensures:
- 🖼 Each `.json` window is visualized via `patterncli render-images`
- 📂 Images are written to organized folders by instrument and date
- 📑 A `manifest.csv` is generated per window
- ⚠️ Failures are caught early (via `set -e` and error checks)

---

### 🛠️ Anatomy of a Render Call

Each window is rendered using:

```bash
poetry run patterncli render-images \
  --config configs/render_config.yaml \
  --input 'artifacts/data_windows/CHF_JPY/2025-06-23/data_20250513_20250623.json' \
  --output-dir './data/rendered/CHF_JPY/2025-06-23/' \
  --manifest './data/rendered/CHF_JPY/2025-06-23/manifest.csv' \
  --backend pil \
  --no-include-close
````

| Flag                 | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `--config`           | Controls image style, colors, DPI                        |
| `--input`            | Path to `.json` file for the window                      |
| `--output-dir`       | Where to save the PNG output                             |
| `--manifest`         | CSV metadata for rendered files                          |
| `--backend`          | Rendering engine (e.g., `pil`)                           |
| `--no-include-close` | Omits plotting the price close line (optional aesthetic) |

---

### ✅ Safety + Logging

Each render step is wrapped like this:

```bash
if [ $? -eq 0 ]; then
    echo 'Successfully processed window 2025-06-23'
else
    echo 'Error processing window 2025-06-23'
    exit 1
fi
```

* ✅ Echoes status to terminal
* ❌ Fails fast on errors (`exit 1`)

---

### 📌 Output Example

```bash
./data/rendered/CHF_JPY/2025-06-23/
├── pattern_0001.png
├── pattern_0002.png
├── ...
├── manifest.csv
```

Each folder corresponds to one window and is named by its ending date.

---

### 🧾 Integration Summary

| File                 | Purpose                     |
| -------------------- | --------------------------- |
| `render.sh`          | Automates rendering process |
| `render_config.yaml` | Controls visual styling     |
| `manifest.csv`       | Output index per image      |
| `summary.json`       | Global execution log        |
| `.json` (input)      | Data per window (30 bars)   |

---

> 📌 To run the script:
>
> ```bash
> chmod +x ./artifacts/scripts/CHF_JPY/chf_jpy_render.sh
> ./artifacts/scripts/CHF_JPY/chf_jpy_render.sh
> ```

---


## 📎 Integration

| Component                  | Role                                    |
| -------------------------- | --------------------------------------- |
| `patterncli render-images` | Consumes each `.json` and renders image |
| `render_config.yaml`       | Controls layout/DPI/theme               |
| `CNN Classifier`           | Trained on rendered images              |
| `Dash App (Optional)`      | For pattern visualization               |

---

## 🧾 Best Practices

* Use `--dry-run` to confirm behavior before disk writes
* Keep script and summary under `artifacts/scripts/`
* Include `summary.json` in ML pipeline logs
* Use consistent `window-bars` across instruments

---

## 📅 Last Updated

**2025-06-26**

