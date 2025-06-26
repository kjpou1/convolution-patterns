# 🧠 Convolution Patterns

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Poetry](https://img.shields.io/badge/package-poetry-8c52ff.svg)
![TensorFlow](https://img.shields.io/badge/framework-TensorFlow-orange.svg)


**CNN-based chart pattern detection for the Convolution Strategy**
This project applies deep learning to classify technical indicator-based chart patterns — such as **trend reversals** and **LC Convergence** — directly from image data. It supports both **live inference** and **offline pattern mining**.

---

- [🧠 Convolution Patterns](#-convolution-patterns)
  - [🎯 Objective](#-objective)
  - [🗂️ Project Structure](#️-project-structure)
  - [🔧 Training Pipeline](#-training-pipeline)
  - [📥 Ingestion Pipeline](#-ingestion-pipeline)
    - [✅ Key Features](#-key-features)
    - [🗂️ Output Structure](#️-output-structure)
    - [🖥️ CLI Usage](#️-cli-usage)
    - [🔧 CLI Arguments](#-cli-arguments)
    - [🧪 Label Modes](#-label-modes)
    - [📝 Example Ingestion Config](#-example-ingestion-config)
    - [📦 Next Steps](#-next-steps)
  - [🔍 Inference Strategy](#-inference-strategy)
  - [🎯 One-Time Local `.venv/` Setup (Optional)](#-one-time-local-venv-setup-optional)
  - [🧠 Why Use a Local `.venv/`?](#-why-use-a-local-venv)
  - [🚀 Getting Started](#-getting-started)
    - [1. Clone the Repo](#1-clone-the-repo)
    - [2. Install Dependencies via Poetry](#2-install-dependencies-via-poetry)
    - [3. Activate the Environment](#3-activate-the-environment)
  - [🖥️ CLI Usage](#️-cli-usage-1)
    - [Available Commands](#available-commands)
    - [Example Usage](#example-usage)
  - [📅 Roadmap](#-roadmap)
  - [📜 License](#-license)

---

## 🎯 Objective

Train a Convolutional Neural Network (CNN) to recognize visual formations based on three core indicators:

* **LC (Leavitt Convolution)** – Green
* **LP (Leavitt Projection)** – Orange
* **AHMA (Adaptive Hull Moving Average)** – Purple / Red shades

Target pattern types include:

* `Bullish_Trend_Reversal`
* `Bearish_Trend_Reversal`
* `LCC_Uptrend`
* `LCC_Downtrend`
* `None_Uptrend`
* `None_Downtrend`

---

## 🗂️ Project Structure

```bash
pattern_images/
├── Bullish_Trend_Reversal/
├── Bearish_Trend_Reversal/
├── LCC_Uptrend/
├── LCC_Downtrend/
├── None_Uptrend/
├── None_Downtrend/
└── labels.csv  # optional label override
```

* Images: **128×128** PNGs with no axes or annotations
* Labels inferred from folder names or `labels.csv`

---

## 🔧 Training Pipeline

Built with **TensorFlow/Keras** using **transfer learning** backbones (e.g., MobileNetV2, EfficientNet). Includes:

* On-the-fly image augmentation (flip, zoom, noise, etc.)
* Stratified train/val/test splitting
* Training metrics: **Accuracy** and **Macro F1 Score**

---

## 📥 Ingestion Pipeline

The ingestion pipeline prepares raw chart pattern images for model training and evaluation. It handles file organization, metadata tracking, stratified splits, and offline augmentations.

### ✅ Key Features

* **Raw Copying**: Snapshots original PNGs to `artifacts/data/raw/`
* **Reorganization**: Converts input into `instrument/pattern_type/filename.png` structure
* **Metadata Generation**: Saves `pattern_metadata.csv` with filename, label, split, etc.
* **Stratified Splits**: Automatically splits into train/val/test by label proportion
* **Disk-Based Augmentations**: Applies flips, noise, contrast, zoom, and shift — stored alongside originals

### 🗂️ Output Structure

```bash
artifacts/data/
├── raw/                         # Immutable copy of input
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    └── pattern_metadata.csv     # Includes filename, label, split, etc.
```

---

### 🖥️ CLI Usage

```bash
poetry run patterncli ingest [OPTIONS]
```

### 🔧 CLI Arguments

| Argument            | Type     | Default        | Description                                              |
| ------------------- | -------- | -------------- | -------------------------------------------------------- |
| `--staging-dir`     | `str`    | *(required\*)* | Path to the staging directory containing source images   |
| `--no-preserve-raw` | `flag`   | `False`        | Skip copying raw images to `artifacts/data/raw/`         |
| `--label-mode`      | `str`    | `pattern_only` | Labeling mode: `pattern_only` or `instrument_specific`   |
| `--split-ratios`    | `int[3]` | `[70, 15, 15]` | Train/val/test split ratios as percentages               |
| `--random-seed`     | `int`    | `42`           | Random seed for reproducibility                          |
| `--config`          | `str`    | `None`         | Optional path to a YAML config file (overrides CLI args) |
| `--debug`           | `flag`   | `False`        | Enable debug logging                                     |

> 💡 `--staging-dir` is **required** unless a `--config` YAML file is used.

---

### 🧪 Label Modes

* `pattern_only` – Labels are based solely on the pattern type (e.g., `Uptrend_Convergence`)
* `instrument_specific` – Includes the instrument in the label (e.g., `AUD_CHF_Uptrend_Convergence`)

---

### 📝 Example Ingestion Config

```yaml
# configs/ingest_config.yaml

staging_dir: ./artifacts/staging
preserve_raw: yes
label_mode: pattern_only
split_ratios: [80, 10, 10]
random_seed: 123
debug: true
```

Run it with:

```bash
poetry run patterncli ingest --config configs/ingest_config.yaml
```

---

### 📦 Next Steps

After ingestion, training can begin via:

```bash
poetry run patterncli train --config configs/train_config.yaml
```

---

## 🔍 Inference Strategy

* **Live Inference**: Use latest **128×128** window from real-time chart
* **Offline Mining**: Apply **sliding window** detection (stride = 32 px)
* Output: predicted pattern label + confidence score

---

## 🎯 One-Time Local `.venv/` Setup (Optional)

To create the virtual environment **inside the project folder** (i.e., `./.venv/`) for **this project only** — without affecting your global Poetry config — run:

```bash
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install
```

This will create:

```
./.venv/
```

> 📝 This is a one-time setup for this repo. You can still use `poetry run` or `poetry shell` as usual.

---

## 🧠 Why Use a Local `.venv/`?

* Keeps the virtual environment **self-contained** inside the repo
* Useful for collaboration, CI/CD, and reproducibility
* Easier to locate and activate: `source .venv/bin/activate`

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourname/convolution-patterns.git
cd convolution-patterns
```

### 2. Install Dependencies via Poetry

```bash
poetry install
```

> ✅ Optional: To create a local `.venv/` inside the project directory:
>
> ```bash
> POETRY_VIRTUALENVS_IN_PROJECT=true poetry install
> ```

### 3. Activate the Environment

```bash
poetry shell
```

> Or use `poetry run` to invoke commands directly.

---

## 🖥️ CLI Usage

Use the project CLI through the registered alias:

```bash
poetry run patterncli [COMMAND] [OPTIONS]
```

### Available Commands

| Command  | Description                              |
| -------- | ---------------------------------------- |
| `ingest` | Run data ingestion and metadata creation |
| `train`  | Train a CNN on processed image data      |
| `infer`  | Run batch inference using sliding window |
| `debug`  | Launch interactive Dash viewer           |

### Example Usage

```bash
poetry run patterncli train --config configs/train_config.yaml
```

> 📝 Tip: Run `poetry run patterncli --help` or append `--help` to any command for details.

---

## 📅 Roadmap

* [ ] Finalize sample set (balanced per pattern type)
* [ ] Normalize image resolution and color format
* [ ] Add data ingestion + training pipeline
* [ ] Benchmark pretrained backbones (MobileNet, EfficientNet, ResNet)

---

## 📜 License

MIT
