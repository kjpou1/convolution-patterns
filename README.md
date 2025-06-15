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
  - [🔍 Inference Strategy](#-inference-strategy)
  - [🎯 One-Time Local `.venv/` Setup (Optional)](#-one-time-local-venv-setup-optional)
  - [🧠 Why Use a Local `.venv/`?](#-why-use-a-local-venv)
  - [🚀 Getting Started](#-getting-started)
    - [1. Clone the Repo](#1-clone-the-repo)
    - [2. Install Dependencies via Poetry](#2-install-dependencies-via-poetry)
    - [3. Activate the Environment](#3-activate-the-environment)
  - [🖥️ CLI Usage](#️-cli-usage)
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
