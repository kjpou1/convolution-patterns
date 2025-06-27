# Pattern View Dashboard for Convolution Patterns

---

## 📑 Overview

The **Pattern View Dashboard** is an interactive web application to **visualize, inspect, and validate indicator chart patterns** used in your Convolution Patterns classification pipeline.

Unlike generic image debuggers, this dashboard is **purpose-built for chart pattern datasets**, enabling you to:

* Rapidly review batches of rendered pattern images.
* Confirm label correctness and consistency across pattern types.
* Verify transformations and augmentations applied to the charts.
* Filter samples by class.
* Zoom into individual pattern visualizations with detailed metadata.

This is essential for **ensuring data quality and correctness before training CNN-based pattern classifiers**.

---

## 🎯 Key Features

| Feature                    | Description                                                          |
| -------------------------- | -------------------------------------------------------------------- |
| **Split Selection**        | Choose between `train`, `val`, and `test` datasets.                  |
| **Mode Selection**         | Display Raw, Transformed, or Augmented patterns.                     |
| **Batch Navigation**       | Browse batches sequentially by index.                                |
| **Class Filtering**        | Focus on a single pattern class for targeted inspection.             |
| **Dynamic Config Reload**  | Reloads your `TransformConfig` YAML live without restarting the app. |
| **Interactive Thumbnails** | Click thumbnails to open enlarged views.                             |
| **Detailed Metadata**      | View pattern label, class index, shape, and min/max pixel values.    |

---

## 🚀 How to Run

You can launch the dashboard **with or without Poetry**, depending on your environment.

---

### 🎩 Method 1 – Using Poetry (Recommended)

From your project root directory:

```bash
poetry run python dash_apps/dash_debug_app.py
```

---

### 🐍 Method 2 – Without Poetry

First, activate your Python virtual environment (example shown for `venv`):

```bash
source .venv/bin/activate
```

Then run:

```bash
python dash_apps/dash_debug_app.py
```

---

### 🌐 Access the dashboard

Open your browser:

* **Local URL:** [http://localhost:8050](http://localhost:8050)
* **Network URL:** Shown in the console output (e.g., `http://<your-ip>:8050`)

---

**Example Commands**

**With Poetry:**

```bash
poetry run python dash_apps/dash_debug_app.py
```

**Without Poetry:**

```bash
source .venv/bin/activate
python dash_apps/dash_debug_app.py
```

---

## 🧭 Example Workflow

1. Launch the dashboard.
2. Set `Split = train`, `Mode = Raw`.
3. Browse batches to inspect raw pattern images.
4. Switch to `Transformed` mode to confirm preprocessing.
5. Switch to `Augmented` mode to validate augmentation consistency.
6. Use the **Filter by Class** dropdown to isolate a specific pattern type (e.g., `Downtrend_Convergence`).
7. Click any thumbnail to zoom and see metadata details.

---

## 🛠️ Troubleshooting

| Issue                    | Possible Cause                         | Solution                                     |
| ------------------------ | -------------------------------------- | -------------------------------------------- |
| **No dataset loaded**    | Invalid config paths or empty datasets | Check `Config()` paths and dataset presence. |
| **Failed to load batch** | Batch index out of range               | Use a smaller batch index.                   |
| **Blank images**         | Invalid transformation pipeline        | Verify `TransformConfig` settings.           |
| **Modal does not open**  | No image clicked                       | Ensure you click a thumbnail image.          |

---

## ✨ Extending the Dashboard

Possible enhancements:

* Add thumbnail size controls.
* Enable exporting images for reporting.
* Visualize additional pattern metadata (e.g., indicator parameters).
* Integrate annotation workflows.

