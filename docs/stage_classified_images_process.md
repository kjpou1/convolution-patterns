# 🧩 Stage Classified Images

This script organizes rendered chart images into clean, classification-based folders ready for model training and validation.

---

## 📂 **Input**

* **Rendered Data Directory**
  A directory containing:

  * Instrument subfolders (e.g., `AUD_CAD`, `GBP_JPY`)
  * Inside each instrument folder: date folders (e.g., `2025-06-01`)
  * Each date folder includes:

    * `manifest.csv` file with image metadata
    * Rendered `.png` images

**Example structure:**

```
artifacts/data/rendered/
└── AUD_CAD/
    ├── 2025-06-01/
    │   ├── manifest.csv
    │   ├── image_001.png
    │   └── ...
    └── 2025-06-02/
        └── ...
```

---

## ⚙️ **Process**

* Reads all manifest files for each instrument and date.
* Identifies the classification label for each image.
* Copies images into a *staging area*, grouping them by classification label.
* Automatically renames images in sequential order.
* Skips any records without valid labels.

---

## 📦 **Output**

* A clean folder structure under the staging directory, organized like this:

```
artifacts/staging/
└── AUD_CAD/
    ├── Trend_Change_Bull/
    │   ├── AUD_CAD_001.png
    │   ├── AUD_CAD_002.png
    │   └── ...
    ├── No_Pattern/
    └── Uptrend_Convergence/
```

* A printed summary showing:

  * Total images processed
  * Counts by classification
  * Skipped images and reasons

---

## ▶️ **How to Run**

**With Python:**

```bash
python scripts/stage_classified_images.py
```

**With Poetry:**

If you are using [Poetry](https://python-poetry.org/):

```bash
poetry run python scripts/stage_classified_images.py
```

This will process **all instruments** in your `artifacts/data/rendered` directory automatically.