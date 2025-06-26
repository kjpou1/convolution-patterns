### 🧠 Title: **Manual Chart Pattern Classification Interface**

---

### 📝 Description

This tool is a **Streamlit-based dashboard** for manually classifying indicator chart images into structured pattern categories (e.g., **trend reversals**, **pullbacks**, **convergence**, etc.). It is used as the second step after chart rendering, providing a supervised labeling interface for training and evaluation purposes.

---

### 📁 Directory Structure

```
artifacts/data/rendered/
└── AUD_JPY/
    └── 2025-06-25/
        ├── chart_00001.png
        ├── chart_00002.png
        ├── ...
        └── manifest.csv
```

Each subfolder corresponds to one instrument-date combination.
The `manifest.csv` must contain at least:

* `filename`: chart image file name
* `classification`: pattern label (auto-updated by the UI)
* `window_size`: (optional) used for image sorting in UI

---

### 🏷️ Supported Pattern Classes

| Label                      | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `Trend_Change_Bull`        | Bullish trend reversal                               |
| `Trend_Change_Bear`        | Bearish trend reversal                               |
| `CT_Uptrend`               | Continuation of uptrend                              |
| `CT_Downtrend`             | Continuation of downtrend                            |
| `PB_Uptrend`               | Pullback within uptrend                              |
| `PB_Downtrend`             | Pullback within downtrend                            |
| `Uptrend_Convergence`      | Uptrend with convergence after divergence            |
| `Downtrend_Convergence`    | Downtrend with convergence after divergence          |
| `Uptrend_No_Convergence`   | Uptrend without convergence                          |
| `Downtrend_No_Convergence` | Downtrend without convergence                        |
| `No_Pattern`               | No identifiable pattern (default/unclassified state) |

---

### 🚀 How to Run

#### ✅ With Poetry (Preferred)

From project root:

```bash
poetry run streamlit run dash_apps/dash_classify_app.py
```

---

#### ✅ Without Poetry (e.g., venv or pip install)

```bash
# (Optional) Activate virtual environment
source .venv/bin/activate

# Run the app directly
streamlit run dash_apps/dash_classify_app.py
```

---

### 💾 Output

* Classification labels are written **back to `manifest.csv`** in the corresponding `instrument/date` folder.
* Classification progress is tracked live via the Streamlit UI.
* Supports label clearing, reclassification, and image sorting by `window_size`.

---

### 🔧 Requirements

Install dependencies via Poetry or pip:

```bash
# Using Poetry
poetry install

# OR using pip
pip install streamlit pandas streamlit-image-select
```

---

### 🧪 Future Enhancements

* Preview auto-labels from CNN classifier
* Add hotkeys for faster labeling
* Add filtering for unclassified-only
* Visual overlays of LC/LP/AHMA
