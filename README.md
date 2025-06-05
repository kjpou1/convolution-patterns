# 🧠 Convolution Patterns

**CNN-based chart pattern detection for the Convolution Strategy**  
This project uses deep learning to classify indicator-based chart patterns — such as trend reversals and LC Convergence — directly from image data. It supports both live prediction and offline pattern mining.

---

## 🎯 Objective

Train a Convolutional Neural Network (CNN) to recognize visual patterns formed by key indicators:
- **LC (Leavitt Convolution)** – Green
- **LP (Leavitt Projection)** – Orange
- **AHMA (Adaptive Hull Moving Average)** – Purple / Red shades

Target pattern types include:
- `Bullish_Trend_Reversal`
- `Bearish_Trend_Reversal`
- `LCC_Uptrend`
- `LCC_Downtrend`
- `None_Uptrend`
- `None_Downtrend`

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
└── labels.csv  # optional
````

* Images: 128×128 PNGs, no axis or annotations
* Label from folder name or `labels.csv`

---

## 🔧 Training Pipeline

Built using **TensorFlow/Keras** with transfer learning support (e.g., MobileNetV2, EfficientNet).
Includes:

* Augmentation (flips, zooms, noise)
* Stratified training/validation split
* Accuracy + macro F1 tracking

---

## 🔍 Inference Strategy

* **Live**: Use last 128×128 window from streaming chart
* **Offline**: Apply sliding window classification (stride = 32 px)
* Output = predicted pattern + confidence

---

## 🔄 Next Steps

* [ ] Finalize sample set (balanced per class)
* [ ] Normalize image resolution and format
* [ ] Add data loader and training loop
* [ ] Benchmark backbones (MobileNet, EfficientNet, ResNet)

---

## 📜 License

MIT 
