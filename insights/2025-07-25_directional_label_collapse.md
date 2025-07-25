# 🧠 Insight: Directional Label Collapse for CNN Pattern Classification

**Date**: 2025-07-21  
**Status**: Finalized  
**Supersedes**: `2025-07-xx_directional_label_conflict.md`

---

## 🔍 Problem

CNN classification performance degraded due to label inconsistency between visually identical chart patterns labeled as `Pullback` vs `Continuation`. The semantic difference depended on external trend context, but the visual structure was often the same (e.g., `LH + LL` or `HH + HL`).

This meant the model was being supervised with **conflicting labels** for **visually indistinct inputs**, leading to label noise and reduced generalization.

---

## ✅ Resolution Strategy

### 🎯 Collapse Labels Based on Implied Directional Structure

| Original Labels     | New Label                       |
| ------------------- | ------------------------------- |
| `PB_Downtrend`      | `Uptrend`                       |
| `CT_Uptrend`        | `Uptrend`                       |
| `PB_Uptrend`        | `Downtrend`                     |
| `CT_Downtrend`      | `Downtrend`                     |
| `Trend_Change_Bull` | `Trend_Change_Bull` (unchanged) |
| `Trend_Change_Bear` | `Trend_Change_Bear` (unchanged) |

This preserves the **directional intent** of the pattern while avoiding overloading the model with unobservable semantic distinctions.

---

## 🧠 Key Insight

> Label conflict was not caused by base/quote inversion, but by trying to teach a CNN to distinguish pullbacks from continuations when the visual cues are nearly identical and the deciding context lies outside the image window.

---

## 📈 Benefits

- ✅ **Clearer visual-to-label mapping** — visually similar patterns now share a label
- ✅ **Reduced label noise** — fewer false gradients and conflicting supervision
- ✅ **Improved generalization** — consistent labels across instruments and trends
- ✅ **Foundation for modular label design** — trend-phase logic can be reintroduced later

---

## 🧪 Next Steps

- Train model using `Uptrend`, `Downtrend`, `Trend_Change_Bull`, `Trend_Change_Bear`, `No_Pattern`
- Audit class confusion matrix pre/post remapping
- Consider reintroducing pullback/continuation semantics downstream if needed
- Evaluate whether base/quote effects persist — initial evidence says no

---

## 🗃️ Related Documents

- `2025-07-xx_directional_label_conflict.md` – Original (now deprecated) base/quote hypothesis
