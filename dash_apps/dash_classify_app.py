import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_image_select import image_select

from convolution_patterns.config import Config

st.set_page_config(
    page_title="Chart Pattern Classification", page_icon="📈", layout="wide"
)

# Configure logging with lazy evaluation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()

# Constants
DATA_ROOT = config.RENDERED_IMAGES_DIR
CLASS_LABELS = [
    "Trend_Change_Bull",  # Bullish trend reversal
    "Trend_Change_Bear",  # Bearish trend reversal
    "CT_Uptrend",  # Continuation uptrend
    "CT_Downtrend",  # Continuation downtrend
    "PB_Uptrend",  # Pullback within an uptrend
    "PB_Downtrend",  # Pullback within a downtrend
    "Uptrend_Convergence",  # Uptrend: indicators first diverge, then converge
    "Downtrend_Convergence",  # Downtrend: indicators first diverge, then converge
    "Uptrend_No_Convergence",  # Uptrend: indicators do not converge
    "Downtrend_No_Convergence",  # Downtrend: indicators do not converge
    "No_Pattern",  # No clear pattern detected (optional)
]

DEFAULT_CLASSIFICATION = "No_Pattern"  # Default classification for unclassified images


def list_instruments(data_root):
    try:
        if not os.path.exists(data_root):
            logger.warning("Data root directory does not exist: %s", data_root)
            return []
        instruments = [
            f
            for f in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, f))
        ]
        logger.info("Found %d instruments", len(instruments))
        return sorted(instruments)
    except Exception as e:
        logger.error("Error listing instruments: %s", str(e))
        return []


def list_dates(instrument_path):
    try:
        if not os.path.exists(instrument_path):
            logger.warning("Instrument path does not exist: %s", instrument_path)
            return []
        dates = [
            f
            for f in os.listdir(instrument_path)
            if os.path.isdir(os.path.join(instrument_path, f))
        ]
        logger.info("Found %d dates for instrument", len(dates))
        return sorted(dates, reverse=True)
    except Exception as e:
        logger.error("Error listing dates: %s", str(e))
        return []


def list_images(date_path):
    try:
        if not os.path.exists(date_path):
            logger.warning("Date path does not exist: %s", date_path)
            return []
        images = [
            f
            for f in os.listdir(date_path)
            if f.endswith(".png") and f.startswith("chart_")
        ]
        logger.info("Found %d images in date directory", len(images))
        return sorted(images)
    except Exception as e:
        logger.error("Error listing images: %s", str(e))
        return []


def load_manifest(date_path):
    manifest_path = os.path.join(date_path, "manifest.csv")
    try:
        if not os.path.exists(manifest_path):
            logger.error("Manifest file not found: %s", manifest_path)
            return None, manifest_path
        df = pd.read_csv(manifest_path, encoding="utf-8")
        logger.info("Loaded manifest with %d rows", len(df))
        if "classification" not in df.columns:
            df["classification"] = ""
        df["classification"] = df["classification"].astype(str)
        return df, manifest_path
    except Exception as e:
        logger.error("Error loading manifest: %s", str(e))
        return None, manifest_path


def save_manifest(df, manifest_path):
    try:
        df.to_csv(manifest_path, index=False, encoding="utf-8")
        logger.info("Saved manifest to %s", manifest_path)
        return True
    except Exception as e:
        logger.error("Error saving manifest: %s", str(e))
        return False


def update_classification(df, filename, label):
    try:
        if "classification" not in df.columns:
            df["classification"] = ""
        df["classification"] = df["classification"].astype(str)
        mask = df["filename"] == filename
        if mask.any():
            df.loc[mask, "classification"] = label
            logger.info("Updated classification for %s to %s", filename, label)
            return df, True
        else:
            logger.warning("Filename not found in manifest: %s", filename)
            return df, False
    except Exception as e:
        logger.error("Error updating classification: %s", str(e))
        return df, False


def get_current_classification(df, filename):
    try:
        if "classification" not in df.columns:
            return "Unclassified"
        mask = df["filename"] == filename
        if mask.any():
            current = df.loc[mask, "classification"].iloc[0]
            return current if current and str(current).strip() else "Unclassified"
        else:
            return "Unclassified"
    except Exception as e:
        logger.error("Error getting current classification: %s", str(e))
        return "Unclassified"


def main():
    st.title("📈 Chart Pattern Classification")
    st.markdown("---")
    if not os.path.exists(DATA_ROOT):
        st.error(
            f"❌ Data directory '{DATA_ROOT}' not found. Please ensure the data is in the correct location."
        )
        st.stop()
    with st.sidebar:
        st.header("Navigation")
        instruments = list_instruments(DATA_ROOT)
        if not instruments:
            st.error("No instruments found in data directory.")
            st.stop()
        instrument = str(
            st.selectbox("📊 Select Instrument", instruments, key="instrument")
        )
        instrument_path = os.path.join(DATA_ROOT, instrument)
        dates = list_dates(instrument_path)
        if not dates:
            st.error(f"No dates found for instrument {instrument}.")
            st.stop()
        date = st.selectbox("📅 Select Date", dates, key="date")
        st.info(f"**Instrument:** {instrument}\n**Date:** {date}")

    date_path = os.path.join(DATA_ROOT, instrument, date)
    manifest, manifest_path = load_manifest(date_path)
    if manifest is None:
        st.error(f"❌ Could not load manifest file for {instrument}/{date}")
        st.stop()
    images = list_images(date_path)
    if not images:
        st.error(f"❌ No chart images found for {instrument}/{date}")
        st.stop()

    # --- Sorting Controls ---
    sort_order = st.radio(
        "Sort images by window size:",
        options=["Largest to Smallest", "Smallest to Largest"],
        index=0,
        horizontal=True,
    )
    ascending = sort_order == "Smallest to Largest"

    # Merge images with manifest and sort
    if "window_size" in manifest.columns:
        manifest_filtered = manifest[manifest["filename"].isin(images)].copy()
        manifest_sorted = manifest_filtered.sort_values(
            by="window_size", ascending=ascending
        )
        sorted_images = manifest_sorted["filename"].tolist()
    else:
        sorted_images = images

    # --- Image Navigation State ---
    if "img_idx" not in st.session_state:
        st.session_state.img_idx = 0
    if (
        "selected_image" not in st.session_state
        or st.session_state.selected_image not in sorted_images
    ):
        st.session_state.selected_image = sorted_images[0]
        st.session_state.img_idx = 0

    total_images = len(sorted_images)
    classified_count = sum(
        1
        for img in sorted_images
        if get_current_classification(manifest, img) != "Unclassified"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🖼️ Select Image")
        row = st.columns([6, 1, 1])  # Wide selectbox, then two arrow buttons

        # --- Image Navigation State ---
        if "img_idx" not in st.session_state:
            st.session_state.img_idx = 0

        # Navigation UI
        with row[0]:
            selected_idx = st.selectbox(
                "Choose a chart image:",
                range(len(sorted_images)),
                format_func=lambda i: sorted_images[i],
                index=st.session_state.img_idx,
                key="image_selectbox",
            )
            # Only update img_idx if selectbox changed
            if selected_idx != st.session_state.img_idx:
                st.session_state.img_idx = selected_idx

        with row[1]:
            if st.button("⬅️", key="prev_img"):
                st.session_state.img_idx = (st.session_state.img_idx - 1) % len(
                    sorted_images
                )
                st.rerun()

        with row[2]:
            if st.button("➡️", key="next_img"):
                st.session_state.img_idx = (st.session_state.img_idx + 1) % len(
                    sorted_images
                )
                st.rerun()

        # After all navigation, set selected_image from img_idx
        st.session_state.selected_image = sorted_images[st.session_state.img_idx]

        # For debugging
        logger.info(
            "img_idx: %d, selected_image: %s",
            st.session_state.img_idx,
            st.session_state.selected_image,
        )

        st.metric(
            "Progress",
            f"{classified_count}/{total_images}",
            f"{(classified_count/total_images*100):.1f}%",
        )

        st.subheader("🏷️ Classify This Pattern")
        current_class = get_current_classification(
            manifest, st.session_state.selected_image
        )
        default_index = CLASS_LABELS.index(DEFAULT_CLASSIFICATION)
        if current_class in CLASS_LABELS:
            default_index = CLASS_LABELS.index(current_class)
        with st.form(key="classification_form"):
            selected_label = st.radio(
                "Select the pattern classification:",
                CLASS_LABELS,
                index=default_index,
                key="classification",
            )
            col_submit, col_clear = st.columns(2)
            with col_submit:
                submit_button = st.form_submit_button(
                    "💾 Save Classification", type="primary"
                )
            with col_clear:
                clear_button = st.form_submit_button("🗑️ Clear Classification")
            if submit_button:
                updated_manifest, success = update_classification(
                    manifest, st.session_state.selected_image, selected_label
                )
                if success:
                    if save_manifest(updated_manifest, manifest_path):
                        st.success(
                            f"✅ Successfully classified '{st.session_state.selected_image}' as '{selected_label}'!"
                        )
                        st.rerun()
                    else:
                        st.error("❌ Failed to save classification to file.")
                else:
                    st.error(
                        "❌ Failed to update classification. Image not found in manifest."
                    )
            if clear_button:
                updated_manifest, success = update_classification(
                    manifest, st.session_state.selected_image, ""
                )
                if success:
                    if save_manifest(updated_manifest, manifest_path):
                        st.success(
                            f"✅ Cleared classification for '{st.session_state.selected_image}'!"
                        )
                        st.rerun()
                    else:
                        st.error("❌ Failed to save changes to file.")
                else:
                    st.error("❌ Failed to clear classification.")

    with col2:
        st.subheader("🔍 Image Preview")
        image_path = os.path.join(date_path, st.session_state.selected_image)
        if os.path.exists(image_path):
            st.image(
                image_path,
                caption=f"Filename: {st.session_state.selected_image}",
                use_container_width=True,
            )
            current_class = get_current_classification(
                manifest, st.session_state.selected_image
            )
            if current_class != "Unclassified":
                st.success(f"✅ Current Classification: **{current_class}**")
            else:
                st.warning("⏳ This image is unclassified")
        else:
            st.error(f"❌ Image file not found: {st.session_state.selected_image}")

    st.markdown("---")
    with st.expander("ℹ️ Application Info"):
        st.markdown(
            f"""
        **Data Directory:** `{DATA_ROOT}`  
        **Current Path:** `{date_path}`  
        **Total Images:** {len(sorted_images)}  
        **Classified:** {classified_count}  
        **Remaining:** {total_images - classified_count}
        """
        )


if __name__ == "__main__":
    main()
