"""
Matplotlib-based chart rendering backend with multi-series support.
"""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .chart_render_service import (
    COLOR_AHMA,
    COLOR_CLOSE,
    COLOR_CONVOLUTION,
    COLOR_PROJECTION,
    DEFAULT_IMAGE_SIZE,
    ChartRenderService,
)


class MatplotlibRenderBackend(ChartRenderService):
    """
    Chart renderer using matplotlib for multi-series financial data visualization.

    Features:
    - Multi-series plotting with color-coded lines
    - High-quality matplotlib rendering
    - Configurable DPI and styling
    """

    def __init__(
        self,
        image_size=DEFAULT_IMAGE_SIZE,
        image_format="png",
        include_close=True,
        dpi=64,
        line_width=1.5,
    ):
        """
        Initialize the MatplotlibRenderBackend.

        Args:
            image_size (tuple): Output image size as (width, height).
            image_format (str): Image format for saving (e.g., 'png').
            include_close (bool): Whether to include Close price series.
            dpi (int): DPI for matplotlib figure rendering.
        """
        self.image_size = image_size
        self.image_format = image_format
        self.include_close = include_close
        self.dpi = dpi
        self.line_width = line_width

        # Color mapping for different series
        self.color_map = {
            "Close": COLOR_CLOSE,
            "AHMA": COLOR_AHMA,
            "Leavitt_Projection": COLOR_PROJECTION,
            "Leavitt_Convolution": COLOR_CONVOLUTION,
        }

    def render(self, window_data, **kwargs):
        """
        Render a multi-series chart image from data using matplotlib.

        Args:
            data (dict): Dictionary mapping series names to numpy arrays.
                        e.g., {"Close": np.array([...]), "AHMA": np.array([...])}
            **kwargs: Additional options (include_close, dpi, etc.)

        Returns:
            numpy.ndarray: The rendered chart as a normalized numpy array (0-1).
        """
        include_close = kwargs.get("include_close", self.include_close)
        dpi = kwargs.get("dpi", self.dpi)
        line_width = kwargs.get("line_width", self.line_width)

        # Create matplotlib figure with specified size and DPI
        width, height = self.image_size
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

        # Process each available series
        for series_name, color in self.color_map.items():
            if series_name in window_data:
                # Skip Close if not requested
                if series_name == "Close" and not include_close:
                    continue

                values = window_data[series_name]
                ax.plot(values, color=color, linewidth=line_width, label=series_name)

        # Style the plot
        ax.axis("off")
        plt.tight_layout(pad=0)

        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(
            buf, format=self.image_format, bbox_inches="tight", pad_inches=0, dpi=dpi
        )
        plt.close(fig)  # Important: close figure to free memory
        buf.seek(0)

        # Convert to PIL Image and then to numpy array
        img = Image.open(buf)
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img) / 255.0

        return img_array

    def render_to_pil_image(self, data, **kwargs):
        """
        Render and return as PIL Image instead of numpy array.

        Args:
            data (dict): Dictionary mapping series names to numpy arrays.
            **kwargs: Additional rendering options.

        Returns:
            PIL.Image: The rendered chart as a PIL Image.
        """
        img_array = self.render(data, **kwargs)
        # Convert back to PIL Image
        return Image.fromarray((img_array * 255).astype(np.uint8))

    def save(self, image, path):
        # image is a numpy array, possibly normalized
        img = (
            Image.fromarray((image * 255).astype(np.uint8))
            if image.max() <= 1.0
            else Image.fromarray(image.astype(np.uint8))
        )
        img.save(path)

    def save_chart(self, data, save_path, **kwargs):
        """
        Render and save chart directly to file.

        Args:
            data (dict): Dictionary mapping series names to numpy arrays.
            save_path (str): Path to save the image file.
            **kwargs: Additional rendering options.
        """
        pil_img = self.render_to_pil_image(data, **kwargs)
        pil_img.save(save_path)
