from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageOps

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
    - Configurable image margin/padding
    """

    def __init__(
        self,
        image_size=DEFAULT_IMAGE_SIZE,
        image_format="png",
        include_close=True,
        dpi=64,
        line_width=1.5,
        image_margin=0,
    ):
        self.image_size = image_size
        self.image_format = image_format
        self.include_close = include_close
        self.dpi = dpi
        self.line_width = line_width
        self.image_margin = image_margin

        self.color_map = {
            "Close": COLOR_CLOSE,
            "AHMA": COLOR_AHMA,
            "Leavitt_Projection": COLOR_PROJECTION,
            "Leavitt_Convolution": COLOR_CONVOLUTION,
        }

    def render(self, window_data, **kwargs):
        include_close = kwargs.get("include_close", self.include_close)
        dpi = kwargs.get("dpi", self.dpi)
        line_width = kwargs.get("line_width", self.line_width)
        image_margin = kwargs.get("image_margin", self.image_margin)

        width, height = self.image_size

        # If margin, render content smaller and add margin after
        if image_margin > 0:
            content_width = width - 2 * image_margin
            content_height = height - 2 * image_margin
            if content_width <= 0 or content_height <= 0:
                raise ValueError(
                    "Image margin %d is too large for image size %s"
                    % (image_margin, self.image_size)
                )
        else:
            content_width, content_height = width, height

        fig, ax = plt.subplots(
            figsize=(content_width / dpi, content_height / dpi), dpi=dpi
        )

        for series_name, color in self.color_map.items():
            if series_name in window_data:
                if series_name == "Close" and not include_close:
                    continue
                values = window_data[series_name]
                ax.plot(values, color=color, linewidth=line_width, label=series_name)

        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Remove all padding and margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0, 0)
        plt.tight_layout(pad=0)

        buf = BytesIO()
        plt.savefig(
            buf, format=self.image_format, bbox_inches="tight", pad_inches=0, dpi=dpi
        )
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf)
        img = img.convert("RGB")  # Ensure consistent mode

        # If margin, add it with PIL
        if image_margin > 0:
            img = img.resize((content_width, content_height), Image.Resampling.LANCZOS)
            img = ImageOps.expand(img, border=image_margin, fill="white")
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        else:
            # Try to crop any remaining border if margin is zero
            bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()
            if bbox:
                img = img.crop(bbox)
            # Finally, resize to requested size (in case crop changed it)
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)

        img_array = np.array(img) / 255.0
        return img_array

    def render_to_pil_image(self, data, **kwargs):
        img_array = self.render(data, **kwargs)
        return Image.fromarray((img_array * 255).astype(np.uint8))

    def save(self, image, path):
        img = (
            Image.fromarray((image * 255).astype(np.uint8))
            if image.max() <= 1.0
            else Image.fromarray(image.astype(np.uint8))
        )
        img.save(path)

    def save_chart(self, data, save_path, **kwargs):
        pil_img = self.render_to_pil_image(data, **kwargs)
        pil_img.save(save_path)
