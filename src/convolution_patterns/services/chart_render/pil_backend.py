import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .chart_render_service import (
    COLOR_AHMA,
    COLOR_CLOSE,
    COLOR_CONVOLUTION,
    COLOR_PROJECTION,
    DEFAULT_IMAGE_SIZE,
    ChartRenderService,
)


def hex_to_rgb(hex_color):
    """Convert hex color string (e.g. '#2196F3') to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class PILRenderBackend(ChartRenderService):
    """
    Chart renderer using PIL with proper upscaling and anti-aliasing.
    """

    def __init__(
        self,
        image_size=DEFAULT_IMAGE_SIZE,
        image_format="png",
        include_close=True,
        upscale_factor=4,
        line_width=1,
        image_margin=0,
    ):
        """
        Initialize the PILRenderBackend.

        Args:
            image_size (tuple): Final output image size as (width, height).
            image_format (str): Image format for saving.
            include_close (bool): Whether to include Close price series.
            upscale_factor (int): Render at this multiple, then downscale for anti-aliasing.
            image_margin (int): Margin/padding in pixels to add around the chart.
        """
        self.image_size = image_size
        self.image_format = image_format
        self.include_close = include_close
        self.upscale_factor = upscale_factor
        self.line_width = line_width
        self.image_margin = image_margin

        # Content area for chart (subtract margin)
        if image_margin > 0:
            self.content_size = (
                image_size[0] - 2 * image_margin,
                image_size[1] - 2 * image_margin,
            )
            if self.content_size[0] <= 0 or self.content_size[1] <= 0:
                raise ValueError(
                    "Image margin %d is too large for image size %s"
                    % (image_margin, image_size)
                )
        else:
            self.content_size = image_size

        self.render_size = (
            self.content_size[0] * upscale_factor,
            self.content_size[1] * upscale_factor,
        )

        # Color mapping for different series (convert hex to RGB)
        self.color_map = {
            "Close": hex_to_rgb(COLOR_CLOSE),
            "AHMA": hex_to_rgb(COLOR_AHMA),
            "Leavitt_Projection": hex_to_rgb(COLOR_PROJECTION),
            "Leavitt_Convolution": hex_to_rgb(COLOR_CONVOLUTION),
        }

    def render(self, window_data, **kwargs):
        include_close = kwargs.get("include_close", self.include_close)
        line_width = kwargs.get("line_width", self.line_width)
        image_margin = kwargs.get("image_margin", self.image_margin)

        # Use content area for chart
        render_size = (
            self.content_size[0] * self.upscale_factor,
            self.content_size[1] * self.upscale_factor,
        )

        img = Image.new("RGB", render_size, "white")
        draw = ImageDraw.Draw(img)

        # Filter and order series
        series_order = ["Close", "AHMA", "Leavitt_Projection", "Leavitt_Convolution"]
        series_to_plot = []
        for series_name in series_order:
            if series_name not in window_data:
                continue
            if series_name == "Close" and not include_close:
                continue
            series_to_plot.append(series_name)

        if not series_to_plot:
            # Return blank image if no series to plot
            final_img = img.resize(self.content_size, Image.LANCZOS)
            if image_margin > 0:
                final_img = ImageOps.expand(
                    final_img, border=image_margin, fill="white"
                )
            return final_img.resize(self.image_size, Image.LANCZOS)

        # Get all data values for global min/max
        all_values = np.concatenate([window_data[name] for name in series_to_plot])
        global_min, global_max = all_values.min(), all_values.max()
        value_range = global_max - global_min + 1e-8

        width, height = render_size

        # Plot each series
        for series_name in series_to_plot:
            values = window_data[series_name]
            color = self.color_map[series_name]

            # Normalize to [0, 1] based on global range
            norm_values = (values - global_min) / value_range

            # Convert to image coordinates
            num_points = len(norm_values)
            points = [
                (
                    int(i * width / (num_points - 1)) if num_points > 1 else width // 2,
                    int(height - (v * height)),
                )
                for i, v in enumerate(norm_values)
            ]

            # Draw the line
            if len(points) > 1:
                draw.line(
                    points, fill=color, width=int(line_width * self.upscale_factor)
                )

        # Downscale for anti-aliasing
        final_img = img.resize(self.content_size, Image.LANCZOS)

        # Add margin if needed
        if image_margin > 0:
            final_img = ImageOps.expand(final_img, border=image_margin, fill="white")

        # Ensure final size matches requested
        final_img = final_img.resize(self.image_size, Image.LANCZOS)
        return final_img

    def save(self, image, path):
        """Save PIL Image to disk."""
        image.save(path)
