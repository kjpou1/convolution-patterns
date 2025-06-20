"""
Base interface for chart rendering backends and shared constants.
"""

# === 🎨 Color Constants ===
COLOR_CLOSE = "#2196F3"  # Blue
COLOR_AHMA = "#880E4F"  # Deep Pink
COLOR_PROJECTION = "#FF9800"  # Orange
COLOR_CONVOLUTION = "#4CAF50"  # Green

# === 🧮 Configuration Constants ===
DEFAULT_IMAGE_SIZE = (128, 128)
# DEFAULT_NUM_DAYS = 21  # Max bars per window (starting window)
# ENDING_NUM_DAYS = 5  # Min bars allowed (final tier)
# WINDOW_STRIDE = 2  # Decrease step per tier


class ChartRenderService:
    """
    Abstract base class for chart rendering backends.
    Provides shared constants for all subclasses.
    """

    # Optionally, expose constants as class attributes
    COLOR_CLOSE = COLOR_CLOSE
    COLOR_AHMA = COLOR_AHMA
    COLOR_PROJECTION = COLOR_PROJECTION
    COLOR_CONVOLUTION = COLOR_CONVOLUTION

    DEFAULT_IMAGE_SIZE = DEFAULT_IMAGE_SIZE
    # DEFAULT_NUM_DAYS = DEFAULT_NUM_DAYS
    # ENDING_NUM_DAYS = ENDING_NUM_DAYS
    # WINDOW_STRIDE = WINDOW_STRIDE

    def render(self, window_data, **kwargs):
        """
        Render a chart image from the provided window_data.

        Args:
            window_data: DataFrame or array-like structure containing the data for the chart.
            **kwargs: Additional keyword arguments for backend-specific options.

        Returns:
            An image object (e.g., PIL.Image or np.ndarray) representing the rendered chart.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement render()")

    def save(self, image, path):
        """
        Save the rendered image to disk.
        Subclasses should implement this for their output type.
        """
        raise NotImplementedError("Subclasses must implement save()")

    @staticmethod
    def get_renderer(backend: str, **kwargs) -> "ChartRenderService":
        """
        Factory method to get the appropriate chart renderer based on backend.

        Args:
            backend (str): The rendering backend to use ('matplotlib', 'pil', etc.)
            **kwargs: Additional arguments to pass to the renderer constructor

        Returns:
            ChartRenderService: An instance of the appropriate renderer

        Raises:
            ValueError: If the backend is not supported
        """
        # NOTE: Backend imports are inside get_renderer to avoid circular imports.
        if backend == "matplotlib":
            from .matplotlib_backend import MatplotlibRenderBackend

            return MatplotlibRenderBackend(**kwargs)
        elif backend == "pil":
            from .pil_backend import PILRenderBackend

            return PILRenderBackend(**kwargs)
        else:
            raise ValueError(f"Unsupported rendering backend: {backend}")
