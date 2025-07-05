import tensorflow as tf
from tensorflow.keras.models import Model

from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


def apply_partial_unfreeze(
    backbone_model: Model, unfreeze_from: int, freeze_batchnorm: bool = False
):
    """
    Freezes all layers before unfreeze_from index.
    Optionally freezes BatchNormalization layers.

    Args:
        backbone_model: The backbone Keras model.
        unfreeze_from: Layer index to start unfreezing (negative indices supported).
        freeze_batchnorm: Whether to keep BatchNorm layers frozen.
    """
    backbone_model.trainable = True

    total_layers = len(backbone_model.layers)
    if unfreeze_from >= 0:
        start_idx = unfreeze_from
    else:
        start_idx = total_layers + unfreeze_from

    if start_idx < 0 or start_idx > total_layers:
        raise ValueError(
            f"'unfreeze_from_layer' index {unfreeze_from} is out of range (total layers: {total_layers})"
        )

    for i, layer in enumerate(backbone_model.layers):
        if freeze_batchnorm and isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = i >= start_idx

    unfrozen_layers = [
        layer.name for i, layer in enumerate(backbone_model.layers) if i >= start_idx
    ]
    logging.info(
        f"Unfrozen {len(unfrozen_layers)} layers from index {start_idx} "
        f"(BatchNorm frozen: {freeze_batchnorm}): {unfrozen_layers}"
    )
