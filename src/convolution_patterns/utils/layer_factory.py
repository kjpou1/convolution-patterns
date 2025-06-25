import tensorflow as tf


def build_layer(x, layer_cfg: dict, num_classes: int):
    layer_type = layer_cfg.get("type")

    # Dynamic injection for output layer
    if layer_type == "Dense" and layer_cfg.get("units") == "num_classes":
        layer_cfg = layer_cfg.copy()
        layer_cfg["units"] = num_classes

    kwargs = {k: v for k, v in layer_cfg.items() if k != "type"}

    if layer_type == "Dense":
        return tf.keras.layers.Dense(**kwargs)(x)
    elif layer_type == "Dropout":
        return tf.keras.layers.Dropout(**kwargs)(x)
    elif layer_type == "BatchNormalization":
        return tf.keras.layers.BatchNormalization(**kwargs)(x)
    elif layer_type == "LayerNormalization":
        return tf.keras.layers.LayerNormalization(**kwargs)(x)
    elif layer_type == "Activation":
        return tf.keras.layers.Activation(**kwargs)(x)
    elif layer_type == "GlobalAveragePooling2D":
        return tf.keras.layers.GlobalAveragePooling2D(**kwargs)(x)
    elif layer_type == "Flatten":
        return tf.keras.layers.Flatten(**kwargs)(x)
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")
