import tensorflow as tf

from convolution_patterns.config.config import Config
from convolution_patterns.config.model_config import ModelConfig
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.utils.layer_factory import build_layer

logging = LoggerManager.get_logger(__name__)


class ModelBuilderService:
    def __init__(self):
        self.config = Config()

        path = self.config.model_config_path
        if path is None:
            raise ValueError("Model config path must not be None.")

        self.model_config = ModelConfig.from_yaml(path)
        self.model_cfg = self.model_config.model
        self.training_cfg = self.model_config.training

        self.backbone_name = self.model_cfg["backbone"]
        self.input_shape = tuple(self.model_cfg["input_shape"])
        self.head_layers_cfg = self.model_cfg["custom_head"]["layers"]
        self.freeze_backbone = self.model_cfg.get(
            "freeze_backbone", True
        )  # default True

    def build(self, num_classes: int) -> tf.keras.Model:
        logging.info(f"Building model with backbone: {self.backbone_name}")

        backbone = self._load_backbone()
        backbone.trainable = not self.freeze_backbone

        x = backbone.output

        for layer_cfg in self.head_layers_cfg:
            x = build_layer(x, layer_cfg, num_classes=num_classes)

        model = tf.keras.Model(inputs=backbone.input, outputs=x)

        model.compile(
            optimizer=self._get_optimizer(),
            loss=self._get_loss_fn(),
            metrics=self._get_metrics(),
        )

        model.summary()

        logging.info("Model built and compiled.")
        return model

    def _load_backbone(self):
        kwargs = dict(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
            pooling=None,
        )

        name = self.backbone_name
        if name == "EfficientNetB0":
            return tf.keras.applications.EfficientNetB0(**kwargs)
        elif name == "EfficientNetB3":
            return tf.keras.applications.EfficientNetB3(**kwargs)
        elif name == "MobileNetV2":
            return tf.keras.applications.MobileNetV2(**kwargs)
        elif name == "MobileNetV3Small":
            return tf.keras.applications.MobileNetV3Small(**kwargs)
        elif name == "MobileNetV3Large":
            return tf.keras.applications.MobileNetV3Large(**kwargs)
        elif name == "ResNet50":
            return tf.keras.applications.ResNet50(**kwargs)
        elif name == "ResNet50V2":
            return tf.keras.applications.ResNet50V2(**kwargs)
        elif name == "DenseNet121":
            return tf.keras.applications.DenseNet121(**kwargs)
        elif name == "InceptionV3":
            return tf.keras.applications.InceptionV3(**kwargs)
        elif name == "NASNetMobile":
            return tf.keras.applications.NASNetMobile(**kwargs)
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def _get_optimizer(self):
        opt_name = self.training_cfg["optimizer"]
        lr = self.training_cfg["learning_rate"]

        if opt_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif opt_name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _get_metrics(self):

        return [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]

    def _get_loss_fn(self):
        loss_cfg = self.training_cfg.get("loss", "categorical_crossentropy")

        if isinstance(loss_cfg, str):
            return tf.keras.losses.get(loss_cfg)

        if isinstance(loss_cfg, dict):
            loss_type = loss_cfg.get("type", "categorical_crossentropy")

            if loss_type == "categorical_crossentropy":
                return tf.keras.losses.CategoricalCrossentropy(
                    from_logits=False,
                    label_smoothing=loss_cfg.get("label_smoothing", 0.0),
                )

            if loss_type == "focal_loss":
                alpha = loss_cfg.get("alpha", 0.25)
                gamma = loss_cfg.get("gamma", 2.0)

                def categorical_focal_loss(alpha, gamma):
                    alpha = tf.constant(alpha, dtype=tf.float32)
                    gamma = tf.constant(gamma, dtype=tf.float32)

                    def loss(y_true, y_pred):
                        y_pred = tf.clip_by_value(
                            y_pred,
                            tf.keras.backend.epsilon(),
                            1.0 - tf.keras.backend.epsilon(),
                        )
                        cross_entropy = -y_true * tf.math.log(y_pred)
                        weight = alpha * tf.math.pow(1 - y_pred, gamma)
                        return tf.reduce_sum(weight * cross_entropy, axis=-1)

                    return loss

                return categorical_focal_loss(alpha, gamma)

            # fallback to other built-in losses if needed
            return tf.keras.losses.get(loss_type)

        raise ValueError(f"Unsupported loss config: {loss_cfg}")

    @property
    def model_name(self) -> str:
        return self.model_cfg.get("name") or f"{self.backbone_name}_custom_head"
