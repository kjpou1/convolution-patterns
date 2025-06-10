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

    def build(self, num_classes: int) -> tf.keras.Model:
        logging.info(f"Building model with backbone: {self.backbone_name}")

        backbone = self._load_backbone()
        x = backbone.output

        for layer_cfg in self.head_layers_cfg:
            x = build_layer(x, layer_cfg, num_classes=num_classes)

        model = tf.keras.Model(inputs=backbone.input, outputs=x)

        model.compile(
            optimizer=self._get_optimizer(),
            loss=self.training_cfg["loss"],
            metrics=self._get_metrics()
        )

        logging.info("Model built and compiled.")
        return model

    def _load_backbone(self):
        if self.backbone_name == "EfficientNetB0":
            return tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=self.input_shape,
                pooling="avg"
            )
        elif self.backbone_name == "MobileNetV2":
            return tf.keras.applications.MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=self.input_shape,
                pooling="avg"
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

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
        return [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
