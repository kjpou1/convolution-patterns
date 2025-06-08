from convolution_patterns.config.config import Config
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.services.image_dataset_service import ImageDatasetService
from convolution_patterns.services.transform_service import TransformService

logging = LoggerManager.get_logger(__name__)

class TrainPipeline:
    def __init__(self, split: str = "train", batch_size: int = 32):
        self.config = Config()
        self.split = split
        self.batch_size = batch_size
        self.print_stats = True


    def run(self):
        # Load transform config
        # transform_service = TransformService.from_yaml(self.config_path)
        # transform_pipeline = transform_service.get_pipeline(mode=self.split)

        # Load dataset
        dataset_service = ImageDatasetService()
        train_ds, train_class_names = dataset_service.get_dataset("train",self.print_stats)
        val_ds, val_class_names = dataset_service.get_dataset("val",self.print_stats)

        return train_ds, val_ds
