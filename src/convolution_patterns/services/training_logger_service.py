import json
import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from convolution_patterns.config.config import Config
from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class TrainingLoggerService:
    def __init__(self, model_name: str):
        self.config = Config()
        self.model_name = model_name

        # Static history path (shared across runs)
        self.history_dir = self.config.HISTORY_DIR
        self.history_file = os.path.join(self.history_dir, "training_history.csv")
        os.makedirs(self.history_dir, exist_ok=True)

        # Versioned model dir: models/{model_name}/vN/model.keras
        self.model_base_dir = os.path.join(self.config.MODEL_DIR, model_name)
        self.model_dir = self._get_next_version_dir(self.model_base_dir)
        self.model_file = os.path.join(self.model_dir, "model.keras")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create/overwrite latest symlink
        self._update_latest_symlink(self.model_base_dir, self.model_dir)

        # Versioned report dir: reports/training/{model_name}/vN/
        self.report_base_dir = os.path.join(
            self.config.REPORTS_DIR, "training", model_name
        )
        self.report_dir = self._get_next_version_dir(self.report_base_dir)
        os.makedirs(self.report_dir, exist_ok=True)

        # Evaluation artifact paths
        self.loss_plot_path = os.path.join(self.report_dir, "loss_plot.png")
        self.accuracy_plot_path = os.path.join(self.report_dir, "accuracy_plot.png")
        self.metrics_json_path = os.path.join(self.report_dir, "metrics.json")
        self.confusion_matrix_path = os.path.join(
            self.report_dir, "confusion_matrix.png"
        )
        self.classification_report_path = os.path.join(
            self.report_dir, "classification_report.json"
        )

        logging.info(f"Training artifacts will be saved under versioned directories:")
        logging.info(f"Model dir: {self.model_dir}")
        logging.info(f"Report dir: {self.report_dir}")

    def _get_next_version_dir(self, base_dir: str) -> str:
        if not os.path.exists(base_dir):
            return os.path.join(base_dir, "v1")
        existing = [
            d for d in os.listdir(base_dir) if d.startswith("v") and d[1:].isdigit()
        ]
        versions = [int(d[1:]) for d in existing]
        next_version = max(versions, default=0) + 1
        return os.path.join(base_dir, f"v{next_version}")

    def _update_latest_symlink(self, base_dir: str, target_dir: str):
        symlink_path = os.path.join(base_dir, "latest")
        try:
            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                os.remove(symlink_path)
            os.symlink(pathlib.Path(target_dir).resolve(), symlink_path)
            logging.info(f"Created symlink: {symlink_path} -> {target_dir}")
        except Exception as e:
            logging.warning(f"Failed to create latest symlink: {e}")

    def save_history(self, history: tf.keras.callbacks.History):
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(self.history_file, index=False)
        logging.info(f"Saved training history to {self.history_file}")

    def save_plots(self, history: tf.keras.callbacks.History):
        history_dict = history.history

        def _plot(metric, title, path):
            plt.figure()
            plt.plot(history_dict[metric], label=f"train_{metric}")
            if f"val_{metric}" in history_dict:
                plt.plot(history_dict[f"val_{metric}"], label=f"val_{metric}")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(path)
            plt.close()
            logging.info(f"Saved {metric} plot to {path}")

        if "loss" in history_dict:
            _plot("loss", "Training & Validation Loss", self.loss_plot_path)
        if "accuracy" in history_dict or "acc" in history_dict:
            acc_key = "accuracy" if "accuracy" in history_dict else "acc"
            _plot(acc_key, "Training & Validation Accuracy", self.accuracy_plot_path)
        additional_metrics = ["precision", "recall", "auc"]
        for metric in additional_metrics:
            if metric in history_dict:
                plot_path = os.path.join(self.report_dir, f"{metric}_plot.png")
                _plot(metric, f"Training & Validation {metric.capitalize()}", plot_path)

    def save_model(self, model: tf.keras.Model):
        model.save(self.model_file)
        logging.info(f"Saved model to {self.model_file}")

    def save_metrics_summary(self, history: tf.keras.callbacks.History):
        metrics = {
            "final_train_loss": history.history["loss"][-1],
            "final_val_loss": history.history.get("val_loss", [None])[-1],
            "final_train_accuracy": history.history.get("accuracy", [None])[-1],
            "final_val_accuracy": history.history.get("val_accuracy", [None])[-1],
        }
        with open(self.metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved metrics summary to {self.metrics_json_path}")

    def save_evaluation_artifacts(self, y_true, y_pred, label_names):
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.confusion_matrix_path)
        plt.close()
        logging.info(f"Saved confusion matrix to {self.confusion_matrix_path}")

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
        )
        with open(self.classification_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logging.info(
            f"Saved classification report to {self.classification_report_path}"
        )
