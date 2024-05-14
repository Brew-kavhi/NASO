import tensorflow as tf
from loguru import logger


class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, write_to_file: bool = False):
        super().__init__()
        self.logs = []
        self.store_in_file = write_to_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logs.append(logs.copy())
        print(f"Epoch {epoch+1}: {logs}")
        if self.store_in_file:
            logger.info(f"Epoch {epoch + 1}: {logs}")

    def on_test_end(self, logs=None):
        logs = logs or {}
        self.logs.append(logs.copy())
        print(f"predict: {logs}")
        if self.store_in_file:
            logger.info(f"Predict: {logs}")

    def on_predict_end(self, logs=None):
        logs = logs or {}
        self.logs.append(logs.copy())
        print(f"predict: {logs}")
        if self.store_in_file:
            logger.info(f"Predict: {logs}")
