import numpy as np
import tensorflow as tf

from helper_scripts.timer import Timer
from naso.settings import ENERGY_MEASUREMENT_FREQUENCY


class TimingCallback(tf.keras.callbacks.Callback):
    """
    Callback for evaluating the execution time of testing and prediction in a neural network model.
    Logs the execution time of each batch and the average execution time of the testing/prediction.
    Furthrmore, it writes the prediction metrics to the database.

    Attributes:
        times (list): List to store the execution times of each batch.
        run (Run): The run object associated with the callback.
        timer (Timer): Timer object for measuring execution time.

    Methods:
        on_test_begin(logs=None): Called at the beginning of testing.
        on_test_end(logs=None): Called at the end of testing.
        on_test_batch_begin(batch, logs=None): Called at the beginning of each testing batch.
        on_test_batch_end(batch, logs=None): Called at the end of each testing batch.
        on_predict_begin(logs=None): Called at the beginning of prediction.
        on_predict_end(logs=None): Called at the end of prediction.
        on_predict_batch_begin(batch, logs=None): Called at the beginning of each prediction batch.
        on_predict_batch_end(batch, logs=None): Called at the end of each prediction batch.
    """

    times = []
    _batch = 0
    _last_measurement = 0

    def __init__(self):
        super().__init__()
        self.timer = Timer()
        self.batch_timer = Timer()
        self.times = []
        self.batch_times = []
        self.batches = 1

    def on_epoch_begin(self, epochs, logs=None):
        """
        Called at the beginning of an epoch.

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.times = []
        self.batch_times = []
        self.timer.start()

    def on_epoch_end(self, epochs, logs=None):
        """
        Called at the end of an epoch.
        Logs the execution time of the epoch

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        elapsed_time = self.timer.stop()
        if logs is None:
            logs = []
        self.times.append(elapsed_time)
        if len(self.times) > 0:
            logs["epoch_execution_time"] = sum(self.times) / len(self.times)
        else:
            logs["epoch_execution_time"] = 0
        if len(self.batch_times) > 0:
            logs["execution_time"] = sum(self.batch_times) / len(self.batch_times)
        else:
            logs["execution_time"] = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_timer.start()

    def on_train_batch_end(self, batch, logs=None):
        batch_time = self.batch_timer.stop()
        self.batch_times.append(batch_time)

    def on_test_begin(self, logs=None):
        """
        Called at the beginning of testing.

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.times = []

    def on_test_end(self, logs=None):
        """
        Called at the end of testing.
        Logs the execution time of the testing

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        logs["execution_time"] = sum(self.times) / len(self.times)

    def on_test_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of each testing batch.

        Args:
            batch: The batch index.
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.timer.start()

    def on_test_batch_end(self, batch, logs=None):
        """
        Called at the end of each testing batch.
        Logs the execution time of the batch.

        Args:
            batch: The batch index.
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        elapsed_time = self.timer.stop()
        self.times.append(elapsed_time)
        logs["execution_time"] = self.timer.get_total_time()

    def on_predict_begin(self, logs=None):
        """
        Called at the beginning of prediction.

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.times = []

    def on_predict_end(self, logs=None):
        """
        Called at the end of prediction.
        Logs the execution time of the prediction and writes metrics to the database

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        logs["execution_time_mean"] = sum(self.times) / len(self.times)
        logs["execution_time_variance"] = np.var(self.times)

    def on_predict_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of each prediction batch.
        Starts the time for measuring the execution time.

        Args:
            batch: The batch index.
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.timer.start()

    def on_predict_batch_end(self, batch, logs=None):
        """
        Called at the end of each prediction batch.
        Logs the execution time of the batch.

        Args:
            batch: The batch index.
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        if not logs:
            logs = []
        elapsed_time = self.timer.stop()
        logs["total_time"] = self.timer.get_total_time()
        if logs["total_time"] - self._last_measurement > ENERGY_MEASUREMENT_FREQUENCY:
            self._last_measurement = logs["total_time"]
        else:
            self.times.append(elapsed_time)
            logs["execution_time"] = elapsed_time
