import numpy as np
import tensorflow as tf

from helper_scripts.power_management import get_gpu_power_usage, get_cpu_power_usage
from inference.models.inference import Inference
from naso.settings import ENERGY_MEASUREMENT_FREQUENCY
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining


class EnergyCallback(tf.keras.callbacks.Callback):
    """
    Callback class to measure and log power consumption during training, testing, and prediction.

    This callback calculates the power usage of the GPU during different stages of the model's lifecycle,
    such as training, testing, and prediction. It uses the `nvidia-smi` command-line tool to query the GPU's
    power draw and calculates the average power usage.

    The power consumption measurements are logged in the `logs` dictionary, which can be accessed by other
    callbacks or during model evaluation.

    Example usage:

    ```python
    energy_callback = EnergyCallback(run, ...)
    model.fit(x_train, y_train, callbacks=[energy_callback])
    ```

    Args:
        run (AutoKerasRun): The AutoKeras run object.
        *args: Additional positional arguments to be passed to the parent class.
        **kwargs: Additional keyword arguments to be passed to the parent class.
    """

    measurements = []
    trial_measurements = []
    _last_measurement = 0

    def __init__(
        self, run: AutoKerasRun | Inference | NetworkTraining, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.run = run
        self.trial_measurements = []

    def measure_power(self):
        if self.run.gpu.startswith("GPU"):
            measurement = get_gpu_power_usage(self.run.gpu)
            self.measurements.append(measurement)
        elif self.run.gpu.startswith("CPU"):
            measurement = get_cpu_power_usage(self.run.gpu)
            self.measurements.append(measurement)

    def on_epoch_begin(self, epoch, logs=None, *args):
        """
        Callback method called at the beginning of each epoch.
        Measures the power usage of the GPU and stores it in the `measurements` list.

         Args:
             epoch (int): The current epoch number.
             logs (dict): Dictionary containing the training metrics for the current epoch.
             *args: Additional arguments.

         Returns:
             None
        """
        if logs is None:
            logs = []
        self.measurements = []
        self.measure_power()

    def on_epoch_end(self, epoch, logs=None, *args):
        """
        Callback function called at the end of each epoch during training.
        Logs the average power usage of the GPU during the epoch in the `logs` dictionary.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training metrics for the current epoch.
            *args: Additional arguments.

        Returns:
            None
        """
        if logs is None:
            logs = []
        self.measure_power()

        # calculate average power usage:
        average_power_usage = sum(self.measurements) / len(self.measurements)
        logs["power_consumption"] = average_power_usage
        self.trial_measurements.append(average_power_usage)
        logs["trial_power_consumption"] = sum(self.trial_measurements) / len(
            self.trial_measurements
        )
        logs["trial_power_consumption_var"] = np.var(self.trial_measurements)
        if "execution_time" in logs:
            logs["energy_consumption [Ws]"] = (
                average_power_usage * logs["execution_time"]
            )

    def on_test_begin(self, logs=None):
        """
        Called at the beginning of the testing phase.
        Measures the power usage of the GPU and stores it in the `measurements` list.

        Args:
            logs (dict): Dictionary of logs containing the current state of the model and the evaluation metrics.

        Returns:
            None
        """
        self.measurements = []
        self.measure_power()

    def on_test_end(self, logs=None):
        """
        Callback function called at the end of testing.

        Args:
            logs (dict): Dictionary containing the evaluation metrics and
                         losses for the current test step.

        Returns:
            None

        This function calculates the power usage measurement at the end of testing,
        appends it to the measurements list, and calculates the average power usage
        based on all the measurements. The average power usage is then added to the
        logs dictionary with the key "power_consumption".
        """
        self.measure_power()
        average_power_usage = sum(self.measurements) / len(self.measurements)
        logs["power_consumption"] = average_power_usage

    def on_predict_begin(self, logs=None):
        """
        Callback method called at the beginning of the prediction process.

        Args:
            logs (dict): Dictionary of logs containing information about the prediction process.

        Returns:
            None
        """
        self.measurements = []

    def on_predict_end(self, logs=None):
        """
        Callback method called at the end of each prediction.

        Args:
            logs (dict): Dictionary containing the logs and metrics of the model.

        Returns:
            None

        This method calculates the power usage measurement at the end of each prediction and updates
        the average power usage. This is then added to the logs dictionary with the key "power_consumption".
        """
        self.measure_power()
        average_power_usage = sum(self.measurements) / len(self.measurements)
        logs["power_consumption"] = average_power_usage
        logs["power_consumption_var"] = np.var(self.measurements)
        if "execution_time_mean" in logs:
            logs["energy_consumption [Ws]"] = (
                average_power_usage * logs["execution_time_mean"]
            )

    def on_predict_batch_end(self, batch, logs=None):
        """
        Callback method called at the end of each batch during prediction.

        Args:
            batch: The batch index.
            logs: Dictionary containing the loss value and other metrics.

        Returns:
            None
        """
        if logs["total_time"] - self._last_measurement > ENERGY_MEASUREMENT_FREQUENCY:
            self._last_measurement = logs["total_time"]
            self.measure_power()
