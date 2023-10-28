import subprocess

import tensorflow as tf

from neural_architecture.models.AutoKeras import AutoKerasRun


class EnergyCallback(tf.keras.callbacks.Callback):
    measurements = []
    trial_measurements = []

    def __init__(self, run: AutoKerasRun, *args, **kwargs):
        super().__init__()
        self.run = run

    def get_power_usage(self, gpu_index):
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        power_usage = float(result.stdout.strip())  # Power usage in watts
        return power_usage

    def get_total_time(self):
        return round(self.timer.get_total_time(), 2)

    def on_epoch_begin(self, epoch, logs=[], *args):
        # start the timer here
        self.measurements = []
        measurement = self.get_power_usage(0)
        self.measurements.append(measurement)

    def on_epoch_end(self, epoch, logs=[], *args):
        # stop the timer here:
        meausrement = self.get_power_usage(0)
        self.measurements.append(meausrement)

        # calculate average power usage:
        average_power_usage = sum(self.measurements) / len(self.measurements)
        logs["energy_consumption"] = average_power_usage
        self.trial_measurements.append(average_power_usage)
        logs["trial_energy_consumption"] = sum(self.trial_measurements) / len(
            self.trial_measurements
        )
