import subprocess
import numpy as np
from decouple import config


def get_cpu_power_usage(cpu):
    """
    Retrieves the power usage of the GPU.
    Returns:
        float: The power usage in watts.
    """
    result = subprocess.run(
        config("CPU_POWERTOOL_CMD"),
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    power_usages = [float(power) for power in lines[1:]]
    return np.mean(power_usages)


def get_gpu_power_usage(gpu):
    """
    Retrieves the power usage of the GPU.
    Returns:
        float: The power usage in watts.
    """
    gpu_index = int(gpu.split(":")[-1])
    result = subprocess.run(
        [
            config("GPU_POWERTOOL_CMD"),
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
