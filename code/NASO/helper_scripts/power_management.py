import subprocess
from jtop import jtop

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


def get_gpu_power_usage(gpu, jetson=None):
    """
    Retrieves the power usage of the GPU.
    Returns:
        float: The power usage in watts.
    gpu_index = int(gpu.split(":")[-1])
    result = subprocess.run(
        config("GPU_POWERTOOL_CMD"),
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    power_usage = float(result.stdout.strip())  # Power usage in watts
    return power_usage
    """
    
    if jetson:
        if jetson.ok():
            #print(jetson.power['tot']['power']/1000)
            return jetson.power['rail']['VDD_CPU_GPU_CV']['power']/1000

    with jtop() as jetson:
        # Check if board is compatible
        if jetson.ok():
            #print(jetson.power['tot']['power']/1000)
            return jetson.power['rail']['VDD_CPU_GPU_CV']['power']/1000
