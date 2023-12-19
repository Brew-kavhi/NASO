import subprocess


def get_power_usage(gpu):
    """
    Retrieves the power usage of the GPU.
    Returns:
        float: The power usage in watts.
    """
    gpu_index = int(gpu.split(":")[-1])
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
