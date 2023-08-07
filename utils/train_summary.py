from pynvml import *


def display_gpu_utilization(gpu_index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU #{gpu_index} memory occupied: {info.used//1024**2} MB.")


def display_summary(result, total_node=2):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    for i in range(total_node):
        display_gpu_utilization(i)
