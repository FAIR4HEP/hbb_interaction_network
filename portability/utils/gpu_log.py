import nvidia_smi

def get_gpu_log():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

#     res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
#     print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f'mem: {mem_res.used / (1024**2)} (GiB)') # usage in GiB
    print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%') # percentage usage
    res = 100 * (mem_res.used / mem_res.total)
    return res