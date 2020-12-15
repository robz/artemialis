####################################################################
# GPU usage
####################################################################


import psutil
import humanize
import os
import GPUtil as GPU

prev = [0]

def printm():
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    process = psutil.Process(os.getpid())
    # print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Delta: {2:.0f}MB | Util {3:3.0f}% | Total     {4:.0f}MB".format(
      gpu.memoryFree,
      gpu.memoryUsed,
      gpu.memoryUsed - prev[0],
      gpu.memoryUtil*100,
      gpu.memoryTotal
    ))
    prev[0] = gpu.memoryUsed
    return gpu.memoryUtil*100
