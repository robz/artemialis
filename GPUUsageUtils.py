####################################################################
# GPU usage
####################################################################


import os
import GPUtil

prev = [0]

def printm():
  GPUs = GPUtil.getGPUs()
  gpu = GPUs[0]
  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Delta: {2:.0f}MB | Util {3:3.0f}% | Total     {4:.0f}MB".format(
    gpu.memoryFree,
    gpu.memoryUsed,
    gpu.memoryUsed - prev[0],
    gpu.memoryUtil*100,
    gpu.memoryTotal
  ))
  prev[0] = gpu.memoryUsed
  return gpu.memoryUtil*100
