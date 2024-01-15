import math
import re
import psutil
import torch
from transformers import is_torch_xpu_available
from user_interface import ui_settings as settings

# Finding the default values for the GPU and CPU memories
total_mem = []
if is_torch_xpu_available():
    for i in range(torch.xpu.device_count()):
        total_mem.append(math.floor(torch.xpu.get_device_properties(i).total_memory / (1024 * 1024)))
else:
    for i in range(torch.cuda.device_count()):
        total_mem.append(math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)))

default_gpu_mem = []
if settings.args.gpu_memory is not None and len(settings.args.gpu_memory) > 0:
    for i in settings.args.gpu_memory:
        if 'mib' in i.lower():
            default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
        else:
            default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)

while len(default_gpu_mem) < len(total_mem):
    default_gpu_mem.append(0)

total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
if settings.args.cpu_memory is not None:
    default_cpu_mem = re.sub('[a-zA-Z ]', '', settings.args.cpu_memory)
else:
    default_cpu_mem = 0

