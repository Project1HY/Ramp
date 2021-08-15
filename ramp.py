import psutil
import platform
import socket
import os
import shutil
import torch
import cpuinfo
from tensorflow.python.client import device_lib

def identify_system():
    print("Computer Name :" , socket.gethostname())
    print("OS : " , platform.system(), platform.release())
    print("Python : ", platform.python_version())
    print("CPU : ", cpuinfo.get_cpu_info()['brand_raw'], "[", psutil.cpu_count(logical=False), "PCores ,",
    psutil.cpu_count(logical=True), "LCores ,", "{:.1f}".format(psutil.virtual_memory()[0]//(2**30)), "GB VRAM", "]")
    print("Free Disk Space : ", "{:.1f}".format(shutil.disk_usage("C:").free // (2**30)), "GB")
    print("Compute Packages : ", "PyTorch", torch.__version__[0:5], )
    print("# Visible GPUs :" )
    #phy_cores = psutil.cpu_count(logical=False)
    #log_cores = psutil.cpu_count(logical=True)
    #vram = psutil.virtual_memory()[0]
    #free_space = "{:.1f}".format(shutil.disk_usage("C:").free // (2**30))

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

identify_system()

print(open(os.path("../cuda/version.txt")))