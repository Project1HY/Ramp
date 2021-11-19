import os
import platform

import psutil
import shutil
from util.fileio import convert_bytes
from util.strings import banner


# TODO - add tensorflow support identify_gpu_system
# TODO - find some better way than dereference the config
# ----------------------------------------------------------------------------------------------------------------------
#                                                       Platform
# ----------------------------------------------------------------------------------------------------------------------

def computer_name():
    from socket import gethostname
    return gethostname()


def python_version():
    return platform.python_version()


def operating_system_identity(short=False):
    import platform
    return platform.platform(aliased=True, terse=short)


def is_linux():
    """Check if the operating system is Linux.
    Returns
    -------
    bool
        True if the OS is Linux. False otherwise
    """
    return os.name == 'posix'


def is_windows():
    """Check if the operating system is Windows.
    Returns
    -------
    bool
        True if the OS is Windows. False otherwise
    """
    return os.name == 'nt'


# ----------------------------------------------------------------------------------------------------------------------
#                                                        CPU
# ----------------------------------------------------------------------------------------------------------------------
def free_disk_space(readable=True):
    total, used, free = shutil.disk_usage("/")
    return convert_bytes(free) if readable else free


def cpu_count(logical=True):
    return psutil.cpu_count(logical)


def total_cpu_virtual_memory(readable=True):
    mem = psutil.virtual_memory().total
    return convert_bytes(mem) if readable else mem


def cpu_identity():
    from util.hardware.cpuinfo import cpu
    cpu_dict = cpu.info[0]
    if 'ProcessorNameString' in cpu_dict:  # Windows
        cpu_name = cpu_dict['ProcessorNameString']
    elif 'model name' in cpu_dict:  # Linux
        cpu_name = cpu_dict['model name']
    else:
        raise NotImplementedError

    cpu_id = f'{cpu_name} [{psutil.cpu_count(logical=False)} PCores , {psutil.cpu_count(logical=True)} ' \
             f'LCores , {total_cpu_virtual_memory(readable=True)} VRAM]'

    return cpu_id


def identify_cpu_platform(use_config_naming_alias=False):
    comp_name = computer_name()
    if use_config_naming_alias:  # TODO - coupling antipattern
        from cfg import PC_NAME_MAPPING
        comp_name = PC_NAME_MAPPING.get(comp_name, comp_name)
    print(f'Computer Name    :    {comp_name}')
    print(f'OS               :    {operating_system_identity()}')
    print(f'Python           :    {python_version()}')
    print(f'CPU              :    {cpu_identity()}')
    print(f'Free Disk Space  :    {free_disk_space(readable=True)}')
    # from cfg import PC_NAME_MAPPING
    # name = PC_NAME_MAPPING.get(name, name)


# ----------------------------------------------------------------------------------------------------------------------
#                                                           CUDA
# ----------------------------------------------------------------------------------------------------------------------
def identify_pytorch_hardware():
    import torch
    banner('PyTorch Hardware Framework')
    identify_cpu_platform(use_config_naming_alias=True)
    print(f'Compute Packages :    PyTorch {torch.__version__}, CUDA {torch.version.cuda}, '
          f'CuDNN {torch.backends.cudnn.version()}')

    gpu_count = torch.cuda.device_count()
    print(f'#Visible GPUs    :    {gpu_count}')
    for i in range(gpu_count):
        p = torch.cuda.get_device_properties(i)
        print(f'\t\tGPU {i}: {p.name} [{p.multi_processor_count} SMPs , {convert_bytes(p.total_memory)} RAM]')


def identify_tensorflow_hardware():
    # TODO - incomplete
    import tensorflow as tf
    banner('Tensorflow Hardware Framework')
    identify_cpu_platform(use_config_naming_alias=True)
    print(f'Compute Packages :    PyTorch {tf.__version__}, CUDA {tf.version.cuda}, '
          f'CuDNN {tf.backends.cudnn.version()}')  # TODO
    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpu_count = len(gpus)
    for i in range(gpu_count):
        pass
        # p = torch.cuda.get_device_properties(i) # TODO
        # print(f'\t\tGPU {i}: {p.name} [{p.multi_processor_count} SMPs , {convert_bytes(p.total_memory)} RAM]')


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def _hardware_test():
    print(operating_system_identity())
    identify_pytorch_hardware()


if __name__ == '__main__':
    total, used, free = shutil.disk_usage("/")
    print(convert_bytes(free))
    _hardware_test()
