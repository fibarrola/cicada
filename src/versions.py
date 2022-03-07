#@title Pre Installation {vertical-output: true}

import subprocess
import sys
import torch

def get_versions():

    CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
    print("CUDA version:", CUDA_version)

    if CUDA_version == "10.0":
        torch_version_suffix = "+cu100"
    elif CUDA_version == "10.1":
        torch_version_suffix = "+cu101"
    elif CUDA_version == "10.2":
        torch_version_suffix = ""
    else:
        torch_version_suffix = "+cu110"

    return torch_version_suffix, CUDA_version


def getinfo():
    print('A', sys.version)
    print('B', torch.__version__)
    print('C', torch.cuda.is_available())
    print('D', torch.backends.cudnn.enabled)
    device = torch.device('cuda')
    print('E', torch.cuda.get_device_properties(device))
    print('F', torch.tensor([1.0, 2.0]).cuda())