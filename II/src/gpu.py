import os
import subprocess
from typing import List

import numpy as np

from src.command import CommandLineArgs


def sort_gpu(args: CommandLineArgs) -> int:
    output: List[str] = subprocess.run(
        'nvidia-smi -q -d Memory | grep -A5 GPU | grep Free', capture_output=True, shell=True,
        encoding='utf8'
    ).stdout.strip().split('\n')

    free_memory: np.ndarray = np.array([int(line.strip().split()[2]) for line in output])
    device_index: List[int] = np.argsort(free_memory).tolist()
    device_index.reverse()

    print(f'Available GPU: {device_index}')
    print(f'Memory (MB): {free_memory[device_index].tolist()}')
    os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in device_index])
    return len(device_index)


def init_gpu(args: CommandLineArgs) -> int:
    if args.max_gpu_num > 0:
        print('Initializing GPU ...')

        try:
            num_gpu: int = sort_gpu(args)
            if num_gpu == 0:
                print('WARNING: no GPU available, will use CPU instead')
            return num_gpu
        except Exception as e:
            print(e)
            return 0
    else:
        return 0
