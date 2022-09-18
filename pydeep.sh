#!/bin/bash

# Load required modules
module purge
module load Stages/2022
module load GCC/11.2.0
module load OpenMPI
module load JupyterKernel-PyDeepLearning/.1.1-2022.3.3

module load OpenCV/4.5.4
#module load TensorFlow/2.6.0-CUDA-11.5
module load protobuf-python/.3.17.3
module load PyTorch/1.11-CUDA-11.5
module load PyTorch-Geometric/2.0.4
module load torchvision/0.12.0-CUDA-11.5
#module load OpenAI-Gym/0.18.0
#module load DeepSpeed/0.5.4
#module load Horovod/0.24.3