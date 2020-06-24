
#! /bin/bash

export SOR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$SOR

cd $SOR
export PYTHONPATH=$SOR/modules:$PYTHONPATH
export PYTHONPATH=$SOR/modules/datastructures:$PYTHONPATH
export PYTHONPATH=$SOR/modules/DynamicReduction:$PYTHONPATH
export PATH=$SOR/scripts:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras:/usr/local/cuda/compat
export CUDA_CACHE_PATH=/tmp/$USER/cuda
export CUDA_VISIBLE_DEVICES=0
