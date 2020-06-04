
#! /bin/bash

export SOR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$SOR

cd $SOR
export PYTHONPATH=$SOR/modules:$PYTHONPATH
export PYTHONPATH=$SOR/modules/datastructures:$PYTHONPATH
export PYTHONPATH=$SOR/modules/DynamicReduction:$PYTHONPATH
export PATH=$SOR/scripts:$PATH

