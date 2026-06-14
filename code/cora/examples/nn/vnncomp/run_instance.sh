#!/bin/bash

# example run_instance.sh script for VNNCOMP'25 for CORA
# Arguments:
# - version string "v1", 
# - benchmark identifier string, e.g., "acasxu", 
# - path to .onnx file,
# - path to .vnnlib file,
# - path to results.csv file,
# - timeout in seconds

TOOL_NAME="CORA"
VERSION_STRING="v1"

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1

fi

BENCHMARK=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running $TOOL_NAME on benchmark instance $BENCHMARK with onnx file $ONNX_FILE, vnnlib file $VNNLIB_FILE, results file $RESULTS_FILE, and timeout $TIMEOUT"

# Check GPU status.
nvidia-smi

# add the toolkit (and CORA under code/) to the MATLAB path; savepath is not
# kept for these sudo matlab runs. do not cd: the onnx/vnnlib/results paths are relative.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# double single quotes so multi-network paths like [('f','..'),('g','..')]
# don't break the MATLAB string literals
BENCHMARK_M=${BENCHMARK//\'/\'\'}
ONNX_M=${ONNX_FILE//\'/\'\'}
VNNLIB_M=${VNNLIB_FILE//\'/\'\'}
RESULTS_M=${RESULTS_FILE//\'/\'\'}

sudo matlab -nodisplay -r "addpath(genpath('$SCRIPT_DIR')); run_instance('$BENCHMARK_M','$ONNX_M','$VNNLIB_M','$RESULTS_M',$TIMEOUT,true); quit;"
