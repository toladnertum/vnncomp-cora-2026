#!/bin/bash

# Example prepare_instance.sh script for VNNCOMP'25 for CORA.
# Arguments:
# - version string "v1", 
# - benchmark identifier string, e.g., "acasxu", 
# - path to .onnx file,
# - path to .vnnlib file

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

echo "Preparing $TOOL_NAME for benchmark instance '$BENCHMARK' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE'"

# Check GPU status.
nvidia-smi

# add the toolkit (and CORA under code/) to the MATLAB path; savepath is not
# kept for these sudo matlab runs. do not cd: the onnx/vnnlib paths are relative.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# double single quotes so multi-network paths like [('f','..'),('g','..')]
# don't break the MATLAB string literals
BENCHMARK_M=${BENCHMARK//\'/\'\'}
ONNX_M=${ONNX_FILE//\'/\'\'}
VNNLIB_M=${VNNLIB_FILE//\'/\'\'}

# Build MATLAB command, optionally with overrides file.
if [ -n "$CORA_OVERRIDES_FILE" ] && [ -f "$CORA_OVERRIDES_FILE" ]; then
    echo "Using overrides file: $CORA_OVERRIDES_FILE"
    OVERRIDES_M=${CORA_OVERRIDES_FILE//\'/\'\'}
    sudo matlab -nodisplay -r "addpath(genpath('$SCRIPT_DIR')); prepare_instance('$BENCHMARK_M','$ONNX_M','$VNNLIB_M','$OVERRIDES_M'); quit;"
else
    sudo matlab -nodisplay -r "addpath(genpath('$SCRIPT_DIR')); prepare_instance('$BENCHMARK_M','$ONNX_M','$VNNLIB_M'); quit;"
fi

exit 0
