#!/bin/bash

SEP="======================================================================="

SRC_ROOT="./src"
BUILD_ROOT="./build"

SRC_NAME="example2_spstmd_a2a"
SRC_FILEEXT="cpp"

SRC_PATH="${SRC_ROOT}/${SRC_NAME}"
BUILD_PATH="${BUILD_ROOT}/${SRC_NAME}"

echo $SEP
echo "Building ${SRC_PATH}.${SRC_FILEEXT} to $BUILD_PATH"
echo $SEP

nvcc -o $BUILD_PATH ${SRC_ROOT}/init_send_buffer.cu ${SRC_PATH}.${SRC_FILEEXT} \
    -lnccl -lcudart -lcuda \
    -I${CUDA_PATH}/include -I${NCCL_PATH}/include -I${MPI_HOME}/include -I${MPI_HOME}/include/openmpi/ \
    -L${CUDA_PATH}/lib64 -L${NCCL_PATH}/lib64 -L${MPI_HOME}/lib -L${MPI_HOME}/lib/openmpi \
    -gencode arch=compute_90,code=sm_90 \
&& \
echo $SEP
echo "Running $BUILD_PATH"
echo $SEP

./${BUILD_PATH}

