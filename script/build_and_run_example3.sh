#!/bin/bash

SEP="======================================================================="

SRC_ROOT="./src"
BUILD_ROOT="./build"

SRC_NAME="example3_mpstsd_ar"
SRC_FILEEXT="cpp"

SRC_PATH="${SRC_ROOT}/${SRC_NAME}"
BUILD_PATH="${BUILD_ROOT}/${SRC_NAME}"

WORLD_SIZE=8

echo $SEP
echo "Building ${SRC_PATH}.${SRC_FILEEXT} to $BUILD_PATH"
echo $SEP

nvcc -c ${SRC_ROOT}/init_send_buffer.cu -o ${BUILD_ROOT}/init_send_buffer.o \
    -lcudart -lcuda \
    -I${CUDA_PATH}/include \
    -L${CUDA_PATH}/lib64 \
    -gencode arch=compute_90,code=sm_90 \
&& \
mpicxx -o $BUILD_PATH  ${SRC_PATH}.${SRC_FILEEXT} ${BUILD_ROOT}/init_send_buffer.o \
    -lnccl -lcudart -lcuda \
    -I${CUDA_PATH}/include -I${NCCL_PATH}/include \
    -L${CUDA_PATH}/lib64 -L${NCCL_PATH}/lib64 \
&& \
echo $SEP
echo "Running $BUILD_PATH"
echo $SEP

mpirun --allow-run-as-root -np $WORLD_SIZE $BUILD_PATH