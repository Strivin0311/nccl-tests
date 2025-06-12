#!/bin/bash

SEP="======================================================================="

SRC_ROOT="./src"
BUILD_ROOT="./build"
OUT_ROOT="./out"

SRC_NAME="example6_mpstsd_group_cast"
SRC_FILEEXT="cpp"

SRC_PATH="${SRC_ROOT}/${SRC_NAME}"
BUILD_PATH="${BUILD_ROOT}/${SRC_NAME}"

WORLD_SIZE=4

echo $SEP
echo "Building ${SRC_PATH}.${SRC_FILEEXT} to $BUILD_PATH"
echo $SEP


nvcc -rdc=true -ccbin mpicxx -gencode arch=compute_90,code=sm_90 \
    -lnccl -lcudart -lcuda \
    -o $BUILD_PATH ${SRC_PATH}.${SRC_FILEEXT} \


echo $SEP
echo "Running $BUILD_PATH"
echo $SEP

CMD="mpirun --allow-run-as-root -np $WORLD_SIZE $BUILD_PATH"

$CMD
exit

nsys profile \
    -t cuda,nvtx \
    --force-overwrite true \
    -o $OUT_ROOT/$SRC_NAME.nsys-rep \
    --capture-range=cudaProfilerApi \
    $CMD

