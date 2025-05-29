#!/bin/bash

SEP="======================================================================="

export BUILD_SCRIPT_ROOT="./build"
export TEST_SCRIPT_ROOT="./script"
export TEST_OUTPUT_ROOT="./out"
export TEST_OUTPUT_PATH="${TEST_OUTPUT_ROOT}/nccl_test_output.txt"
export TEST_REPORT_PATH="${TEST_OUTPUT_ROOT}/nccl_test_report.csv"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="${TEST_OUTPUT_ROOT}/nccl_debug_info.txt"

export NCCL_TOPO_DUMP_FILE="${TEST_OUTPUT_ROOT}/topo.xml"
# export NCCL_TOPO_FILE="${TEST_OUTPUT_ROOT}/topo.xml"

export NCCL_GRAPH_DUMP_FILE="${TEST_OUTPUT_ROOT}/graph.xml"
# export NCCL_GRAPH_FILE="${TEST_OUTPUT_ROOT}/graph.xml"

# The NCCL_MAX_NCHANNELS variable limits the number of channels NCCL can use. 
# Reducing the number of channels also reduces the number of CUDA blocks used for communication, 
# hence the impact on GPU computing resources.
# export NCCL_MAX_NCHANNELS=16

# find out from cmd: `nvidia-smi nvlink -s`, and peak_bandwidth = bandwidth_per_link x num_links for intra-node
export INTRA_NODE_PEAK_BANDWIDTH=212.496
export PEAK_BANDWIDTH=$INTRA_NODE_PEAK_BANDWIDTH


NNODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


MULTI_PROCESS=true

if [[ $NNODES -gt 1 ]]; then
    MULTI_PROCESS=true
fi


START_BYTES=128M
END_BYTES=2G


# COMM_PRIMITIVE=all_reduce
# COMM_PRIMITIVE=all_gather
# COMM_PRIMITIVE=reduce_scatter
# COMM_PRIMITIVE=p2p
# NCCL does not define specific verbs for sendrecv, gather, gatherv, scatter, scatterv, alltoall, alltoallv, alltoallw, nor neighbor collectives. 
# All those operations can be simply expressed using a combination of ncclSend, ncclRecv, and ncclGroupStart/ncclGroupEnd, 
# similarly to how they can be expressed with MPI_Isend, MPI_Irecv and MPI_Waitall.
COMM_PRIMITIVE=all2all

COMM_ALG=all
# # On two nodes, NCCL uses the tree (or nvlstree) algorithm
# which can indeed go beyond the network bottleneck (provided the intra-node bandwidth can sustain that higher speed). 
# If you want to benchmark your network performance through NCCL on 2 nodes, you may want to force NCCL_ALGO=RING.
# COMM_ALG=ring
# COMM_ALG=tree
# COMM_ALG=tree,ring
# COMM_ALG=collnet
# COMM_ALG=collnetchain
# COMM_ALG=collnetdirect
# COMM_ALG=nvls
# COMM_ALG=nvlstree
# The PAT algorithm is a variation of the Bruck algorithm, 
# which features a logarithmic number of network steps for small sizes at scale, 
# progressively increasing the number of network transfers as sizes increase, 
# to keep buffering needs minimal.
# Initially, PAT only supports one GPU per node
# COMM_ALG=pat


if [[ $COMM_PRIMITIVE == "all_reduce" ]]; then
    echo $SEP
    echo "Testing AllReduce with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/all_reduce_perf
elif [[ $COMM_PRIMITIVE == "reduce_scatter" ]]; then
    echo $SEP
    echo "Testing ReduceScatter with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/reduce_scatter_perf
elif [[ $COMM_PRIMITIVE == "all_gather" ]]; then
    echo $SEP
    echo "Testing AllGather with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/all_gather_perf
elif [[ $COMM_PRIMITIVE == "broadcast" ]]; then
    echo $SEP
    echo "Testing Broadcast with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/broadcast_perf
elif [[ $COMM_PRIMITIVE == "reduce" ]]; then
    echo $SEP
    echo "Testing Reduce with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/reduce_perf
elif [[ $COMM_PRIMITIVE == "scatter" ]]; then
    echo $SEP
    echo "Testing Scatter with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/scatter_perf
elif [[ $COMM_PRIMITIVE == "gather" ]]; then
    echo $SEP
    echo "Testing Gather with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/gather_perf
elif [[ $COMM_PRIMITIVE == "all2all" ]]; then
    echo $SEP
    echo "Testing AllToAll with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/alltoall_perf
elif [[ $COMM_PRIMITIVE == "p2p" ]]; then
    echo $SEP
    echo "Testing P2P Send/Recv with algorithms: $COMM_ALG"
    echo $SEP

    TEST_SCRIPT=${BUILD_SCRIPT_ROOT}/sendrecv_perf
fi

if [[ $COMM_ALG != "all" ]]; then
    export NCCL_ALGO=$COMM_ALG
    if [[ $COMM_ALG =~ "collnet" ]]; then
        export NCCL_COLLNET_ENABLE=1
    fi
    if [[ $COMM_ALG =~ "nvls" ]]; then
        # Enable the use of NVLink SHARP (NVLS). 
        # NVLink SHARP is available in third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures, 
        # allowing collectives such as ncclAllReduce to be offloaded to the NVSwitch domain. 
        # NVLS will be disabled automatically on systems which do not support the feature.
        export NCCL_NVLS_ENABLE=1
    fi
fi


if [[ $MULTI_PROCESS == true ]]; then
    mpirun \
    --allow-run-as-root  \
    -np $WORLD_SIZE -N $GPUS_PER_NODE \
    --bind-to numa \
    $TEST_SCRIPT \
    -b $START_BYTES -e $END_BYTES \
    -f 2 -g 1 \
    | tee $TEST_OUTPUT_PATH
else
    $TEST_SCRIPT \
    -b $START_BYTES -e $END_BYTES \
    -f 2 -g $GPUS_PER_NODE \
    | tee $TEST_OUTPUT_PATH
fi


echo $SEP
echo "Concise NCCL Test Report"
echo $SEP

python ${TEST_SCRIPT_ROOT}/nccl_test_report.py