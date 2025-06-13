#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <assert.h>
#include <vector>
#include <numeric>
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_profiler_api.h>
#include "nccl.h"
#include "mpi.h"
#include "nvtx3/nvToolsExt.h"


#define MPICHECK(cmd) do {                              \
    int err = cmd;                                      \
    if (err != MPI_SUCCESS) {                           \
        printf(                                         \
            "Falied, MPI Error: %s:%d '%d'\n",          \
            __FILE__, __LINE__, err                     \
        );                                              \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while (0)


#define CUDACHECK(cmd) do {                             \
    cudaError_t err = cmd;                              \
    if (err != cudaSuccess) {                           \
        printf(                                         \
            "Falied, Cuda Error: %s:%d '%s'\n",         \
            __FILE__, __LINE__, cudaGetErrorString(err) \
        );                                              \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)


#define NCCLCHECK(cmd) do {                             \
    ncclResult_t res = cmd;                             \
    if (res != ncclSuccess) {                           \
        printf(                                         \
            "Failed, NCCL Error: %s:%d '%s'\n",         \
            __FILE__, __LINE__, ncclGetErrorString(res) \
        );                                              \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while (0)


#define RESET_TXT "\033[0m"
#define RED_TXT "\033[1;31m"

inline void assert_fail_msg(const char* msg) {
    printf(RED_TXT "%s\n" RESET_TXT, msg);
}

#define Assert(cond, msg) \
    do { \
        if (!(cond)) { \
            assert_fail_msg(msg); \
        } \
        assert(cond); \
    } while(0)


// #define NATIVE_ALL2ALL
// #define NATIVE_ALL2ALL_V


bool checkRecvCorrect(
    std::vector<int>& host_recv_buffer,
    std::vector<int>& expected_recv_buffer,
    int* recv_buffer,
    int rank_id,
    int recv_size,
    int recv_size_byte
) {
    CUDACHECK(cudaMemcpy(host_recv_buffer.data(), recv_buffer, recv_size_byte, cudaMemcpyDeviceToHost));

    bool recv_buffer_correct = true;

    // check recv buffer
    std::cout << "[Rank " << rank_id << "] " << "actual recv buffer: ";
    for (int i = 0; i < recv_size; ++i) {
        std::cout << host_recv_buffer[i] << ", ";
    } std::cout << std::endl;
    
    for (int i = 0; i < recv_size; ++i) {
        if (host_recv_buffer[i] != expected_recv_buffer[i]) {
            std::cout << "[Rank " << rank_id << "] " \
            << " Unexpected value in recv buffer at idx " << i << " : actual=" \
            << host_recv_buffer[i] << " | expected=" << expected_recv_buffer[i] \
            << std::endl;
            recv_buffer_correct = false;
        }
    }

    return recv_buffer_correct;
}


int main(int argc, char* argv[]) {
    // init MPI to get num_ranks and this_rank id
    int this_rank, num_ranks = 0;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &this_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));
    Assert(num_ranks == 4, "Only support 4 ranks for this example");

    // init nccl unique id at rank0 and broadcast to all ranks
    ncclUniqueId nccl_uid;
    if (this_rank == 0) ncclGetUniqueId(&nccl_uid);
    MPICHECK(MPI_Bcast((void *)&nccl_uid, sizeof(nccl_uid), MPI_BYTE, 0, MPI_COMM_WORLD));
    std::stringstream ss;
    for (int i = 0; i < 4; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)(unsigned char)nccl_uid.internal[i];
    }

    // set device and stream
    cudaStream_t stream;
    CUDACHECK(cudaSetDevice(this_rank));
    CUDACHECK(cudaStreamCreate(&stream));

    // init send size
    #ifdef NATIVE_ALL2ALL
    int send_size = 4; // 4 ints
    #elifdef NATIVE_ALL2ALL_V
    int send_size = 8; // 8 ints
    #else
    int send_size = 4; // 4 ints
    #endif
    int send_size_byte = send_size * sizeof(int);

    // allocate and init send buffer on both host and device
    std::vector<int> host_send_buffer(send_size);
    for (int i = 0; i < send_size; ++i) {
        #ifdef NATIVE_ALL2ALL
        //  r0: [0, 1, 2, 3]
        //  r1: [4, 5, 6, 7]
        //  r2: [8, 9, 10, 11]
        //  r3: [12, 13, 14, 15]
        #elifdef NATIVE_ALL2ALL_V
        //  r0: [0, 1, 2, 3, 4, 5, 6, 7]
        //  r1: [8, 9, 10, 11, 12, 13, 14, 15]
        //  r2: [16, 17, 18, 19, 20, 21, 22, 23]
        //  r3: [24, 25, 26, 27, 28, 29, 30, 31]
        #else
        //  r0: [0, 1, 2, 3]
        //  r1: [4, 5, 6, 7]
        //  r2: [8, 9, 10, 11]
        //  r3: [12, 13, 14, 15]
        #endif
        host_send_buffer[i] = i + this_rank * send_size;
    }

    int *send_buffer;
    CUDACHECK(cudaMalloc(&send_buffer, send_size_byte));
    CUDACHECK(cudaMemcpy(send_buffer, host_send_buffer.data(), send_size_byte, cudaMemcpyHostToDevice));

    // init meta args and expected recv buffer as ground truth for each rank
    #ifdef NATIVE_ALL2ALL
    std::vector<std::vector<int>> input_split_size_per_rank = {
        {1, 1, 1, 1}, // rank0
        {1, 1, 1, 1}, // rank1
        {1, 1, 1, 1}, // rank2
        {1, 1, 1, 1}  // rank3
    };
    std::vector<std::vector<int>> output_split_size_per_rank = {
        {1, 1, 1, 1}, // rank0
        {1, 1, 1, 1}, // rank1
        {1, 1, 1, 1}, // rank2
        {1, 1, 1, 1}  // rank3
    };
    std::vector<std::vector<int>> expected_recv_buffer_per_rank = {
        {0, 4, 8, 12}, // rank0
        {1, 5, 9, 13}, // rank1
        {2, 6, 10, 14}, // rank2
        {3, 7, 11, 15}  // rank3
    };
    std::vector<std::vector<std::vector<int>>> dst_indices_list_per_rank = {
        {{0}, {1}, {2}, {3}}, // rank0
        {{0}, {1}, {2}, {3}}, // rank1
        {{0}, {1}, {2}, {3}}, // rank2
        {{0}, {1}, {2}, {3}}  // rank3
    };
    std::vector<std::vector<int>> src_index_list_per_rank = {
        {0, 1, 2, 3}, // rank0
        {0, 1, 2, 3}, // rank1
        {0, 1, 2, 3}, // rank2
        {0, 1, 2, 3}  // rank3
    };
    #elifdef NATIVE_ALL2ALL_V
    std::vector<std::vector<int>> input_split_size_per_rank = {
        {2, 2, 2, 2}, // rank0
        {1, 3, 1, 3}, // rank1
        {3, 1, 1, 3}, // rank2
        {2, 1, 2, 3}  // rank3
    };
    std::vector<std::vector<int>> output_split_size_per_rank = {
        {2, 1, 3, 2}, // rank0
        {2, 3, 1, 1}, // rank1
        {2, 1, 1, 2}, // rank2
        {2, 3, 3, 3}  // rank3
    };
    std::vector<std::vector<int>> expected_recv_buffer_per_rank = {
        {0, 1, 8, 16, 17, 18, 24, 25}, // rank0
        {2, 3, 9, 10, 11, 19, 26}, // rank1
        {4, 5, 12, 20, 27, 28}, // rank2
        {6, 7, 13, 14, 15, 21, 22, 23, 29, 30, 31}  // rank3
    };
    std::vector<std::vector<std::vector<int>>> dst_indices_list_per_rank = {
        {{0}, {1}, {2}, {3}}, // rank0
        {{0}, {1}, {2}, {3}}, // rank1
        {{0}, {1}, {2}, {3}}, // rank2
        {{0}, {1}, {2}, {3}}  // rank3
    };
    std::vector<std::vector<int>> src_index_list_per_rank = {
        {0, 1, 2, 3}, // rank0
        {0, 1, 2, 3}, // rank1
        {0, 1, 2, 3}, // rank2
        {0, 1, 2, 3}  // rank3
    };
    #else
    std::vector<std::vector<int>> input_split_size_per_rank = {
        {2, 1, 1}, // rank0
        {1, 1, 2}, // rank1
        {1, 1, 2}, // rank2
        {1, 1, 2}  // rank3
    };
    std::vector<std::vector<int>> output_split_size_per_rank = {
        {1, 1, 1, 2}, // rank0
        
        // {2, 2, 1, 2}, // rank1 => BUG: recv number < send number, causing: [Rank 1] actual recv buffer: 0, 1, 10, 11, 2, 12, 0,
        {2, 2, 1, 1, 1}, // rank1
        
        {1, 2, 2}, // rank2
        
        // {2, 2}  // rank3 => BUG: recv number < send number, causing: [Rank 3] actual recv buffer: 8, 0, 4, 0, 
        {1, 1, 1, 1}  // rank3
    };
    std::vector<std::vector<int>> expected_recv_buffer_per_rank = {
        {5, 9, 13, 0, 1}, // rank0
        {0, 1, 10, 11, 2, 12, 13}, // rank1
        {2, 6, 7, 14, 15}, // rank2
        {8, 9, 4, 5}  // rank3
    };
    // the transfer table w.r.t. number of packages of this group-cast is as follows:
    // (send to) ->     rank0   rank1   rank2   rank3
    // rank0            1       2       1       0
    // rank1            1       0       1       2
    // rank2            1       1       0       2
    // rank3            1       2       1       0
    // total recv       4       5       3       4

    /* 
    * NOTE: As nccl's requirements, the number of ncclRecv calls of each rank should be same as the `total recv` cell in the table
    * i.e., if rankj receives n packages from ranki, then rankj should call ncclRecv() n times,
    * even though these n packages are continuous in rankj's recv buffer and can be combined into one ncclRecv call
    */
    std::vector<std::vector<std::vector<int>>> dst_indices_list_per_rank = {
        {{0, 1}, {1, 2}, {}}, // rank0
        {{3}, {0, 3}, {2}}, // rank1
        {{3}, {0, 3}, {1}}, // rank2
        {{1}, {0, 1}, {2}}  // rank3
    };
    std::vector<std::vector<int>> src_index_list_per_rank = {
        {1, 2, 3, 0}, // rank0
        
        // {0, 2, 0, 3}, // rank1 => BUG: recv number < send number, causing: [Rank 1] actual recv buffer: 0, 1, 10, 11, 2, 12, 0,
        {0, 2, 0, 3, 3}, // rank1
        
        {0, 1, 3}, // rank2

        // {2, 1}  // rank3 => BUG: recv number < send number, causing: [Rank 3] actual recv buffer: 8, 0, 4, 0, 
        {2, 2, 1, 1}  // rank3
    };
    #endif

    // init meta args of group-cast for this rank
    int* input_split_size = input_split_size_per_rank[this_rank].data();
    int num_input_splits = input_split_size_per_rank[this_rank].size();

    int* output_split_size = output_split_size_per_rank[this_rank].data();
    int num_output_splits = output_split_size_per_rank[this_rank].size();

    auto dst_indices_list = dst_indices_list_per_rank[this_rank];
    auto src_index_list = src_index_list_per_rank[this_rank];

    // init expected recv buffer as ground truth, as well as recv size, for this rank
    std::vector<int> expected_recv_buffer = expected_recv_buffer_per_rank[this_rank];
    int recv_size = expected_recv_buffer.size();
    int recv_size_byte = recv_size * sizeof(int);

    // allocate and init recv buffer on both host and device
    std::vector<int> host_recv_buffer(recv_size);
    int *recv_buffer;
    CUDACHECK(cudaMalloc(&recv_buffer, recv_size_byte));

    // initialization check
    std::cout << "[Rank " << this_rank << "] " << "send size: " << send_size << ", recv size: " << recv_size << std::endl;
    std::cout << "[Rank " << this_rank << "] " << "send buffer: ";
    for (int i = 0; i < send_size; ++i) {
        std::cout << host_send_buffer[i] << ", ";
    } std::cout << std::endl;
    std::cout << "[Rank " << this_rank << "] " << "input split size: ";
    for (int i = 0; i < num_input_splits; ++i) {
        std::cout << input_split_size[i] << ", ";
    } std::cout << std::endl;
    std::cout << "[Rank " << this_rank << "] " << "output split size: ";
    for (int i = 0; i < num_output_splits; ++i) {
        std::cout << output_split_size[i] << ", ";
    } std::cout << std::endl;
    std::cout << "[Rank " << this_rank << "] " << "dst indices list: ";
    for (auto dst_indices : dst_indices_list) {
        std::cout << "{";
        for (int dst_rank : dst_indices) {
            std::cout << dst_rank << ",";
        }
        std::cout << "}, ";
    } std::cout << std::endl;
    std::cout << "[Rank " << this_rank << "] " << "src index list: ";
    for (int a : src_index_list) {
        std::cout << a << ", ";
    } std::cout << std::endl;
    std::cout << "[Rank " << this_rank << "] " << "expected recv buffer: ";
    for (int i = 0; i < recv_size; ++i) {
        std::cout << expected_recv_buffer[i] << ", ";
    } std::cout << std::endl;

    // init nccl comm object
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, num_ranks, nccl_uid, this_rank));

    // call nccl comm primitives
    CUDACHECK(cudaProfilerStart());
    nvtxRangePushA("nccl group-cast");
    NCCLCHECK(ncclGroupStart());
    int input_offset = 0, output_offset = 0;
    // int input_split_idx = 0, output_split_idx = 0;
    // for (auto dst_indices : dst_indices_list) {
    //     for (int dst_rank : dst_indices) {
    //         NCCLCHECK(ncclSend(
    //             (const void*) (send_buffer + input_offset),
    //             input_split_size[input_split_idx],
    //             ncclInt32,
    //             dst_rank,
    //             comm,
    //             stream
    //         ));   
    //     }
    //     input_offset += input_split_size[input_split_idx];
    //     input_split_idx++;
    // }
    // for (int src_rank : src_index_list) {
    //     NCCLCHECK(ncclRecv(
    //         (void *) (recv_buffer + output_offset),
    //         output_split_size[output_split_idx],
    //         ncclInt32,
    //         src_rank,
    //         comm,
    //         stream
    //     ));
    //     output_offset += output_split_size[output_split_idx];
    //     output_split_idx++;
    // }
    for (int input_split_idx = 0; input_split_idx < num_input_splits; ++input_split_idx) {
        for (auto dst_rank : dst_indices_list[input_split_idx]) {
            NCCLCHECK(ncclSend(
                (const void*) (send_buffer + input_offset),
                input_split_size[input_split_idx],
                ncclInt32,
                dst_rank,
                comm,
                stream
            ));
        }
        input_offset += input_split_size[input_split_idx];
    }
    for (int output_split_idx = 0; output_split_idx < num_output_splits; ++output_split_idx) {
        NCCLCHECK(ncclRecv(
            (void *) (recv_buffer + output_offset),
            output_split_size[output_split_idx],
            ncclInt32,
            src_index_list[output_split_idx],
            comm,
            stream
        ));
        output_offset += output_split_size[output_split_idx];
    }
    
    NCCLCHECK(ncclGroupEnd());
    nvtxRangePop();
    CUDACHECK(cudaProfilerStop());

    // sync cuda stream
    CUDACHECK(cudaStreamSynchronize(stream));

    // check if all2all-v correct
    bool success = checkRecvCorrect(
        host_recv_buffer,
        expected_recv_buffer,
        recv_buffer,
        this_rank,
        recv_size,
        recv_size_byte
    );
    if (!success) {
        std::cout << "[MPI Rank " << this_rank << " of " << num_ranks << " Ranks] Failed" << "\n";
        return 1;
    }
    std::cout << "[MPI Rank " << this_rank << " of " << num_ranks << " Ranks] Success" << "\n";

    // free send/recv buffer
    CUDACHECK(cudaFree(send_buffer));
    CUDACHECK(cudaFree(recv_buffer));

    // destroy nccl comm object
    ncclCommDestroy(comm);

    // destroy cuda stream
    CUDACHECK(cudaStreamDestroy(stream));

    // finalize MPI
    MPICHECK(MPI_Finalize());

    return 0;
}