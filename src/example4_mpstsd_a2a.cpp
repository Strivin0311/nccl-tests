#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_profiler_api.h>
#include "nccl.h"
#include "mpi.h"
#include "init_send_buffer.h"
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


bool checkCorrect(
    std::vector<float>& host_send_buffer,
    std::vector<float>& host_recv_buffer,
    float* send_buffer,
    float* recv_buffer,
    int rank_id,
    int comm_size,
    int comm_size_byte,
    int chunk_size,
    float init_offset
) {
    CUDACHECK(cudaMemcpy(host_send_buffer.data(), send_buffer, comm_size_byte, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(host_recv_buffer.data(), recv_buffer, comm_size_byte, cudaMemcpyDeviceToHost));

    bool send_buffer_correct = true;
    bool recv_buffer_correct = true;

    // check send buffer
    for (int i = 0; i < comm_size; ++i) {
        float expected_value = rank_id + init_offset;
        if (host_send_buffer[i] != expected_value) {
            std::cout << "For rank " << rank_id \
            << " Unexpected value in send buffer at idx " << i << " : " \
            << host_send_buffer[i] << " | " << expected_value \
            << std::endl;
            send_buffer_correct = false;
        }
    }

    // check recv buffer
    for (int i = 0; i < comm_size; ++i) {
        int chunk_id = i / chunk_size;
        float expected_value = chunk_id + init_offset;
        if (host_recv_buffer[i] != expected_value) {
            std::cout << "For rank " << rank_id \
            << " Unexpected value in recv buffer at idx " << i << " : " \
            << host_recv_buffer[i] << " | " << expected_value \
            << std::endl;
            recv_buffer_correct = false;
        }
    }

    if (send_buffer_correct) {
        std::cout << "Rank " << rank_id << " send buffer is correct" << std::endl;
    }
    if (recv_buffer_correct) {
        std::cout << "Rank " << rank_id << " recv buffer is correct" << std::endl;
    }

    return send_buffer_correct & recv_buffer_correct;
}


int main(int argc, char* argv[]) {
    // init MPI to get num_ranks and this_rank id
    int this_rank, num_ranks = 0;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &this_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

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

    // allocate and init send/recv buffer
    int comm_size = 4 * 1024; int comm_size_byte = comm_size * sizeof(float); // 32K * sizeof(float) = 32K * 4B = 128KB
    int chunk_size = comm_size / num_ranks;
    float init_offset = 0.5;
    float init_value = (float) this_rank + init_offset; // init value for send buffer (float)
    float *send_buffer, *recv_buffer;
    CUDACHECK(cudaMalloc(&send_buffer, comm_size_byte)); initSendBuffer(send_buffer, comm_size, init_value);
    CUDACHECK(cudaMalloc(&recv_buffer, comm_size_byte));
    
    // init nccl comm object
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, num_ranks, nccl_uid, this_rank));

    // call nccl comm primitives
    CUDACHECK(cudaProfilerStart());
    nvtxRangePushA("nccl all2all");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < num_ranks; ++i) {
        for (int j = 0; j < num_ranks; ++j) {
            NCCLCHECK(ncclSend(
                (const void*) (send_buffer + j * chunk_size),
                chunk_size,
                ncclFloat,
                j,
                comm,
                stream
            ));
            NCCLCHECK(ncclRecv(
                (void *) (recv_buffer + j * chunk_size),
                chunk_size,
                ncclFloat,
                j,
                comm,
                stream
            ));
        }
    }
    NCCLCHECK(ncclGroupEnd());
    nvtxRangePop();
    CUDACHECK(cudaProfilerStop());

    // sync cuda stream
    CUDACHECK(cudaStreamSynchronize(stream));

    // check if allreduce correct
    std::vector<float> host_send_buffer(comm_size);
    std::vector<float> host_recv_buffer(comm_size);
    bool success = checkCorrect(
        host_send_buffer,
        host_recv_buffer,
        send_buffer,
        recv_buffer,
        this_rank,
        comm_size,
        comm_size_byte,
        chunk_size,
        init_offset
    );

    // free send/recv buffer
    CUDACHECK(cudaFree(send_buffer));
    CUDACHECK(cudaFree(recv_buffer));

    // destroy nccl comm object
    ncclCommDestroy(comm);

    // destroy cuda stream
    CUDACHECK(cudaStreamDestroy(stream));

    // finalize MPI
    MPICHECK(MPI_Finalize());
    
    if (!success) {
        std::cout << "[MPI Rank " << this_rank << " of " << num_ranks << " Ranks] Failed" << "\n";
        return 1;
    }
    std::cout << "[MPI Rank " << this_rank << " of " << num_ranks << " Ranks] Success" << "\n";
    return 0;
}