#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include "cuda.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include "init_send_buffer.h"


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


void checkCorrect(
    int device_id, int num_devices,
    std::vector<float>& host_send_buffer, std::vector<float>& host_recv_buffer,
    float* send_buffer, float* recv_buffer, 
    int comm_size, int comm_size_byte,
    float init_value, float init_value_sum,
    bool is_init = true
) {
    CUDACHECK(cudaSetDevice(device_id));
    CUDACHECK(cudaMemcpy(host_send_buffer.data(), send_buffer, comm_size_byte, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(host_recv_buffer.data(), recv_buffer, comm_size_byte, cudaMemcpyDeviceToHost));

    float send_sum = accumulate(host_send_buffer.begin(), host_send_buffer.end(), 0.);
    float recv_sum = accumulate(host_recv_buffer.begin(), host_recv_buffer.end(), 0.);
    // print send/recv buffer
    // std::cout << "Device " << i << ": " << std::endl;
    // std::cout << "  Send buffer: ";
    // for (size_t j = 0; j < comm_size; ++j) {
    //     std::cout << host_send_buffer[j] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "  Recv buffer: ";
    // for (size_t j = 0; j < comm_size; ++j) {
    //     std::cout << host_recv_buffer[j] << " ";
    // }
    // std::cout << std::endl;

    float expected_send_sum = (float) comm_size * init_value;
    float expected_recv_sum = is_init ? (float) 0 : (((float) comm_size * init_value_sum) / num_devices);

    bool send_buffer_corr = send_sum == expected_send_sum ? true : false;
    bool recv_buffer_corr = recv_sum == expected_recv_sum ? true : false;
    std::cout << "For device " << device_id \
    << (is_init ? ", after initialization" : ", after all-to-all") \
    << ", is send buffer correctly: " << (
        send_buffer_corr ? "yes" : "no"
    ) << " (with send sum: " << send_sum << " | expected: " << expected_send_sum << ")" \
    << " | is recv buffer correctly: " << (
        recv_buffer_corr ? "yes" : "no"
    )  << " (with recv sum: " << recv_sum << " | expected: " << expected_recv_sum << ")" \
    << std::endl;
}

int main(int argc, char* argv[]) {
    // define devices
    int num_devices = 4;
    int devlist[num_devices] = {0, 1, 2, 3};
    // std::cout << "devlist: ";
    // for (int i = 0; i < num_devices; ++i) {
    //     std::cout << "rank" << devlist[i];
    //     if (i < num_devices - 1) {
    //         std::cout << ", ";
    //     }
    // }
    // std::cout << std::endl;

    // allocate send/recv buffer and cuda stream for each device
    // std::cout << "sizeof(float): " << sizeof(float) << std::endl;
    int chunk_size = 4 * 1024; int comm_size = num_devices * chunk_size; int comm_size_byte = comm_size * sizeof(float); // 4K * 8 * sizeof(float) = 32K * 4B = 128KB
    float** send_buffer_list = (float**)malloc(num_devices * sizeof(float*));
    float** recv_buffer_list = (float**)malloc(num_devices * sizeof(float*));
    cudaStream_t* stream_list = (cudaStream_t*)malloc(num_devices * sizeof(cudaStream_t));

    float init_value_sum = 0.f;
    for (int i = 0; i < num_devices; ++i) {
        float init_value = (float) i + 0.5; // init value for send buffer (float)
        init_value_sum += init_value;

        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void**) send_buffer_list + i, comm_size_byte));
        CUDACHECK(cudaMalloc((void**) recv_buffer_list + i, comm_size_byte));
        initSendBuffer(send_buffer_list[i], comm_size, init_value);
        CUDACHECK(cudaMemset(recv_buffer_list[i], 0, comm_size_byte));
        CUDACHECK(cudaStreamCreate(stream_list + i));
    }
    
    // check if initialization correct
    std::vector<float> host_send_buffer(comm_size);
    std::vector<float> host_recv_buffer(comm_size);
    for (int i = 0; i < num_devices; ++i) {
        float init_value = (float) i + 0.5; // init value for send buffer (float)
        checkCorrect(
            i, num_devices,
            host_send_buffer,
            host_recv_buffer,
            send_buffer_list[i],
            recv_buffer_list[i],
            comm_size,
            comm_size_byte,
            init_value, init_value_sum,
            true
        );
    }

    // init nccl comm object for each device
    ncclComm_t comms[num_devices];
    NCCLCHECK(ncclCommInitAll(comms, num_devices, devlist));

    // call nccl comm primitives for each device.
    // where group API is required when using multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < num_devices; ++i) {
        for (int j = 0; j < num_devices; ++j) {
            NCCLCHECK(ncclSend(
                (const void*) (send_buffer_list[i] + j * chunk_size),
                chunk_size, 
                ncclFloat, 
                j, 
                comms[i], 
                stream_list[i]
            ));
            NCCLCHECK(ncclRecv(
               (void*) (recv_buffer_list[i] + j * chunk_size),
                chunk_size,
                ncclFloat,
                j,
                comms[i],
                stream_list[i]
            ));
        }
    }
    NCCLCHECK(ncclGroupEnd());

    // sync cuda stream for each device
    for (int i = 0; i < num_devices; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(stream_list[i]));
    }

    // check if all reduce correct
    for (int i = 0; i < num_devices; ++i) {
        float init_value = (float) i + 0.5; // init value for send buffer (float)
        checkCorrect(
            i, num_devices,
            host_send_buffer,
            host_recv_buffer,
            send_buffer_list[i],
            recv_buffer_list[i],
            comm_size,
            comm_size_byte,
            init_value, init_value_sum,
            false
        );
    }

    // free send/recv buffers
    for (int i = 0; i < num_devices; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(send_buffer_list[i]));
        CUDACHECK(cudaFree(recv_buffer_list[i]));
    }

    // destroy nccl comm objects
    for (int i = 0; i < num_devices; ++i)
        NCCLCHECK(ncclCommDestroy(comms[i]));

    return 0;
}