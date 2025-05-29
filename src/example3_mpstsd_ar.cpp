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
#include "nccl.h"
#include "mpi.h"
#include "init_send_buffer.h"


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


static uint64_t getHash(const char* string) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
    char hostHash[1024];

    // Fall back is the hostname if something fails
    (void) strncpy(hostHash, hostname, sizeof(hostHash));
    int offset = strlen(hostHash);

    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
        char *p;
        if (fscanf(file, "%ms", &p) == 1) {
            strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
            free(p);
        }
    }
    fclose(file);

    // Make sure the string is terminated
    hostHash[sizeof(hostHash)-1]='\0';

    return getHash(hostHash);
}


static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
      if (hostname[i] == '.') {
          hostname[i] = '\0';
          return;
      }
    }
}

  void checkCorrect(
    int rank_id, int device_id,
    std::vector<float>& host_send_buffer, std::vector<float>& host_recv_buffer,
    float* send_buffer, float* recv_buffer,
    int comm_size_byte
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

    std::cout << "For rank " << rank_id << " device " << device_id \
    << " send sum: " << send_sum << " | " \
    << " recv sum: " << recv_sum \
    << std::endl;
}


int main(int argc, char* argv[]) {
    // init MPI to get num_ranks and this_rank id
    int this_rank, num_ranks = 0;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &this_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

    // determine device id for this rank
    int device_id = 0;
    uint64_t hostHashs[num_ranks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[this_rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < num_ranks; p++) {
        if (p == this_rank) break;
        if (hostHashs[p] == hostHashs[this_rank]) device_id++;
    }

    // init nccl unique id at rank0 and broadcast to all ranks
    ncclUniqueId nccl_uid;
    if (this_rank == 0) ncclGetUniqueId(&nccl_uid);
    MPICHECK(MPI_Bcast((void *)&nccl_uid, sizeof(nccl_uid), MPI_BYTE, 0, MPI_COMM_WORLD));
    std::stringstream ss;
    for (int i = 0; i < 4; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)(unsigned char)nccl_uid.internal[i];
    }

    // allocate send/recv buffer and cuda stream
    int comm_size = 4 * 1024; int comm_size_byte = comm_size * sizeof(float); // 32K * sizeof(float) = 32K * 4B = 128KB
    float init_value = (float) device_id + 0.5; // init value for send buffer (float)
    float *send_buffer, *recv_buffer;
    cudaStream_t stream;
    CUDACHECK(cudaSetDevice(device_id));
    CUDACHECK(cudaMalloc(&send_buffer, comm_size_byte)); initSendBuffer(send_buffer, comm_size, init_value);
    CUDACHECK(cudaMalloc(&recv_buffer, comm_size_byte));
    CUDACHECK(cudaStreamCreate(&stream));

    // init nccl comm object
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, num_ranks, nccl_uid, this_rank));

    // call nccl comm primitives
    NCCLCHECK(ncclAllReduce(
        (const void*) send_buffer,
        (void*) recv_buffer,
        comm_size,
        ncclFloat,
        ncclSum,
        comm,
        stream
    ));

    // sync cuda stream
    CUDACHECK(cudaStreamSynchronize(stream));

    // check if allreduce correct
    std::vector<float> host_send_buffer(comm_size);
    std::vector<float> host_recv_buffer(comm_size);
    checkCorrect(
        this_rank,
        device_id,
        host_send_buffer,
        host_recv_buffer,
        send_buffer,
        recv_buffer,
        comm_size_byte
    );

    // free send/recv buffer
    CUDACHECK(cudaFree(send_buffer));
    CUDACHECK(cudaFree(recv_buffer));

    // destroy nccl comm object
    ncclCommDestroy(comm);

    // finalize MPI
    MPICHECK(MPI_Finalize());
    
    std::cout << "[MPI Rank " << this_rank << " of " << num_ranks << " Ranks] Success" \
    << " with device id " << device_id \
    << " and nccl unique id " << ss.str() << "..." \
    << "\n";

    return 0;


    // // define devices
    // int num_devices = 4;
    // int devlist[num_devices] = {0, 1, 2, 3};
    // // std::cout << "devlist: ";
    // // for (int i = 0; i < num_devices; ++i) {
    // //     std::cout << "rank" << devlist[i];
    // //     if (i < num_devices - 1) {
    // //         std::cout << ", ";
    // //     }
    // // }
    // // std::cout << std::endl;

    // // allocate send/recv buffer and cuda stream for each device
    // // std::cout << "sizeof(float): " << sizeof(float) << std::endl;
    // int comm_size = 32 * 1024; int comm_size_byte = comm_size * sizeof(float); // 32K * sizeof(float) = 32K * 4B = 128KB
    // float** send_buffer_list = (float**)malloc(num_devices * sizeof(float*));
    // float** recv_buffer_list = (float**)malloc(num_devices * sizeof(float*));
    // cudaStream_t* stream_list = (cudaStream_t*)malloc(num_devices * sizeof(cudaStream_t));

    // float init_value_sum = 0.f;
    // for (int i = 0; i < num_devices; ++i) {
    //     float init_value = (float) i + 0.5; // init value for send buffer (float)
    //     init_value_sum += init_value;

    //     CUDACHECK(cudaSetDevice(i));
    //     CUDACHECK(cudaMalloc((void**) send_buffer_list + i, comm_size_byte));
    //     CUDACHECK(cudaMalloc((void**) recv_buffer_list + i, comm_size_byte));
    //     initSendBuffer(send_buffer_list[i], comm_size, init_value);
    //     CUDACHECK(cudaMemset(recv_buffer_list[i], 0, comm_size_byte));
    //     CUDACHECK(cudaStreamCreate(stream_list + i));
    // }
    
    // // check if initialization correct
    // std::vector<float> host_send_buffer(comm_size);
    // std::vector<float> host_recv_buffer(comm_size);
    // for (int i = 0; i < num_devices; ++i) {
    //     float init_value = (float) i + 0.5; // init value for send buffer (float)
    //     checkCorrect(
    //         i,
    //         host_send_buffer,
    //         host_recv_buffer,
    //         send_buffer_list[i],
    //         recv_buffer_list[i],
    //         comm_size,
    //         comm_size_byte,
    //         init_value, init_value_sum,
    //         true
    //     );
    // }

    // // init nccl comm object for each device
    // ncclComm_t comms[num_devices];
    // NCCLCHECK(ncclCommInitAll(comms, num_devices, devlist));

    // // call nccl comm primitives for each device.
    // // where group API is required when using multiple devices per thread
    // NCCLCHECK(ncclGroupStart());
    // for (int i = 0; i < num_devices; ++i)
    //     NCCLCHECK(ncclAllReduce(
    //         (const void*) send_buffer_list[i], 
    //         (void*) recv_buffer_list[i], 
    //         comm_size, 
    //         ncclFloat,
    //         ncclSum,
    //         comms[i], 
    //         stream_list[i]
    //     ));
    // NCCLCHECK(ncclGroupEnd());

    // // sync cuda stream for each device
    // for (int i = 0; i < num_devices; ++i) {
    //     CUDACHECK(cudaSetDevice(i));
    //     CUDACHECK(cudaStreamSynchronize(stream_list[i]));
    // }

    // // check if all reduce correct
    // for (int i = 0; i < num_devices; ++i) {
    //     float init_value = (float) i + 0.5; // init value for send buffer (float)
    //     checkCorrect(
    //         i,
    //         host_send_buffer,
    //         host_recv_buffer,
    //         send_buffer_list[i],
    //         recv_buffer_list[i],
    //         comm_size,
    //         comm_size_byte,
    //         init_value, init_value_sum,
    //         false
    //     );
    // }

    // // free send/recv buffers
    // for (int i = 0; i < num_devices; ++i) {
    //     CUDACHECK(cudaSetDevice(i));
    //     CUDACHECK(cudaFree(send_buffer_list[i]));
    //     CUDACHECK(cudaFree(recv_buffer_list[i]));
    // }

    // // destroy nccl comm objects
    // for (int i = 0; i < num_devices; ++i)
    //     NCCLCHECK(ncclCommDestroy(comms[i]));

    // return 0;
}