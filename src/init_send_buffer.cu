#include "cuda.h"
#include "cuda_runtime.h"


template<typename T>
__global__ void initMatrix(T* mat, int width, int height, T val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x) {
        mat[i] = val;
    }
}


extern "C" void initSendBuffer(float* send_buffer, int comm_size, float init_value) {
    
    /*  method 1 (failed) since you can NOT use float value to init with cudaMemset */
    // you cannot 
    // int init_value_int_repr = reinterpret_cast<int&>(init_value); // reinterpret to int
    //     std::cout << "For device " << i << ", " \
    //     << "init value: " << init_value \
    //     << " and reinterpreted to int: " << init_value_int_repr << std::endl;
    // CUDACHECK(cudaMemset(send_buffer_list[i], init_value_int_repr, comm_size_byte));
   
    /*  method2 (succeeded) with a simple cuda kernel to fill the float value */
    int blockSize = 256;
    int numBlocks = (comm_size + blockSize - 1) / blockSize;
    initMatrix<<<numBlocks, blockSize>>>(send_buffer, comm_size, 1, init_value);
}

