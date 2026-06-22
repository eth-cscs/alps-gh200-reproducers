#include "allreduce.hpp"

namespace test {
namespace kernel {

template<typename T>
__global__ void sum(T* dest, T const * src, std::size_t size) {
    auto idx = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] += src[idx];
    }
}

} // namespace kernel

template<>
void sum_device(float* dest, float const * src, std::size_t size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1)/block_size;
    kernel::sum<<<grid_size, block_size>>>(dest, src, size);
    CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace test
