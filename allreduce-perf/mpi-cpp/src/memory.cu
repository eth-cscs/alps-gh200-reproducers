#include "memory.hpp"

namespace test {

namespace kernel {

template<typename T>
__global__ void fill(T* ptr, std::size_t size, T def) {
    auto idx = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (idx < size) {
        ptr[idx] = def;
    }
}

} // namespace kernel

template<>
void fill_device<float>(float* ptr, std::size_t size, float def) {
    int block_size = 256;
    int grid_size = (size + block_size - 1)/block_size;
    kernel::fill<<<grid_size, block_size>>>(ptr, size, def);
    CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace test
