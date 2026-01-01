#pragma once

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <memory>

#define CHECK_MPI(x)                             \
    if (auto e = x; e != MPI_SUCCESS) {          \
        std::cout << "MPI error: " << e << '\n'; \
        std::exit(1);                            \
    }
#define CHECK_CUDA(x)                             \
    if (auto e = x; e != cudaSuccess) {           \
        std::cout << "CUDA error: " << e << '\n'; \
        std::exit(1);                             \
    }

namespace test {

enum class mem_type {
    host,
    pinned_host,
    mpi_host,
    device,
};

template <typename T>
T* malloc(mem_type t, std::size_t n) {
    switch (t) {
        case mem_type::host:
            return static_cast<T*>(::malloc(n));

        case mem_type::pinned_host: {
            void* p = nullptr;
            CHECK_CUDA(cudaMallocHost(&p, n));
            return static_cast<T*>(p);
        }

        case mem_type::mpi_host: {
            void* p = nullptr;
            CHECK_MPI(MPI_Alloc_mem(n, MPI_INFO_NULL, &p));
            return static_cast<T*>(p);
        }

        case mem_type::device: {
            void* p = nullptr;
            CHECK_CUDA(cudaMalloc(&p, n));
            return static_cast<T*>(p);
        }

        default:
            std::terminate();
    }
}

inline void free(mem_type t, void* p) {
    switch (t) {
        case mem_type::host:
            ::free(p);
            break;

        case mem_type::pinned_host:
            CHECK_CUDA(cudaFreeHost(p));
            break;

        case mem_type::mpi_host:
            CHECK_MPI(MPI_Free_mem(p));
            break;

        case mem_type::device:
            CHECK_CUDA(cudaFree(p));
            break;

        default:
            std::terminate();
    }
}

template<typename T>
struct buffer {

    mem_type _type      = mem_type::host;
    std::size_t _size   = 0ul;
    unsigned _padding   = 0ul;
    T* _data            = nullptr;

    buffer(mem_type t, std::size_t size, unsigned padding = 0u)
    : _type{t}
    , _size{size}
    , _padding{padding}
    {
        std::size_t space = (_size + _padding)*sizeof(T);
        T* ptr = malloc<T>(_type, space);
        if (!ptr) {
            std::cout << "allocation failed" << '\n';
            std::exit(1);
        }
        _data = ptr + _padding;
    }

    ~buffer() {
        if (_data) free(_type, _data - _padding);
    }

    buffer(buffer const&) = delete;
    buffer(buffer&& other) = delete;

    mem_type type() const noexcept { return _type; }
    unsigned padding() const noexcept { return _padding; }
    T* data() const noexcept { return _data; }
    std::size_t size() const noexcept { return _size; }
};

template<typename T>
void fill_device(T* ptr, std::size_t size, T def);

template<typename T>
void fill(buffer<T>& b, T def = T{1}) {
    switch (b._type) {
        case mem_type::device:
            fill_device(b.data(), b.size(), def);
            break;

        default:
            std::fill(b.data(), b.data()+b.size(), def);
    }
}

}  // namespace test
