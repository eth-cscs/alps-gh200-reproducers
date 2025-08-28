#pragma once

#include "memory.hpp"
#include  <cstring>

namespace test {

template<typename T>
void all_reduce(buffer<T>& buffer, MPI_Datatype mpi_type) {
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), mpi_type, MPI_SUM, MPI_COMM_WORLD));
}

template<typename T>
void sum_device(T* dest, T const * src, std::size_t size);

template<typename T>
void sum(buffer<T>& dest, buffer<T> const & src) {
    switch (dest.type()) {
        case mem_type::device:
            sum_device(dest.data(), src.data(), dest.size());
            break;

        default:
            for (std::size_t i=0; i<dest.size(); ++i) {
                dest.data()[i] += src.data()[i];
            }
    }
}

template<typename T>
void copy(buffer<T>& dest, buffer<T> const & src) {
    if (dest.type() == mem_type::device) {
        if (src.type() == mem_type::device) {
            CHECK_CUDA(cudaMemcpy(dest.data(), src.data(), dest.size()*sizeof(T), cudaMemcpyDeviceToDevice));
        }
        else {
            CHECK_CUDA(cudaMemcpy(dest.data(), src.data(), dest.size()*sizeof(T), cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (src.type() == mem_type::device) {
        CHECK_CUDA(cudaMemcpy(dest.data(), src.data(), dest.size()*sizeof(T), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else {
        std::memcpy(dest.data(), src.data(), dest.size()*sizeof(T));
    }
}

template<typename T>
bool check(const buffer<T>& b, T expected) {
    buffer<T> c(mem_type::host, b.size(), 0);
    copy(c, b);
    for (std::size_t i=0; i<b.size(); ++i) {
        if (c.data()[i] != expected) {
            std::cout << "Allreduce went wrong at index = " << i << ": expexted " << expected << " got " << c.data()[i] << '\n';
            return false;
        }
    }
    return true;
}

template<typename T>
void all_reduce_naive_ring(buffer<T>& b, MPI_Datatype mpi_type) {
    static thread_local std::unique_ptr<buffer<T>> recv_buffer_ptr;
    if (!recv_buffer_ptr || 
        (recv_buffer_ptr->size() < b.size()) || 
        (recv_buffer_ptr->type() != b.type()) || 
        (recv_buffer_ptr->padding() != b.padding())) {
        recv_buffer_ptr.reset();
        recv_buffer_ptr = std::make_unique<buffer<T>>(b.type(), b.size(), b.padding());
    }
    auto& recv_buffer = *recv_buffer_ptr;

    static thread_local std::unique_ptr<buffer<T>> send_buffer_ptr;
    if (!send_buffer_ptr || 
        (send_buffer_ptr->size() < b.size()) || 
        (send_buffer_ptr->type() != b.type()) || 
        (send_buffer_ptr->padding() != b.padding())) {
        send_buffer_ptr.reset();
        send_buffer_ptr = std::make_unique<buffer<T>>(b.type(), b.size(), b.padding());
    }
    auto& send_buffer = *send_buffer_ptr;

    int comm_rank;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
    int comm_size;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

    copy(send_buffer, b);
    auto const recv_rank = (comm_rank + 1) % comm_size;
    auto const send_rank = ((comm_rank + comm_size) - 1) % comm_size;
    
    auto* s = &send_buffer;
    auto* r = &recv_buffer;

    for (int i=0; i<comm_size-1; ++i) {
        CHECK_MPI(MPI_Sendrecv(
            s->data(), s->size(), mpi_type, send_rank, 0,
            r->data(), s->size(), mpi_type, recv_rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        sum(b, *r);
        std::swap(s, r);
    }
}

} // namespace test
