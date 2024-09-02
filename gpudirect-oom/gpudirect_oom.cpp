#include <cuda_runtime.h>
#include <mpi.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

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

static int mpi_rank{-1};

void malloc_verbose(void** p, std::size_t n) {
    CHECK_CUDA(cudaMalloc(p, n));
    std::ostringstream os;
    os << "rank: " << mpi_rank << ", ";
    os << "allocated ptr: " << *p << " of size: " << n << '\n';
    std::cerr << os.str();
}

void free_verbose(void* p) {
    std::ostringstream os;
    os << "rank: " << mpi_rank << ", ";
    os << "freeing ptr: " << p << '\n';
    std::cerr << os.str();
    CHECK_CUDA(cudaFree(p));
}

void print_cuda_mem_info() {
    std::size_t cuda_free{0};
    std::size_t cuda_total{0};
    cudaMemGetInfo(&cuda_free, &cuda_total);

    std::ostringstream os;
    os << "rank: " << mpi_rank << ", ";
    os << "cuda_free: " << cuda_free << ", ";
    os << "cuda_total: " << cuda_total << '\n';
    std::cerr << os.str();
}

int main(int argc, char* argv[]) {
    CHECK_MPI(MPI_Init(&argc, &argv));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

    // When misbehaving, this should fail in ~10 iterations on a 96GiB GPU. The
    // large allocation per iteration is ~8GiB and the small allocation of 2MiB
    // is insignificant in comparison.
    const std::size_t niter = 100;
    std::vector<void*> ptrs(niter, nullptr);
    for (std::size_t iter = 0; iter < niter; ++iter) {
        std::ostringstream os;
        os << "rank: " << mpi_rank << ", ";
        os << "iteration " << iter << '\n';
        std::cerr << os.str();

        std::size_t size =
            (std::size_t(1) << 33) + iter * (std::size_t(1) << 21);
        void* p = nullptr;
        malloc_verbose(&p, size);
        malloc_verbose(&ptrs[iter], std::size_t(1) << 21);

        if (mpi_rank == 0) {
            CHECK_MPI(MPI_Send(p, size, MPI_CHAR, 1, 0, MPI_COMM_WORLD));
        } else {
            MPI_Status status;
            CHECK_MPI(MPI_Recv(p, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status));
        }

        free_verbose(p);

        print_cuda_mem_info();
    }

    for (auto ptr : ptrs) {
        free_verbose(ptr);
    }

    CHECK_MPI(MPI_Finalize());
    print_cuda_mem_info();
}
