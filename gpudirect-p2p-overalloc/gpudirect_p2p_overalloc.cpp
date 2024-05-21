#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

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

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cout << "Usage: gpudirect_p2p_overalloc <num_iterations> "
                     "<num_sub_iterations> <p2p_size> <buffer_size> "
		     "<overalloc_size>\n";
        std::terminate();
    }

    CHECK_MPI(MPI_Init(&argc, &argv));

    int comm_rank;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
    int comm_size;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

    const auto rank_recv = comm_size - 1;

    std::size_t const num_iterations = std::stoul(argv[1]);
    std::size_t const num_sub_iterations = std::stoul(argv[2]);
    std::size_t const p2p_size = std::stoul(argv[3]);
    std::size_t const buffer_size = std::stoul(argv[4]);
    std::size_t const overalloc_size = std::stoul(argv[5]);

    if (comm_rank == 0) {
        std::cout << "p2p_size: " << p2p_size << '\n';
        std::cout << "buffer_size: " << buffer_size << '\n';
        std::cout << "overalloc_size: " << overalloc_size << '\n';
    }

    if (p2p_size > buffer_size + overalloc_size) {
        if (comm_rank == 0) {
            std::cout << "p2p_size (" << p2p_size
                      << ") must be at most buffer_size + overalloc_size ("
                      << (buffer_size + overalloc_size) << ")\n";
        }
        std::exit(1);
    }

    // Warmup
    {
        const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();

        if (comm_rank == 0) {
            char* mem = nullptr;
            CHECK_CUDA(cudaMalloc(&mem, p2p_size));
            CHECK_MPI(MPI_Send(mem, p2p_size, MPI_CHAR, rank_recv, 0,
                               MPI_COMM_WORLD));
            CHECK_CUDA(cudaFree(mem));
        } else if (comm_rank == rank_recv) {
            char* mem = nullptr;
            CHECK_CUDA(cudaMalloc(&mem, p2p_size));
            MPI_Status status;
            CHECK_MPI(MPI_Recv(mem, p2p_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD,
                               &status));
            CHECK_CUDA(cudaFree(mem));
        }

        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        auto const stop = std::chrono::steady_clock::now();
        if (comm_rank == 0) {
            std::cout << "[-1] time: "
                      << std::chrono::duration<double>(stop - start).count()
                      << '\n';
        }
    }

    auto test = [&](bool overalloc) {
        std::size_t const alloc_size =
            buffer_size + (overalloc ? overalloc_size : 0);

        if (comm_rank == 1) {
            std::cout << "Doing MPI_Send/Recv from rank 0 to rank " << rank_recv
                      << ", of " << p2p_size << " bytes from a buffer of "
                      << alloc_size << " bytes ("
                      << (overalloc ? "with overalloc" : "without overalloc")
                      << ").\n";
        }

        for (std::size_t i = 0; i < num_iterations; ++i) {
            char* mem = nullptr;
            if (comm_rank == 0) {
                // Only send side is affected
                CHECK_CUDA(cudaMalloc(&mem, alloc_size));
            } else if (comm_rank == rank_recv) {
                CHECK_CUDA(cudaMalloc(&mem, p2p_size));
	    }

	    for (std::size_t j = 0; j < num_sub_iterations; ++j) {
                const std::chrono::time_point<std::chrono::steady_clock> start =
                    std::chrono::steady_clock::now();

                if (comm_rank == 0) {
                    CHECK_MPI(MPI_Send(mem, p2p_size, MPI_CHAR, rank_recv, 0,
                                       MPI_COMM_WORLD));
                } else if (comm_rank == rank_recv) {
                    MPI_Status status;
                    CHECK_MPI(MPI_Recv(mem, p2p_size, MPI_CHAR, 0, 0,
                                       MPI_COMM_WORLD, &status));
                }

                CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

                auto const stop = std::chrono::steady_clock::now();
                if (comm_rank == 0) {
                    std::cout << "[" << i << ":" << j << "] time: "
                              << std::chrono::duration<double>(stop - start).count()
                              << '\n';
                }
	    }

            CHECK_CUDA(cudaFree(mem));
        }
    };

    test(true);
    test(false);

    return EXIT_SUCCESS;
}
