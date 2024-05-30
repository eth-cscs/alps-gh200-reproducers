#include "memory.hpp" 
#include "allreduce.hpp"

#include <iomanip>
#include <chrono>
#include <string>
#include <string_view>
#include <vector>

namespace test {

inline mem_type parse_mem_type(std::string_view s) {
    if (s == "host") {
        return mem_type::host;
    } else if (s == "pinned_host") {
        return mem_type::pinned_host;
    } else if (s == "mpi_host") {
        return mem_type::mpi_host;
    } else if (s == "device") {
        return mem_type::device;
    } else {
        std::cout << "Memory type must be host, pinned_host, or device; got \""
                  << s << "\"\n";
        std::exit(1);
    }
}

} // namespace test

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cout << "Usage: allreduce_bench <num_iterations> "
                     "<num_sub_iterations> <memory_type> <size> <padding> <naive>\n"
                     "got " << (argc-1) << " args\n";
        std::exit(1);
    }

    CHECK_MPI(MPI_Init(&argc, &argv));

    int comm_rank;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
    int comm_size;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

    std::size_t const num_iterations = std::stoul(argv[1]);
    std::size_t const num_sub_iterations = std::stoul(argv[2]);
    test::mem_type const t = test::parse_mem_type(argv[3]);
    std::size_t const size = std::stoul(argv[4]);
    std::size_t const padding = std::stoul(argv[5]);
    bool const naive = std::stoul(argv[6]);

    CHECK_CUDA(cudaDeviceSynchronize());
    if (comm_rank == 0) {
        std::cout << "mem_type:   " << argv[3] << '\n';
        std::cout << "size:       " << size << '\n';
        std::cout << "padding:    " << padding << '\n';
        std::cout << "naive algo: " << naive << '\n';
        std::cout << std::endl;
    }

    using type = float;
    auto mpi_type = MPI_FLOAT;
    
    const auto num_data_transfers = 2*(comm_size - 1);
    const double data_transfers_per_rank = (double)num_data_transfers/comm_size;

    std::vector<std::vector<double>> times(num_iterations);
    std::vector<double> start_times(num_iterations);
    std::vector<std::vector<double>> bws(num_iterations);
    std::vector<double> start_bws(num_iterations);
    for (auto& v : times) v.resize(num_sub_iterations-1);
    for (auto& v : bws) v.resize(num_sub_iterations-1);

    // Warmup
    {
        const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();

        {
            auto buffer = test::buffer<type>(t, size, padding);
            test::fill(buffer, (type)1);
            if (!naive)
                test::all_reduce(buffer, mpi_type);
            else
                test::all_reduce_naive_ring(buffer);

        }

        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        auto const stop = std::chrono::steady_clock::now();
        if (comm_rank == 0) {
            std::cout << "[-1] time: "
                      << std::chrono::duration<double>(stop - start).count()
                      << '\n';
        }
    }

    for (std::size_t i = 0; i < num_iterations; ++i) {
        auto buffer = test::buffer<type>(t, size, padding);
        test::fill(buffer, (type)1);

        for (std::size_t j = 0; j < num_sub_iterations; ++j) {
            const std::chrono::time_point<std::chrono::steady_clock> start =
                std::chrono::steady_clock::now();

            if (!naive)
                test::all_reduce(buffer, mpi_type);
            else
                test::all_reduce_naive_ring(buffer);

            CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

            auto const stop = std::chrono::steady_clock::now();
            auto const delta = std::chrono::duration<double>(stop - start).count();

            const double algbw = (size*sizeof(type))/delta*10e-9;     //GB/s
            const double bw = algbw * data_transfers_per_rank;
            if (comm_rank == 0) {
                std::cout << "[" << i << ":" << j << "] time: " << std::setw(12) << delta << "  bw: " << std::setw(12) << bw << '\n';
            }
            if (j==0) {
                start_times[i] = delta;
                start_bws[i] = bw;
            }
            else {
                times[i][j-1] = delta;
                bws[i][j-1] = bw;
            }
        }
    }

    auto _mean = [](std::vector<double> const & v, double& mu, std::size_t& n) {
        for (auto x : v) {
            double const delta = x - mu;
            mu += delta/(++n);
        }
    };

    auto mean = [&](std::vector<std::vector<double>> const & v) {
        double mu = 0;
        std::size_t n = 0;
        for (auto& x : v) _mean(x, mu, n);
        return mu;
    };

    if (comm_rank == 0) {
        std::cout << "\n=======================================";
        std::cout << "\ntype:  " << std::setw(12) << argv[3];
        std::cout << "\nsize:  " << std::setw(12) << size*sizeof(type);
        std::cout << "\npad:   " << std::setw(12) << padding*sizeof(type);
        std::cout << "\nnaive: " << std::setw(12) << naive;
        std::cout << "\ntime:  " << std::setw(12) << mean(times);
        std::cout << "\ntime0: " << std::setw(12) << mean({start_times});
        std::cout << "\nbw:    " << std::setw(12) << mean(bws);
        std::cout << "\nbw0:   " << std::setw(12) << mean({start_bws});
        std::cout << "\n=======================================";
        std::cout << "\n";
        std::cout << std::endl;
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    CHECK_MPI(MPI_Finalize());

    return EXIT_SUCCESS;
}
