//
// uenv start prgenv-gnu-openmpi/25.12:v1 --view=default
// MPIDIR=$(dirname ${MPICC})/../
// CC=mpicc CXX=mpicxx nvcc -o gpu590 gpu590.cu -g -std=c++17 -L${MPIDIR}/lib -I${MPIDIR}/include -Xcompiler "-fPIC" -L/lib/aarch64-linux-gnu/ -lcudart -lmpi -Xlinker /usr/lib64/libcxi.so.1 -Xlinker /usr/lib64/libnl-3.so.200
//
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, -1);                    \
        }                                                     \
    } while (0)

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) 
    {
        if (rank == 0)
            fprintf(stderr, "Run with exactly 2 ranks on same node\n");
        MPI_Finalize();
        return 1;
    }

    // Select GPU based on rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const int N = 8 * 1024 * 1024 * sizeof(float); // 8M floats (~32 MB)
    float* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, N));
    CUDA_CHECK(cudaMemset(d_buffer, rank, N));

    MPI_Barrier(MPI_COMM_WORLD);

    float* h_buffer = (float *)malloc(N);
    cudaMemcpy(h_buffer, d_buffer, N, cudaMemcpyDeviceToHost);

    if (rank == 0) 
    {
        MPI_Send(d_buffer, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } 
    else 
    {
        MPI_Recv(d_buffer, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_buffer));

    MPI_Finalize();

    if (rank == 0)
    {
        fprintf(stdout, "OK\n");
    }
    return 0;
}

