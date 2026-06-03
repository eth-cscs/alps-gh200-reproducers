//
// Benchmark calling cuPointerGetAttribute and cuPointerGetAttributes.  Since
// Nvidia driver 580 the performance of these functions has decreased, and this
// has severely impacted applications. It appears to be called repeatedly by
// CrayMPICH and OpenMPI. All code paths through these functions appear to be
// affected, with the code path associated with an error code most heavily
// impacted.
//
//
// nvcc -O3 ptrbench.cu -lcuda -lnvidia-ml -o ptrbench
// 
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

static double elapsed_us(struct timeval *start, struct timeval *end)
{
    return (end->tv_sec - start->tv_sec) * 1e6 +
           (end->tv_usec - start->tv_usec);
}

static void benchmark_attribute(
    const char *ptr_name,
    const char *attr_name,
    CUpointer_attribute attr,
    void *ptr,
    size_t ptr_size,
    int num_iter)
{
    struct timeval start, end;
    double us;

    int nfail = 0;

    union {
        unsigned int u32;
        int i32;
        CUcontext ctx;
        void *ptr;
    } value;

    gettimeofday(&start, NULL);

    for (int i = 0; i < num_iter; ++i) {

        CUdeviceptr p =
            (CUdeviceptr)((char *)ptr + (i % ptr_size));

        CUresult result =
            cuPointerGetAttribute(
                &value,
                attr,
                p);

        nfail += (result != CUDA_SUCCESS);
    }

    gettimeofday(&end, NULL);

    us = elapsed_us(&start, &end);

    printf("%-24s %-24s : %10.0f us total  %8.3f us/call #fail: %i\n",
           ptr_name,
           attr_name,
           us,
           us / num_iter,
           nfail);
}

static void benchmark_attributes_api(
    const char *ptr_name,
    void *ptr,
    size_t ptr_size,
    int num_iter)
{
    struct timeval start, end;
    int nfail = 0;

    CUpointer_attribute attrs[4] = {
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        CU_POINTER_ATTRIBUTE_IS_MANAGED,
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        CU_POINTER_ATTRIBUTE_CONTEXT
    };

    unsigned int memory_type;
    unsigned int is_managed;
    int device_ordinal;
    CUcontext context;

    void *data[4] = {
        &memory_type,
        &is_managed,
        &device_ordinal,
        &context
    };

    gettimeofday(&start, NULL);

    for (int i = 0; i < num_iter; ++i) {

        CUdeviceptr p =
            (CUdeviceptr)((char *)ptr + (i % ptr_size));

        CUresult result =
            cuPointerGetAttributes(
                4,
                attrs,
                data,
                p);

        assert(result == CUDA_SUCCESS);
        nfail += (result != CUDA_SUCCESS);
    }

    gettimeofday(&end, NULL);

    double us = elapsed_us(&start, &end);

    printf("%-24s %-24s : %10.0f us total  %8.3f us/call #fail: %i\n",
           ptr_name,
           "GetAttributes(4)",
           us,
           us / num_iter,
           nfail);
}

static void benchmark_pointer(
    const char *name,
    void *ptr,
    size_t ptr_size,
    int num_iter)
{
    printf("\n============================================================\n");
    printf("%s\n", name);
    printf("============================================================\n");

    benchmark_attribute(
        name,
        "MEMORY_TYPE",
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        ptr,
        ptr_size,
        num_iter);

    benchmark_attribute(
        name,
        "IS_MANAGED",
        CU_POINTER_ATTRIBUTE_IS_MANAGED,
        ptr,
        ptr_size,
        num_iter);

    benchmark_attribute(
        name,
        "DEVICE_ORDINAL",
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        ptr,
        ptr_size,
        num_iter);

    benchmark_attribute(
        name,
        "CONTEXT",
        CU_POINTER_ATTRIBUTE_CONTEXT,
        ptr,
        ptr_size,
        num_iter);

    benchmark_attributes_api(
        name,
        ptr,
        ptr_size,
        num_iter);
}

int main(int argc, char **argv)
{
    cuInit(0);

    int num_iter = 100000;
    size_t ptr_size = 4096;

    if (argc >= 2)
        num_iter = atoi(argv[1]);

    if (argc >= 3)
        ptr_size = atol(argv[2]);

    int driver_version;
    int runtime_version;

    CUresult cu_result = cuDriverGetVersion(&driver_version);
    assert(cu_result == CUDA_SUCCESS);

    cudaError_t cuda_result = cudaRuntimeGetVersion(&runtime_version);
    assert(cuda_result == cudaSuccess);

    printf("Iterations : %d\n", num_iter);
    printf("Ptr size   : %zu bytes\n", ptr_size);

    printf("CUDA Runtime Version  : %d (%d.%d)\n",
           runtime_version,
           runtime_version / 1000,
           (runtime_version % 1000) / 10);

    printf("CUDA Driver Version   : %d (%d.%d)\n",
           driver_version,
           driver_version / 1000,
           (driver_version % 1000) / 10);

    nvmlReturn_t nvml_result;
    char driver_version_string[80];

    nvml_result = nvmlInit();
    if (nvml_result == NVML_SUCCESS) {
        nvml_result = nvmlSystemGetDriverVersion(
            driver_version_string,
            sizeof(driver_version_string));

        if (nvml_result == NVML_SUCCESS) {
            printf("NVIDIA Driver Version : %s\n",
                   driver_version_string);
        }

        nvmlShutdown();
    }

    void *host_ptr = malloc(ptr_size);
    assert(host_ptr);

    void *device_ptr = NULL;
    cudaError_t cerr =
        cudaMalloc(&device_ptr, ptr_size);
    assert(cerr == cudaSuccess);

    void *managed_ptr = NULL;
    cerr =
        cudaMallocManaged(&managed_ptr, ptr_size);
    assert(cerr == cudaSuccess);

    benchmark_pointer(
        "malloc()",
        host_ptr,
        ptr_size,
        num_iter);

    benchmark_pointer(
        "cudaMalloc()",
        device_ptr,
        ptr_size,
        num_iter);

    benchmark_pointer(
        "cudaMallocManaged()",
        managed_ptr,
        ptr_size,
        num_iter);

    cudaFree(device_ptr);
    cudaFree(managed_ptr);
    free(host_ptr);

    return 0;
}

