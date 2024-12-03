#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

void report_kernel_benchmark(const char *kernel_name, cudaEvent_t start, cudaEvent_t stop)
{
    float benchmark_ms = 0;
    cudaEventSynchronize(stop);                       // wait for the stop event, if it isn’t done
    cudaEventElapsedTime(&benchmark_ms, start, stop); // get the elapsed time
    printf("- Kernel: %s, Benchmark(ms): %f\n", kernel_name, benchmark_ms);
}

// elapsed time in milliseconds
float cpu_time(timespec *start, timespec *end)
{
    return ((1e9 * end->tv_sec + end->tv_nsec) - (1e9 * start->tv_sec + start->tv_nsec)) / 1e6;
}

// handle error macro
inline void HandleError(cudaError_t err, const char *file, const int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#endif