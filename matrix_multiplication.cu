#include <stdio.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define BLOCK_SIZE 32
#define DEBUG_LEVEL 0
#define HARD_FAIL false

//handle error macro
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line );
    }
}
    
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

__global__
void init_A_matrix_kernel(float * A, int m, int n) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < m && col < n) {
        A[col + (row * n)] = 1.0;
    }
}

__global__
void init_B_matrix_kernel(float * B, int n, int k) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < n && col < k) {
        B[col + (row * k)] = 1.0;
    }
}

__global__
void matrix_multiplication_kernel(float * A, float * B, float * C, int m, int n, int k) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < m && col < k) {
        float result = 0.0; // dot product result
        for (int z = 0; z < n; ++z) {
            result += A[z + (row * n)] * B[col + (z * k)]; // [row][z] (on mxn matrix) and [z][col] (on nxk matrix)
        }
        C[col + (row * k)] = result; // trying to avoid global memory writes more than once at [row][col] (on mxk matrix)
    }
}

__global__
void matrix_multiplication_tiled_kernel(float * A, float * B, float * C, int m, int n, int k) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    float result = 0.0; // dot product result
    for (int phase_ind = 0; phase_ind < ((n - 1) / BLOCK_SIZE + 1); phase_ind++) {
        int phase_start = phase_ind * BLOCK_SIZE;
        
        // load the section into shared memory
        // get A[row][phase_start + threadIdx.x]
        if (row < m && (phase_start + threadIdx.x) < n) // if the row is within A's height and the phase's rep is within A's width
            shared_A[threadIdx.y][threadIdx.x] = A[(phase_start + threadIdx.x) + (row * n)];
        else shared_A[threadIdx.y][threadIdx.x] = 0.0;
        // get B[phase_start + threadIdx.y][col]
        if ((phase_start + threadIdx.y) < n && col < k) // if the phase's rep is within B's height and the the column is within B's width
            shared_B[threadIdx.y][threadIdx.x] = B[col + ((phase_start + threadIdx.y) * k)];
        else shared_B[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        // add partial inner product
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            result += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x]; // do the phase of the dot product (horizontal across A and vertical across B)
        }
        __syncthreads();
    }
    if (row < m && col < k) // if y < height of A and x < width of B
        C[col + (row * k)] = result; // trying to avoid global memory writes more than once at [y][x] (on mxk matrix)
}

void report_kernel_benchmark(const char * kernel_name, cudaEvent_t start, cudaEvent_t stop) {
    float benchmark_ms = 0;
    cudaEventSynchronize(stop); //wait for the stop event, if it isn’t done
    cudaEventElapsedTime(&benchmark_ms, start, stop); //get the elapsed time
    printf("- Kernel: %s, Benchmark (ms): %f\n", kernel_name, benchmark_ms);
}

void matrix_multiplication_gpu(float * A_h, float * B_h, float * C1_h, float * C2_h, int m, int n, int k) {
    float *A_d, *B_d, *C_d;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1); // make sure the product of these does not exceed max threads/block (1024)

    cudaEvent_t start, stop; //declare a start and stop event
    cudaEventCreate(&start); //create both events
    cudaEventCreate(&stop);

    HANDLE_ERROR(cudaMalloc((void**)&A_d, m * n * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&B_d, n * k * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&C_d, m * k * sizeof(float)));

    // init A
    dim3 dimGrid_initA((n - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1, 1);
    cudaEventRecord(start); //insert the start event into the stream
    init_A_matrix_kernel<<<dimGrid_initA, dimBlock>>>(A_d, m, n);
    cudaEventRecord(stop); //insert the stop event into the stream
    report_kernel_benchmark("init_A_matrix", start, stop);

    // init B
    dim3 dimGrid_initB((k - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
    cudaEventRecord(start); //insert the start event into the stream
    init_B_matrix_kernel<<<dimGrid_initB, dimBlock>>>(B_d, n, k);
    cudaEventRecord(stop); //insert the stop event into the stream
    report_kernel_benchmark("init_B_matrix", start, stop);

    // perform matrix multiplication
    dim3 dimGrid_matrix_multiplication((k - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1, 1);
    cudaEventRecord(start); //insert the start event into the stream
    matrix_multiplication_kernel<<<dimGrid_matrix_multiplication, dimBlock>>>(A_d, B_d, C_d, m, n, k);
    cudaEventRecord(stop); //insert the stop event into the stream
    report_kernel_benchmark("matrix_multiplication", start, stop);

    // save basic matrix multiplication result
    HANDLE_ERROR(cudaMemcpy(C1_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));

    // reset memory
    HANDLE_ERROR(cudaMemset(C_d, 0, m * k * sizeof(float)));
    HANDLE_ERROR(cudaDeviceSynchronize());

    // perform tiled matrix multiplication
    cudaEventRecord(start); //insert the start event into the stream
    dim3 dimGrid_matrix_multiplication_tiled((k - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1, 1);
    matrix_multiplication_tiled_kernel<<<dimGrid_matrix_multiplication_tiled, dimBlock>>>(A_d, B_d, C_d, m, n, k);
    cudaEventRecord(stop); //insert the stop event into the stream
    HANDLE_ERROR(cudaGetLastError());
    report_kernel_benchmark("matrix_multiplication_tiled", start, stop);

    // save tiled matrix multiplication result
    HANDLE_ERROR(cudaMemcpy(C2_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(A_d));
    HANDLE_ERROR(cudaFree(B_d));
    HANDLE_ERROR(cudaFree(C_d));
}

void validate_output(float * c, int m, int n, int k) {
    double sum = 0.0;
    for (int i = 0; i < m * k; ++i) {
        sum += c[i];
        if (abs(c[i] - (float)n) > 1e-3 && DEBUG_LEVEL > 0) {
            printf("missing %d, %d!\n", i / k, i % k);
        }
    }
    double result = sum / (double)(m * k);
    printf("Matrix Multiplication Sum / %d = %f\n", m * k, result);
    if (HARD_FAIL)
        assert(abs(result - (float)n) < 1e-4 && "Matrix multiplication results not correct!");
    else if (abs(result - (float)n) > 1e-4) {
        printf("WARNING - Matrix multiplication results not correct!\n");
    }
}

void benchmark_gpu(int m, int n, int k) {
    float * a = (float *)malloc(m * n * sizeof(float));
    float * b = (float *)malloc(n * k * sizeof(float));
    float * c1 = (float *)malloc(m * k * sizeof(float));
    float * c2 = (float *)malloc(m * k * sizeof(float));
    matrix_multiplication_gpu(a, b, c1, c2, m, n, k);
    validate_output(c1, m, n, k);
    validate_output(c2, m, n, k);
}

int main(int argc, char* argv[]) {
    assert(argc == 4 && "There should be three integer arguments m, n, and k!");

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    printf("m = %d, n = %d, k = %d\n", m, n, k);

    benchmark_gpu(m, n, k);

    return 0;
}