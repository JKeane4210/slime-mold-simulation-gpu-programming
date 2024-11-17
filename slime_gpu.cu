#include <stdio.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 32
#define BLOCK_SIZE_PARTICLE 512
#define DEBUG_LEVEL 0
#define HARD_FAIL false

#define RA M_PI / 4 // = rotation angle
#define SA M_PI / 4 // = sensor angle
#define SO 9 // pixel(s) = sensor offset (original = 9)
#define SW 1 // pixel(s) = sensor width
#define SS 1 // pixel(s) = step size
#define depT 20 // how much chemoattractant is deposited (original = 5)
#define decayT 0.5 // decay rate of chemoattractant
#define ENV_WIDTH 400
#define ENV_HEIGHT 300

// float env[ENV_HEIGHT][ENV_WIDTH];
// bool occupied[ENV_HEIGHT][ENV_WIDTH];

struct SlimeParticle {
    float x;
    float y;
    float orientation;
    curandState_t rng;
};

#define MASK_WIDTH 3    
__constant__ float K[MASK_WIDTH][MASK_WIDTH];

//handle error macro
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line );
    }
}
    
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

// LEARNING POINT - If I am trying to initialize randomly for each point and synchronize,
// it seems like this will be a very difficult task. I can think of it however as every pixel
// having a probability of being taken and then I might get closer to what I am looking for

__global__ 
void init_particle_kernel(SlimeParticle * particles, int n, int * occupied, int w, int h) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        curandState_t state;
        curand_init(i, 0, 0, &state); // Initialize the generator
        int rx = ((int)ceilf(curand_uniform(&state) * w)) - 1;
        int ry = ((int)ceilf(curand_uniform(&state) * h)) - 1;
        float r_theta = curand_uniform(&state) * 2 * M_PI;
        while (atomicAdd(&(occupied[ry * w + rx]), 1) != 0) {
            rx = ((int)ceilf(curand_uniform(&state) * w)) - 1;
            ry = ((int)ceilf(curand_uniform(&state) * h)) - 1;
        }
        particles[i].x = (float)rx;
        particles[i].y = (float)ry;
        particles[i].orientation = r_theta;
        particles[i].rng = state;
    }
}

void report_kernel_benchmark(const char * kernel_name, cudaEvent_t start, cudaEvent_t stop) {
    float benchmark_ms = 0;
    cudaEventSynchronize(stop); //wait for the stop event, if it isn’t done
    cudaEventElapsedTime(&benchmark_ms, start, stop); //get the elapsed time
    printf("- Kernel: %s, Benchmark(ms): %f\n", kernel_name, benchmark_ms);
}

// elapsed time in milliseconds
float cpu_time(timespec* start, timespec* end){
    return ((1e9*end->tv_sec + end->tv_nsec) - (1e9*start->tv_sec + start->tv_nsec))/1e6;
}

__device__
float sample_chemoattractant(SlimeParticle* p, float * env, int w, int h, float rotation_offset, float sensor_offset) {
    float angle = p->orientation + rotation_offset;
    if (angle < 0) angle += 2 * M_PI;
    if (angle > 2 * M_PI) angle -= 2 * M_PI;
    int s_x = (int)round(p->x + sensor_offset * cos(angle));
    int s_y = (int)round(p->y + sensor_offset + sin(angle));
    if (s_y >= 0 && s_y < ENV_HEIGHT && s_x >= 0 && s_x < ENV_WIDTH)
        return env[s_y * w + s_x];
    else
        p->orientation += curand_uniform(&(p->rng)) * M_PI_2 + M_PI_4; // keep it in bounds
        if (p->orientation > 2 * M_PI) p->orientation -= 2 * M_PI;
        return 0;
}

__global__
void sensor_stage_kernel(SlimeParticle * particles, int n, float * env, int w, int h, float sensor_angle, float rotation_angle) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        SlimeParticle * p = &particles[i];
        float F = sample_chemoattractant(p, env, w, h, 0, SO);
        float FR = sample_chemoattractant(p, env, w, h, -sensor_angle, SO);
        float FL = sample_chemoattractant(p, env, w, h, sensor_angle, SO);
        if ((F > FL) && (F > FR)) return;
        else if ((F < FL) && (F < FR)) {
            int random_sign = (int)(ceilf(curand_uniform(&(p->rng)) * 2)) - 1;
            p->orientation += (random_sign ? 1 : -1) * rotation_angle;
        } else if (FL < FR) {
            p->orientation -= rotation_angle;
        } else if (FR < FL) {
            p->orientation += rotation_angle;
        }
        if (p->orientation < 0) p->orientation += 2 * M_PI;
        if (p->orientation > 2 * M_PI) p->orientation -= 2 * M_PI;
    }
}

__global__
void motor_stage_kernel(SlimeParticle * particles, int n, float * env, int * occupied, int w, int h, float sensor_angle, float rotation_angle) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        SlimeParticle * p = &particles[i];
        float n_x = p->x + cos(p->orientation) * SS;
        float n_y = p->y + sin(p->orientation) * SS;
        int p_x_i = (int)round(p->x);
        int p_y_i = (int)round(p->y);
        int n_x_i = (int)round(n_x);
        int n_y_i = (int)round(n_y);
        if (atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1) == 0) { // not occupied
            atomicExch(&(occupied[p_y_i * w + p_x_i]), 0); // clear previous location
            p->x = n_x;
            p->y = n_y;
            if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
                atomicAdd(&(env[p_y_i * w + p_x_i]), depT); // deposit trail in new location
        } else {
            p->orientation += curand_uniform(&(p->rng)) * M_PI_2 - M_PI_4; // choose random new orientation
            if (p->orientation < 0) p->orientation += 2 * M_PI;
            if (p->orientation > 2 * M_PI) p->orientation -= 2 * M_PI;
        }
    }
}

__global__
void decay_chemoattractant_kernel(float * env, int w, int h) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < w && col >= 0 && row < h && row >= 0) {
        env[row * w + col] = max(env[row * w + col] - decayT, 0.0);
    }
}

void gray_scale_image_to_file(const char *cpu_output_file, float * env) {
    // printf("Filename: %s\n", cpu_output_file);
    
    FILE *output_file_handle = fopen(cpu_output_file, "w");
    if (output_file_handle == NULL) {
        // Print a descriptive error message
        perror("Error opening file");
        
        // Alternatively, use strerror to get the error message
        printf("fopen failed: %s\n", strerror(errno));
        return;
    }
    fprintf(output_file_handle, "%s\n#\n%d %d\n%d\n", "P5", ENV_WIDTH, ENV_HEIGHT, 255);
    for (int i = 0; i < ENV_HEIGHT; ++i)
    {
        for (int j = 0; j < ENV_WIDTH; ++j)
        {
            fputc((int)(min(255.0, env[i * ENV_WIDTH + j])), output_file_handle); // TODO
        }
    }
    fflush(output_file_handle);
    fclose(output_file_handle);
}

// __global__
// void test_kernel() {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int test = 0;
//     printf("r1: %d\n", atomicAdd(&test, 1));
//     printf("r2: %d\n", atomicAdd(&test, 1));
//     printf("%d\n", test);
// }

int main(int argc, char* argv[]) {
    int * occupied_d;
    HANDLE_ERROR(cudaMalloc((void **)&occupied_d, ENV_WIDTH * ENV_HEIGHT * sizeof(int)));
    HANDLE_ERROR(cudaMemset(occupied_d, 0, ENV_WIDTH * ENV_HEIGHT * sizeof(int)));

    float * env_d;
    float * env_h = (float *)malloc(ENV_WIDTH * ENV_HEIGHT * sizeof(float));
    HANDLE_ERROR(cudaMalloc((void **)&env_d, ENV_WIDTH * ENV_HEIGHT * sizeof(float)));
    HANDLE_ERROR(cudaMemset(env_d, 0, ENV_WIDTH * ENV_HEIGHT * sizeof(float)));

    const int N_PARTICLES = 1600;
    SlimeParticle* particles_d;
    HANDLE_ERROR(cudaMalloc((void **)&particles_d, N_PARTICLES * sizeof(SlimeParticle)));
    printf("Dims: %d %d\n", (N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE);
    init_particle_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, occupied_d, ENV_WIDTH, ENV_HEIGHT);
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    // debug because dgb is being weird
    // SlimeParticle* particles_h = (SlimeParticle *)malloc(N_PARTICLES * sizeof(SlimeParticle));
    // HANDLE_ERROR(cudaMemcpy(particles_h, particles_d, N_PARTICLES * sizeof(SlimeParticle), cudaMemcpyDeviceToHost));
    // printf("Size: %lu\n", N_PARTICLES * sizeof(SlimeParticle));
    // for (int i = 0; i < N_PARTICLES; ++i) {
    //     printf("Particle: %.2f, %.2f @ %.3f\n", particles_h[i].x, particles_h[i].y, particles_h[i].orientation);
    // }

    char buffer[300];
    const int N_STEPS = 1000;
    for (int i = 0; i < N_STEPS; ++i) {
        sensor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, ENV_WIDTH, ENV_HEIGHT, SA, RA);
        motor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, occupied_d, ENV_WIDTH, ENV_HEIGHT, SA, RA);
        dim3 dg((ENV_WIDTH - 1) / BLOCK_SIZE + 1, (ENV_WIDTH - 1) / BLOCK_SIZE + 1, 1);
        dim3 db(32, 32, 1);
        decay_chemoattractant_kernel<<<dg, db>>>(env_d, ENV_WIDTH, ENV_HEIGHT);
        cudaMemcpy(env_h, env_d, ENV_WIDTH * ENV_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
        sprintf(buffer, "./Desktop/Projects/slime-mold-simulation-gpu-programming/frames/frame_%d.ppm", i);
        gray_scale_image_to_file(buffer, env_h);
    }

    HANDLE_ERROR(cudaFree(particles_d));
    HANDLE_ERROR(cudaFree(env_d));
    HANDLE_ERROR(cudaFree(occupied_d));

    printf("Done!\n");
    
    // init environment
    // for (int i = 0; i < ENV_HEIGHT; ++i) {
    //     for (int j = 0; j < ENV_WIDTH; ++j) {
    //         env[i][j] = 0.0;
    //         occupied[i][j] = false;
    //     }
    // }
    
    // random init particles in center
    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         float random_orientation = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;

    //         particles[i * 10 + j] = { (float)(i + ENV_WIDTH / 2 - 5),  (float)(j + ENV_HEIGHT / 2 - 5), random_orientation};
    //         printf("Particle: %.3f %.3f @ %.3f\n", particles[i * 10 + j].x, particles[i * 10 + j].y, particles[i * 10 + j].orientation);
    //         occupied[i + ENV_HEIGHT / 2 - 5][j + ENV_WIDTH / 2 - 5] = true;
    //         env[i + ENV_HEIGHT / 2 - 5][j + ENV_WIDTH / 2 - 5] = depT; // initial deposit
    //     }
    // }

    // circle init
    // const float RADIUS_MAX = 45;
    // for (int i = 0; i < N_PARTICLES; ++i) {
    //     float r_i = ((float)rand() / (float)RAND_MAX) * RADIUS_MAX;
    //     float theta_i = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
    //     int y_i = ENV_HEIGHT / 2 + (int)(r_i * sin(theta_i));
    //     int x_i = ENV_WIDTH / 2 + (int)(r_i * cos(theta_i));
    //     while (occupied[y_i][x_i]) {
    //         r_i = ((float)rand() / (float)RAND_MAX) * RADIUS_MAX;
    //         theta_i = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
    //         y_i = ENV_HEIGHT / 2 + (int)(r_i * sin(theta_i));
    //         x_i = ENV_WIDTH / 2 + (int)(r_i * cos(theta_i));
    //     }
    //     float random_orientation = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
    //     particles[i] = { (float)x_i, (float)y_i, random_orientation};
    //     printf("Particle: %.3f %.3f @ %.3f\n", particles[i].x, particles[i].y, particles[i].orientation);
    //     occupied[y_i][x_i] = true;
    //     env[y_i][x_i] = depT; // initial deposit
    // }

    // char buffer[300];
    // // simulation loop
    // for (int step = 0; step < 1000; ++step) {
    //     for (int i = 0; i < N_PARTICLES; ++i) 
    //         sensory_stage_cpu(&particles[i], SA, RA);
    //     for (int i = 0; i < N_PARTICLES; ++i)
    //         motor_stage_cpu(&particles[i]);
    //     decay_chemoattractant();
        
    //     sprintf(buffer, "./source/final-project/frames/frame_%d.ppm", step);
    //     gray_scale_image_to_file(buffer);
    // }

    return 0;
}
