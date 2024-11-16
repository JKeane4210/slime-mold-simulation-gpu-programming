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

#define BLOCK_SIZE 32
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
bool occupied[ENV_HEIGHT][ENV_WIDTH];

struct SlimeParticle {
    float x;
    float y;
    float orientation;
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
        p->orientation += ((float)rand() / (float)RAND_MAX) * M_PI_2 + M_PI_4; // keep it in bounds
        if (p->orientation > 2 * M_PI) p->orientation -= 2 * M_PI;
        return 0;
}

__global__
void sensor_stage_kernel(SlimeParticle * particles, int n, float * env, int w, int h, float sensor_angle, float rotation_angle) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        SlimeParticle * p  = &particles[i];
        float F = sample_chemoattractant(p, env, w, h, 0, SO);
        float FR = sample_chemoattractant(p, env, w, h, -sensor_angle, SO);
        float FL = sample_chemoattractant(p, env, w, h, sensor_angle, SO);
        if ((F > FL) && (F > FR)) return;
        else if ((F < FL) && (F < FR)) {
            int random_sign = rand() % 2;
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
void motor_stage_kernel(SlimeParticle * particles, int n, float * env, int w, int h, float sensor_angle, float rotation_angle) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        SlimeParticle * p = &particles[i];
        float n_x = p->x + cos(p->orientation) * SS;
        float n_y = p->y + sin(p->orientation) * SS;
        if (!occupied[(int)round(n_y)][(int)round(n_x)]) {
            occupied[(int)round(p->y)][(int)round(p->x)] = false;
            occupied[(int)round(n_y)][(int)round(n_x)] = true;
            p->x = n_x;
            p->y = n_y;
            if ((int)round(p->y) < h && (int)round(p->y) >= 0 && (int)round(p->x) < w && (int)round(p->x) >= 0)
                env[(int)round(p->y) * w + (int)round(p->x)] += depT; // deposit trail in new location
        } else {
            p->orientation += ((float)rand() / (float)RAND_MAX) * M_PI_2 - M_PI_4; // choose random new orientation
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

void gray_scale_image_to_file(const char *cpu_output_file) {
    printf("Filename: %s\n", cpu_output_file);
    
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
            // fputc((int)(min(255.0, env[i][j])), output_file_handle); // TODO
        }
    }
    fflush(output_file_handle);
    fclose(output_file_handle);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    const int N_PARTICLES = 1600;
    SlimeParticle particles[N_PARTICLES];
    
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
