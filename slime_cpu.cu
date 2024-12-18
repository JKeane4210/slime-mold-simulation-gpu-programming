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

#define RA M_PI / 4 // = rotation angle
#define SA M_PI / 4 // = sensor angle
#define SO 9        // pixel(s) = sensor offset (original = 9)
#define SW 1        // pixel(s) = sensor width
#define SS 1        // pixel(s) = step size
#define depT 20     // how much chemoattractant is deposited (original = 5)
#define decayT 0.5  // decay rate of chemoattractant

const int N_FOOD_TYPES = 1;
#define DIFFUSION_KERNEL_R 1
#define diffK (DIFFUSION_KERNEL_R * 2 + 1)
#define MEAN_FILTER_CENTER_WEIGHT 0.9

int ENV_HEIGHT = 300;
int ENV_WIDTH = 400;
int PARTICLE_PERCENT = 33;
int N_PARTICLES;

float *env;     // array for environment (holds chemical trails)
bool *occupied; // array for if cells are occupied by particles

// individual slime particle
struct SlimeParticle
{
    float x;
    float y;
    float orientation;
};

#define MASK_WIDTH 3
__constant__ float K[MASK_WIDTH][MASK_WIDTH];

// handle error macro
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

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

float sample_chemoattractant(SlimeParticle *p, float rotation_offset, float sensor_offset)
{
    float angle = p->orientation + rotation_offset;
    if (angle < 0)
        angle += 2 * M_PI;
    if (angle > 2 * M_PI)
        angle -= 2 * M_PI;
    int s_x = (int)round(p->x + sensor_offset * cos(angle));
    int s_y = (int)round(p->y + sensor_offset * sin(angle));
    if (s_y >= 0 && s_y < ENV_HEIGHT && s_x >= 0 && s_x < ENV_WIDTH)
        return env[s_y * ENV_WIDTH + s_x];
    else
        p->orientation += ((float)rand() / (float)RAND_MAX) * M_PI_2 + M_PI_4; // keep it in bounds
    if (p->orientation > 2 * M_PI)
        p->orientation -= 2 * M_PI;
    return 0;
}

void sensory_stage_cpu(SlimeParticle *p, float sensor_angle, float rotation_angle)
{
    float F = sample_chemoattractant(p, 0, SO);
    float FR = sample_chemoattractant(p, -sensor_angle, SO);
    float FL = sample_chemoattractant(p, sensor_angle, SO);
    if ((F > FL) && (F > FR))
        return;
    else if ((F < FL) && (F < FR))
    {
        int random_sign = rand() % 2;
        p->orientation += (random_sign ? 1 : -1) * rotation_angle;
    }
    else if (FL < FR)
    {
        p->orientation -= rotation_angle;
    }
    else if (FR < FL)
    {
        p->orientation += rotation_angle;
    }
    if (p->orientation < 0)
        p->orientation += 2 * M_PI;
    if (p->orientation > 2 * M_PI)
        p->orientation -= 2 * M_PI;
}

void motor_stage_cpu(SlimeParticle *p)
{
    float n_x = p->x + cos(p->orientation) * SS;
    float n_y = p->y + sin(p->orientation) * SS;
    int p_x_i = (int)round(p->x);
    int p_y_i = (int)round(p->y);
    int n_x_i = (int)round(n_x);
    int n_y_i = (int)round(n_y);
    int w = ENV_WIDTH;
    int h = ENV_HEIGHT;
    if (n_x_i >= 0 && n_x_i < ENV_WIDTH && n_y_i >= 0 && n_y_i < ENV_HEIGHT)
    {
        const float scale = 0.005;
        if (n_x_i == p_x_i && n_y_i == p_y_i)
        {
            // if in same discrete location, don't worry about atomic changes
            p->x = n_x;
            p->y = n_y;
            if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
            {
                env[p_y_i * ENV_WIDTH + p_x_i] += depT * 1 * 1.0; // deposit trail in new location
            }
        }
        else if (occupied[n_y_i * ENV_WIDTH + n_x_i] == 0) // not occupied
        {
            occupied[n_y_i * ENV_WIDTH + n_x_i] = 1;
            occupied[p_y_i * ENV_WIDTH + p_x_i] = 0;
            p->x = n_x;
            p->y = n_y;
            if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
            {
                env[p_y_i * ENV_WIDTH + p_x_i] += depT * 1 * 1.0; // deposit trail in new location
            }
        }
        else
        {
            p->orientation += ((float)rand() / (float)RAND_MAX) * M_PI_2 - M_PI_4; // choose random new orientation
        }
    }
}

void diffusion()
{
    for (int dest_r = 0; dest_r < ENV_HEIGHT; ++dest_r)
    {
        for (int dest_c = 0; dest_c < ENV_WIDTH; ++dest_c)
        {
            for (int channel = 0; channel < N_FOOD_TYPES; ++channel)
            {
                float result = 0.0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        int kernel_r = dest_r - 1 + i;
                        int kernel_c = dest_c - 1 + j;
                        float weight = (i == DIFFUSION_KERNEL_R && j == DIFFUSION_KERNEL_R) ? MEAN_FILTER_CENTER_WEIGHT : (((1.0 - MEAN_FILTER_CENTER_WEIGHT) / 1.8) / (3 * 3 - 1));
                        if (kernel_r >= 0 && kernel_r < ENV_HEIGHT && kernel_c >= 0 && kernel_c < ENV_WIDTH)
                            result += weight * env[kernel_r * ENV_WIDTH + kernel_c]; // mean kernel
                    }
                }
                env[dest_r * ENV_WIDTH + dest_c] = result;
            }
        }
    }
}

void decay_chemoattractant()
{
    for (int i = 0; i < ENV_HEIGHT; ++i)
    {
        for (int j = 0; j < ENV_WIDTH; ++j)
        {
            env[i * ENV_WIDTH + j] = max(env[i * ENV_WIDTH + j] - decayT, 0.0);
        }
    }
}

void gray_scale_image_to_file(const char *cpu_output_file)
{
    // printf("Filename: %s\n", cpu_output_file);

    FILE *output_file_handle = fopen(cpu_output_file, "w");
    if (output_file_handle == NULL)
    {
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
            fputc((int)(min(255.0, env[i * ENV_WIDTH + j] * 4)), output_file_handle);
        }
    }
    fflush(output_file_handle);
    fclose(output_file_handle);
}

int main(int argc, char *argv[])
{
    // parsing command line arguments
    if (argc > 1)
        ENV_HEIGHT = atoi(argv[1]);
    if (argc > 2)
        ENV_WIDTH = atoi(argv[2]);
    if (argc > 3)
        PARTICLE_PERCENT = atoi(argv[3]);
    env = (float *)malloc(ENV_WIDTH * ENV_HEIGHT * sizeof(float));
    occupied = (bool *)malloc(ENV_WIDTH * ENV_WIDTH * sizeof(bool));
    N_PARTICLES = (int)((float)PARTICLE_PERCENT / 100.0 * (ENV_WIDTH * ENV_HEIGHT));
    SlimeParticle particles[N_PARTICLES];
    printf("# Particles: %d\n", N_PARTICLES);
    srand(time(NULL));

    // init environment
    for (int i = 0; i < ENV_HEIGHT; ++i)
    {
        for (int j = 0; j < ENV_WIDTH; ++j)
        {
            env[i * ENV_WIDTH + j] = 0.0;
            occupied[i * ENV_WIDTH + j] = false;
        }
    }

    // random init particles
    for (int i = 0; i < N_PARTICLES; ++i)
    {
        float random_orientation = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;

        particles[i] = {(float)(rand() % ENV_WIDTH), (float)(rand() % ENV_HEIGHT), random_orientation};
        // printf("Particle: %.3f %.3f @ %.3f\n", particles[i * 10 + j].x, particles[i * 10 + j].y, particles[i * 10 + j].orientation);
        occupied[(int)particles[i].y * ENV_WIDTH + (int)particles[i].x] = true;
        env[(int)particles[i].y * ENV_WIDTH + (int)particles[i].x] = depT; // initial deposit
    }

    // circle init
    // const float RADIUS_MAX = 45;
    // for (int i = 0; i < N_PARTICLES; ++i)
    // {
    //     float r_i = ((float)rand() / (float)RAND_MAX) * RADIUS_MAX;
    //     float theta_i = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
    //     int y_i = ENV_HEIGHT / 2 + (int)(r_i * sin(theta_i));
    //     int x_i = ENV_WIDTH / 2 + (int)(r_i * cos(theta_i));
    //     while (occupied[y_i][x_i])
    //     {
    //         r_i = ((float)rand() / (float)RAND_MAX) * RADIUS_MAX;
    //         theta_i = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
    //         y_i = ENV_HEIGHT / 2 + (int)(r_i * sin(theta_i));
    //         x_i = ENV_WIDTH / 2 + (int)(r_i * cos(theta_i));
    //     }
    //     float random_orientation = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
    //     particles[i] = {(float)x_i, (float)y_i, random_orientation};
    //     // printf("Particle: %.3f %.3f @ %.3f\n", particles[i].x, particles[i].y, particles[i].orientation);
    //     occupied[y_i][x_i] = true;
    //     env[y_i][x_i] = depT; // initial deposit
    // }

    char buffer[300];
    // simulation loop
    for (int step = 0; step < 1000; ++step)
    {
        for (int i = 0; i < N_PARTICLES; ++i)
            sensory_stage_cpu(&particles[i], SA, RA);
        for (int i = 0; i < N_PARTICLES; ++i)
            motor_stage_cpu(&particles[i]);
        diffusion();
        decay_chemoattractant();

        char *pwd = getenv("PWD");
        sprintf(buffer, "%s/frames/frame_%d.ppm", pwd, step);
        gray_scale_image_to_file(buffer);
    }

    // benchmark trials
    int n_trials = 10;
    timespec ts, te;
    for (int trial = 0; trial < n_trials; ++trial)
    {
        // sensory_stage_cpu
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        for (int i = 0; i < N_PARTICLES; ++i)
            sensory_stage_cpu(&particles[i], SA, RA);
        clock_gettime(CLOCK_MONOTONIC_RAW, &te);
        printf("- Kernel: %s, Benchmark(ms): %f\n", "sensory_stage_cpu", cpu_time(&ts, &te));

        // motor_stage_cpu
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        for (int i = 0; i < N_PARTICLES; ++i)
            motor_stage_cpu(&particles[i]);
        clock_gettime(CLOCK_MONOTONIC_RAW, &te);
        printf("- Kernel: %s, Benchmark(ms): %f\n", "motor_stage_cpu", cpu_time(&ts, &te));

        // motor_stage_cpu
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        diffusion();
        clock_gettime(CLOCK_MONOTONIC_RAW, &te);
        printf("- Kernel: %s, Benchmark(ms): %f\n", "diffusion_cpu", cpu_time(&ts, &te));

        // decay_chemoattractant
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        decay_chemoattractant();
        clock_gettime(CLOCK_MONOTONIC_RAW, &te);
        printf("- Kernel: %s, Benchmark(ms): %f\n", "decay_chemoattractant_cpu", cpu_time(&ts, &te));
    }

    return 0;
}
