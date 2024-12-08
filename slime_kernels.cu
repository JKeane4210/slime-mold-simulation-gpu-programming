#include "slime_kernels.h"
// #include "constant_definitions.h"
#include <stdio.h>

// #define OVERLAPPING_PARTICLES
// #define DISPLAY_PARTICLE_LOCATION
#define DISPLAY_SLIME_TRAIL
// #define DISPLAY_FOOD_REGION

// extern __constant__ float K[diffK][diffK];

__global__ void init_circle_particle_kernel(SlimeParticle *particles, int n, int *occupied, int w, int h)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        curandState_t state;
        curand_init(i, 0, 0, &state); // Initialize the generator
        int r_radius = ((int)ceilf(curand_uniform(&state) * (float)min(w * 3 / 4, h * 3 / 4) / 2)) - 1;
        float r_theta = curand_uniform(&state) * 2 * M_PI;
        int rx = w / 2 + (int)(cos(r_theta) * r_radius);
        int ry = h / 2 + (int)(sin(r_theta) * r_radius);
#ifndef OVERLAPPING_PARTICLES
        while (atomicAdd(&(occupied[ry * w + rx]), 1) != 0)
        {
            r_radius = ((int)ceilf(curand_uniform(&state) * (float)min(w - 2, h - 2) / 2)) - 1;
            r_theta = curand_uniform(&state) * 2 * M_PI;
            rx = w / 2 + (int)(cos(r_theta) * r_radius);
            ry = h / 2 + (int)(sin(r_theta) * r_radius);
        }
#endif
        particles[i].x = (float)rx;
        particles[i].y = (float)ry;
        particles[i].orientation = r_theta + 0 * M_PI;
        // if (particles[i].orientation > 2 * M_PI)
        //     particles[i].orientation -= 2 * M_PI;
        particles[i].rng = state;
        particles[i].food = 0;
    }
}

__global__ void init_particle_kernel(SlimeParticle *particles, int n, int *occupied, int w, int h)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        curandState_t state;
        curand_init(i, 0, 0, &state); // Initialize the generator
        int rx = ((int)ceilf(curand_uniform(&state) * w)) - 1;
        int ry = ((int)ceilf(curand_uniform(&state) * h)) - 1;
        float r_theta = curand_uniform(&state) * 2 * M_PI;
#ifndef OVERLAPPING_PARTICLES
        while (atomicAdd(&(occupied[ry * w + rx]), 1) != 0)
        {
            rx = ((int)ceilf(curand_uniform(&state) * w)) - 1;
            ry = ((int)ceilf(curand_uniform(&state) * h)) - 1;
        }
#endif
        particles[i].x = (float)rx;
        particles[i].y = (float)ry;
        particles[i].orientation = r_theta;
        particles[i].rng = state;
        particles[i].food = 0;
    }
}

__global__ void add_food_kernel(float *food, float *food_pattern, int w, int h)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < w && row < h)
    {
        food[row * w + col] += food_pattern[row * w + col] + 1;
    }
}

__device__ float sample_chemoattractant(SlimeParticle *p, float *env, float *food, int w, int h, float rotation_offset, float sensor_offset)
{
    float angle = p->orientation + rotation_offset;
    int s_x = (int)round(p->x + sensor_offset * cos(angle));
    int s_y = (int)round(p->y + sensor_offset * sin(angle));
    if (s_y >= 0 && s_y < h && s_x >= 0 && s_x < w)
        return env[s_y * w + s_x]; // + 10 * food[s_y * w + s_x];
    else
        p->orientation += curand_uniform(&(p->rng)) * M_PI; // keep it in bounds
    return 0;
}

__global__ void sensor_stage_kernel(SlimeParticle *particles, int n, float *env, float *food, int w, int h, float sensor_angle, float rotation_angle, float sensor_offset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        SlimeParticle *p = &particles[i];
        float F = sample_chemoattractant(p, env, food, w, h, 0, sensor_offset);
        float FR = sample_chemoattractant(p, env, food, w, h, -sensor_angle, sensor_offset);
        float FL = sample_chemoattractant(p, env, food, w, h, sensor_angle, sensor_offset);
        if ((F > FL) && (F > FR))
            return;
        else if ((F < FL) && (F < FR))
        {
            int random_sign = (int)(ceilf(curand_uniform(&(p->rng)) * 2)) - 1;
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
    }
}

__global__ void motor_stage_kernel(SlimeParticle *particles, int n, float *env, float *food, int *occupied, int w, int h, float sensor_angle, float rotation_angle, float step_size, float deposit_amount, float delta_t)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        SlimeParticle *p = &particles[i];
        float n_x = p->x + cos(p->orientation) * step_size * delta_t;
        float n_y = p->y + sin(p->orientation) * step_size * delta_t;
        int p_x_i = (int)round(p->x);
        int p_y_i = (int)round(p->y);
        int n_x_i = (int)round(n_x);
        int n_y_i = (int)round(n_y);
        if (n_x_i >= 0 && n_x_i < w && n_y_i >= 0 && n_y_i < h)
        {
            p->food += food[n_y_i * w + n_x_i] - 0.5; // 0.1 is decay rate
            p->food = max(p->food, 0.0);
            const float scale = 0.005;
#ifndef OVERLAPPING_PARTICLES
            if (n_x_i == p_x_i && n_y_i == p_y_i)
            {
                // if in same discrete location, don't worry about atomic changes
                p->x = n_x;
                p->y = n_y;
                if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
                    atomicAdd(&(env[p_y_i * w + p_x_i]), deposit_amount * deltaT * (0.0 + scale * p->food)); // deposit trail in new location
            }
            else if (atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1) == 0) // not occupied
            {
                // atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1);
                atomicExch(&(occupied[p_y_i * w + p_x_i]), 0); // clear previous location
                p->x = n_x;
                p->y = n_y;
                if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
                    atomicAdd(&(env[p_y_i * w + p_x_i]), deposit_amount * deltaT * (0.0 + scale * p->food)); // deposit trail in new location
            }
            else
            {
                p->orientation += curand_uniform(&(p->rng)) * M_PI_2 - M_PI_4; // choose random new orientation
                if (p->orientation < 0)
                    p->orientation += 2 * M_PI;
                if (p->orientation > 2 * M_PI)
                    p->orientation -= 2 * M_PI;
            }
#else
            atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1);
            atomicExch(&(occupied[p_y_i * w + p_x_i]), 0); // clear previous location
            p->x = n_x;
            p->y = n_y;
            if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
                atomicAdd(&(env[p_y_i * w + p_x_i]), deposit_amount * deltaT * (0.0 + scale * p->food)); // deposit trail in new location
#endif
        }
    }
}

__global__ void decay_chemoattractant_kernel(float *env, float *food, int *occupied, uint *result, int w, int h, float decay_amount)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < w && col >= 0 && row < h && row >= 0)
    {
        float value = max(env[row * w + col] - decay_amount * deltaT, 0.0);
        env[row * w + col] = value;
        value *= VISUAL_SCALING;
        value = min(value, 255.0);
#ifndef DISPLAY_SLIME_TRAIL
        value = 0;
#endif
#ifdef DISPLAY_PARTICLE_LOCATION
        value = max(value, (occupied[row * w + col] > 0) ? 255.0 : 0);
#endif
        uint r = (0u << 24) |
                 ((unsigned int)(value) << 16) |
                 ((unsigned int)(value) << 8) |
                 ((unsigned int)(value));
#ifdef DISPLAY_FOOD_REGION
        if (food[row * w + col] > 0.0)
            r &= 0xFF0000FF;
#endif
        result[row * w + col] = r;
    }
}

// since the paper described just a simple mean filter, I will
// proceed without the kernel and just use weights of 1 / 9
__global__ void diffusion_kernel(float *env, float *env_dest, int w, int h)
{
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x - DIFFUSION_KERNEL_R;
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y - DIFFUSION_KERNEL_R;
    __shared__ float tile_s[BLOCK_SIZE + diffK - 1][BLOCK_SIZE + diffK - 1];

    if (col >= 0 && col < w && row >= 0 && row < h)
        tile_s[threadIdx.y][threadIdx.x] = env[row * w + col];
    else
        tile_s[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    // changes role of thread from data loading to loading the kernel that starts in top left corner at [threadIdx.y][threadIdx.x]
    if (threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE)
    {
        int dest_c = BLOCK_SIZE * blockIdx.x + threadIdx.x;
        int dest_r = BLOCK_SIZE * blockIdx.y + threadIdx.y;
        if (dest_c >= 0 && dest_c < w && dest_r >= 0 && dest_r < h)
        {
            float result = 0.0;
            for (int i = 0; i < diffK; ++i)
            {
                for (int j = 0; j < diffK; ++j)
                {
                    float weight = (i == DIFFUSION_KERNEL_R && j == DIFFUSION_KERNEL_R) ? MEAN_FILTER_CENTER_WEIGHT : (((1.0 - MEAN_FILTER_CENTER_WEIGHT) / 1.8) / (3 * 3 - 1));
                    // if (col == 0 && row == 0) {
                    //     printf("%d %d %f %d %d %f\n", i, j, weight, diffK * diffK, diffK, (1.0 - MEAN_FILTER_CENTER_WEIGHT));
                    // }
                    result += weight * tile_s[threadIdx.y + i][threadIdx.x + j]; // mean kernel
                }
            }
            env_dest[dest_r * w + dest_c] = result;
        }
    }
    // if (col == 0 && row == 0)
    //     printf("%f\n", K[0][0]);
}