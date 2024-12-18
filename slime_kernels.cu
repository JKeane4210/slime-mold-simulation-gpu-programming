#include "slime_kernels.h"
#include <stdio.h>

// #define OVERLAPPING_PARTICLES     // allow for particles to overlap
// #define DISPLAY_PARTICLE_LOCATION // display the actual location of the slime particles
#define DISPLAY_SLIME_TRAIL       // display chemical trail left behind by slime particles
// #define DISPLAY_FOOD_REGION       // display where the food is laid out (deprecated)

const uint MSOE_RED = 0x0C05C5;
const uint MSOE_WHITE = 0xFFFFFF;

// initializes particles in a circular shape in the center of the environment
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
        particles[i].rng = state;
        particles[i].food[0] = 0;
        particles[i].food[1] = 0;
    }
}

// initializes particles randomly around the environment
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
        particles[i].food[0] = 0;
        particles[i].food[1] = 0;
    }
}

// adds food of different types around the environment
__global__ void add_food_kernel(float *food, float *food_pattern, int w, int h)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < w && row < h)
    {
        float value = food_pattern[(row * w + col) * N_FOOD_TYPES];
        food[(row * w + col) * N_FOOD_TYPES] += value;                                                // white food
        food[(row * w + col) * N_FOOD_TYPES + 1] += food_pattern[(row * w + col) * N_FOOD_TYPES + 1]; // red food
        // if (value == 0)
        //     food[(row * w + col) * N_FOOD_TYPES + 1] += 0;
    }
}

// gets the chemical trail in front of a particle and at a given angle offset from its current trajectory (represents retrieving from a sensor field)
__device__ float sample_chemoattractant(SlimeParticle *p, float *env, float *food, int w, int h, float rotation_offset, float sensor_offset)
{
    float angle = p->orientation + rotation_offset;
    int s_x = (int)round(p->x + sensor_offset * cos(angle));
    int s_y = (int)round(p->y + sensor_offset * sin(angle));
    if (s_y >= 0 && s_y < h && s_x >= 0 && s_x < w)
        return env[(s_y * w + s_x) * N_FOOD_TYPES] + env[(s_y * w + s_x) * N_FOOD_TYPES + 1]; // + 10 * food[s_y * w + s_x];
    else
        p->orientation += curand_uniform(&(p->rng)) * M_PI; // keep it in bounds
    return 0;
}

// sensory stage for slime particle
__global__ void sensor_stage_kernel(SlimeParticle *particles, int n, float *env, float *food, int w, int h, float sensor_angle, float rotation_angle, float sensor_offset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        SlimeParticle *p = &particles[i];
        float F = sample_chemoattractant(p, env, food, w, h, 0, sensor_offset);
        float FR = sample_chemoattractant(p, env, food, w, h, -sensor_angle, sensor_offset);
        float FL = sample_chemoattractant(p, env, food, w, h, sensor_angle, sensor_offset);
        if ((F > FL) && (F > FR)) // if forward has most chemicaltrail, return
            return;
        else if ((F < FL) && (F < FR)) // if either side has more chemical trails, turn a fixed amount (choosing left/right randomly)
        {
            int random_sign = (int)(ceilf(curand_uniform(&(p->rng)) * 2)) - 1;
            p->orientation += (random_sign ? 1 : -1) * rotation_angle;
        }
        else if (FL < FR) // go right if the most chemical trail is to the right
        {
            p->orientation -= rotation_angle;
        }
        else if (FR < FL) // go left if the most chemical trail is to the left
        {
            p->orientation += rotation_angle;
        }
    }
}

// motor stage for slime particle
__global__ void motor_stage_kernel(SlimeParticle *particles, int n, float *env, float *food, int *occupied, int w, int h, float sensor_angle, float rotation_angle, float step_size, float deposit_amount, float delta_t, float base_food)
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
            // update food amounts based on location
            p->food[0] += food[(n_y_i * w + n_x_i) * N_FOOD_TYPES] - 0.5; // 0.1 is decay rate
            p->food[0] = min(max(p->food[0], 0.0), PARTICLE_STOMACH_SIZE);
            p->food[1] += food[(n_y_i * w + n_x_i) * N_FOOD_TYPES + 1] - 0.5;
            p->food[1] = min(max(p->food[1], 0.0), PARTICLE_STOMACH_SIZE);
            const float scale = 0.005;
#ifndef OVERLAPPING_PARTICLES
            // if in same discrete location, deposit in current location
            if (n_x_i == p_x_i && n_y_i == p_y_i)
            {
                p->x = n_x;
                p->y = n_y;
                if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
                {
                    atomicAdd(&(env[(p_y_i * w + p_x_i) * N_FOOD_TYPES]), deposit_amount * deltaT * (base_food + scale * p->food[0])); // deposit trail in new location
                    atomicAdd(&(env[(p_y_i * w + p_x_i) * N_FOOD_TYPES + 1]), deposit_amount * deltaT * (0.0 + scale * p->food[1]));   // deposit trail in new location
                }
            }
            else if (atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1) == 0) // else if target location not occupied, move there and deposit chemcail trail
            {
                // atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1);
                atomicExch(&(occupied[p_y_i * w + p_x_i]), 0); // clear previous location
                p->x = n_x;
                p->y = n_y;
                if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
                {
                    atomicAdd(&(env[(p_y_i * w + p_x_i) * N_FOOD_TYPES]), deposit_amount * deltaT * (base_food + scale * p->food[0])); // deposit trail in new location
                    atomicAdd(&(env[(p_y_i * w + p_x_i) * N_FOOD_TYPES + 1]), deposit_amount * deltaT * (0.0 + scale * p->food[1]));   // deposit trail in new location
                }
            }
            else
            {
                p->orientation += curand_uniform(&(p->rng)) * M_PI_2 - M_PI_4; // otherwise, choose random new orientation
            }
#else
            atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1);
            atomicExch(&(occupied[p_y_i * w + p_x_i]), 0); // clear previous location
            p->x = n_x;
            p->y = n_y;
            if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
            {
                atomicAdd(&(env[(p_y_i * w + p_x_i) * N_FOOD_TYPES]), deposit_amount * deltaT * (base_food + scale * p->food[0])); // deposit trail in new location
                atomicAdd(&(env[(p_y_i * w + p_x_i) * N_FOOD_TYPES + 1]), deposit_amount * deltaT * (0.0 + scale * p->food[1]));   // deposit trail in new location
            }
#endif
        }
    }
}

// gets a percentage of a color between black (0x00) and the given color
__device__ uint ratioed_color(float value, uint color)
{
    uint r = 0;
    float brightness = value / 255.0; // normalize to be 0-1 (I can fix later)
    for (int shift = 0; shift < 24; shift += 8)
    {
        r |= ((uint)(brightness * (float)((color >> shift) & 0xFF))) << shift;
    }
    return r;
}

// decays chemical trail in the environment (also adding the correct binary representations of colors to a result array)
__global__ void decay_chemoattractant_kernel(float *env, float *food, int *occupied, uint *result, int w, int h, float decay_amount)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < w && col >= 0 && row < h && row >= 0)
    {
        // white food
        float value_white = max(env[(row * w + col) * N_FOOD_TYPES] - decay_amount * deltaT, 0.0);
        env[(row * w + col) * N_FOOD_TYPES] = value_white;
        value_white *= VISUAL_SCALING;
        value_white = min(value_white, 255.0);

        // red food
        float value_red = max(env[(row * w + col) * N_FOOD_TYPES + 1] - decay_amount * deltaT, 0.0);
        env[(row * w + col) * N_FOOD_TYPES + 1] = value_red;
        value_red *= VISUAL_SCALING;
        value_red = min(value_red, 255.0);

#ifndef DISPLAY_SLIME_TRAIL
        value = 0;
#endif
#ifdef DISPLAY_PARTICLE_LOCATION
        value = max(value, (occupied[row * w + col] > 0) ? 255.0 : 0);
#endif
        uint r = value_red > value_white ? ratioed_color(value_red, MSOE_RED) : ratioed_color(value_white, MSOE_WHITE);

#ifdef DISPLAY_FOOD_REGION
        if (food[row * w + col] > 0.0)
            r &= 0xFF0000FF;
#endif
        // adds resulting color as a uint to the result array (what will be rendered)
        result[row * w + col] = r;
    }
}

// diffuses chemical trail in the environment (tiled convolution using a blur filter)
__global__ void diffusion_kernel(float *env, float *env_dest, int w, int h)
{
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x - DIFFUSION_KERNEL_R;
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y - DIFFUSION_KERNEL_R;
    __shared__ float tile_s[BLOCK_SIZE + diffK - 1][BLOCK_SIZE + diffK - 1][N_FOOD_TYPES];

    for (int channel = 0; channel < N_FOOD_TYPES; ++channel)
    {
        if (col >= 0 && col < w && row >= 0 && row < h)
            tile_s[threadIdx.y][threadIdx.x][channel] = env[(row * w + col) * N_FOOD_TYPES + channel];
        else
            tile_s[threadIdx.y][threadIdx.x][channel] = 0.0;
    }

    __syncthreads();

    // changes role of thread from data loading to loading the kernel that starts in top left corner at [threadIdx.y][threadIdx.x]
    if (threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE)
    {
        int dest_c = BLOCK_SIZE * blockIdx.x + threadIdx.x;
        int dest_r = BLOCK_SIZE * blockIdx.y + threadIdx.y;
        if (dest_c >= 0 && dest_c < w && dest_r >= 0 && dest_r < h)
        {
            for (int channel = 0; channel < N_FOOD_TYPES; ++channel)
            {
                float result = 0.0;
                for (int i = 0; i < diffK; ++i)
                {
                    for (int j = 0; j < diffK; ++j)
                    {
                        float weight = (i == DIFFUSION_KERNEL_R && j == DIFFUSION_KERNEL_R) ? MEAN_FILTER_CENTER_WEIGHT : (((1.0 - MEAN_FILTER_CENTER_WEIGHT) / 1.8) / (3 * 3 - 1));
                        result += weight * tile_s[threadIdx.y + i][threadIdx.x + j][channel]; // mean kernel
                    }
                }
                env_dest[(dest_r * w + dest_c) * N_FOOD_TYPES + channel] = result;
            }
        }
    }
}