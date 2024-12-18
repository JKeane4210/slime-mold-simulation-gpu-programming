#ifndef SLIME_KERNELS_H
#define SLIME_KERNELS_H

#define DIFFUSION_KERNEL_R 1               // diffusion filter radius
#define diffK (DIFFUSION_KERNEL_R * 2 + 1) // diffusion filter width
#define BLOCK_SIZE 30                      // block size of environment based kernels
#define MEAN_FILTER_CENTER_WEIGHT 0.9      // how much weight center of diffusion filter should have (larger leads to slower diffusion)
#define VISUAL_SCALING 4                   // how much to multiply colors by in the final rendering of results
#define deltaT 1                           // time step between update steps
#define N_FOOD_TYPES 2                     // number of types of food (currently red/white)
#define PARTICLE_STOMACH_SIZE 500.0        // how much food of a certain type particles can have (unbounded leads to oversaturation of environment)

#include <cuda_runtime.h>
#include <curand_kernel.h>

// individual slime particle
struct SlimeParticle
{
    float x;
    float y;
    float orientation;
    curandState_t rng; // for CUDA random number generations
    float food[N_FOOD_TYPES];
};

__global__ void init_circle_particle_kernel(SlimeParticle *particles, int n, int *occupied, int w, int h);
__global__ void init_particle_kernel(SlimeParticle *particles, int n, int *occupied, int w, int h);
__global__ void add_food_kernel(float *food, float *food_pattern, int w, int h);
__device__ float sample_chemoattractant(SlimeParticle *p, float *env, float *food, int w, int h, float rotation_offset, float sensor_offset);
__global__ void sensor_stage_kernel(SlimeParticle *particles, int n, float *env, float *food, int w, int h, float sensor_angle, float rotation_angle, float sensor_offset);
__global__ void motor_stage_kernel(SlimeParticle *particles, int n, float *env, float *food, int *occupied, int w, int h, float sensor_angle, float rotation_angle, float step_size, float deposit_amount, float delta_t, float base_food);
__global__ void decay_chemoattractant_kernel(float *env, float *food, int *occupied, uint *result, int w, int h, float decay_amount);
__global__ void diffusion_kernel(float *env, float *env_dest, int w, int h);

#endif