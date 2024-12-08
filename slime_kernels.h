#ifndef SLIME_KERNELS_H
#define SLIME_KERNELS_H

#define DIFFUSION_KERNEL_R 1
#define diffK (DIFFUSION_KERNEL_R * 2 + 1)
#define BLOCK_SIZE 30
#define MEAN_FILTER_CENTER_WEIGHT 0.9
#define VISUAL_SCALING 4
#define deltaT 1

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct SlimeParticle
{
    float x;
    float y;
    float orientation;
    curandState_t rng;
    float food;
};

__global__ void init_circle_particle_kernel(SlimeParticle *particles, int n, int *occupied, int w, int h);
__global__ void init_particle_kernel(SlimeParticle *particles, int n, int *occupied, int w, int h);
__global__ void add_food_kernel(float *food, float *food_pattern, int w, int h);
__device__ float sample_chemoattractant(SlimeParticle *p, float *env, float *food, int w, int h, float rotation_offset, float sensor_offset);
__global__ void sensor_stage_kernel(SlimeParticle *particles, int n, float *env, float *food, int w, int h, float sensor_angle, float rotation_angle, float sensor_offset);
__global__ void motor_stage_kernel(SlimeParticle *particles, int n, float *env, float *food, int *occupied, int w, int h, float sensor_angle, float rotation_angle, float step_size, float deposit_amount, float delta_t);
__global__ void decay_chemoattractant_kernel(float *env, float *food, int *occupied, uint *result, int w, int h, float decay_amount);
__global__ void diffusion_kernel(float *env, float *env_dest, int w, int h);

#endif