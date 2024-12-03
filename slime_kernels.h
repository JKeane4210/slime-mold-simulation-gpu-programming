#ifndef SLIME_KERNELS_H
#define SLIME_KERNELS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct SlimeParticle {
    float x;
    float y;
    float orientation;
    curandState_t rng;
};

__global__ void init_particle_kernel(SlimeParticle *particles, int n, int *occupied, int w, int h);
__device__ float sample_chemoattractant(SlimeParticle *p, float *env, int w, int h, float rotation_offset, float sensor_offset);
__global__ void sensor_stage_kernel(SlimeParticle *particles, int n, float *env, int w, int h, float sensor_angle, float rotation_angle, float sensor_offset);
__global__ void motor_stage_kernel(SlimeParticle *particles, int n, float *env, int *occupied, int w, int h, float sensor_angle, float rotation_angle, float step_size, float deposit_amount);
__global__ void decay_chemoattractant_kernel(float *env, int * occupied_d, uint *result, int w, int h, float decay_amount);

#endif