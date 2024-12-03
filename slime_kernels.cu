#include "slime_kernels.h"

#define OVERLAPPING_PARTICLES
// #define DISPLAY_PARTICLE_LOCATION
#define DISPLAY_SLIME_TRAIL

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
        particles[i].orientation = r_theta; // + M_PI;
        // if (particles[i].orientation > 2 * M_PI)
        //     particles[i].orientation -= 2 * M_PI;
        particles[i].rng = state;
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
    }
}

__device__ float sample_chemoattractant(SlimeParticle *p, float *env, int w, int h, float rotation_offset, float sensor_offset)
{
    float angle = p->orientation + rotation_offset;
    // if (angle < 0)
    //     angle += 2 * M_PI;
    // if (angle > 2 * M_PI)
    //     angle -= 2 * M_PI;
    int s_x = (int)round(p->x + sensor_offset * cos(angle));
    int s_y = (int)round(p->y + sensor_offset * sin(angle));
    if (s_y >= 0 && s_y < h && s_x >= 0 && s_x < w)
        return env[s_y * w + s_x];
    else
        p->orientation += curand_uniform(&(p->rng)) * M_PI; // keep it in bounds
    // if (p->orientation < 0)
    //     p->orientation += 2 * M_PI;
    // if (p->orientation > 2 * M_PI)
    //     p->orientation -= 2 * M_PI;
    return 0;
}

__global__ void sensor_stage_kernel(SlimeParticle *particles, int n, float *env, int w, int h, float sensor_angle, float rotation_angle, float sensor_offset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        SlimeParticle *p = &particles[i];
        float F = sample_chemoattractant(p, env, w, h, 0, sensor_offset);
        float FR = sample_chemoattractant(p, env, w, h, -sensor_angle, sensor_offset);
        float FL = sample_chemoattractant(p, env, w, h, sensor_angle, sensor_offset);
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
        // if (p->orientation < 0)
        //     p->orientation += 2 * M_PI;
        // if (p->orientation > 2 * M_PI)
        //     p->orientation -= 2 * M_PI;
    }
}

__global__ void motor_stage_kernel(SlimeParticle *particles, int n, float *env, int *occupied, int w, int h, float sensor_angle, float rotation_angle, float step_size, float deposit_amount, float delta_t)
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
#ifndef OVERLAPPING_PARTICLES
            if (atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1) == 0) // not occupied
            {
                // atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1);
                atomicExch(&(occupied[p_y_i * w + p_x_i]), 0); // clear previous location
                p->x = n_x;
                p->y = n_y;
                if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
                    atomicAdd(&(env[p_y_i * w + p_x_i]), deposit_amount); // deposit trail in new location
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
                atomicAdd(&(env[p_y_i * w + p_x_i]), deposit_amount); // deposit trail in new location
#endif
        }
    }
}

__global__ void decay_chemoattractant_kernel(float *env, int *occupied_d, uint *result, int w, int h, float decay_amount)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < w && col >= 0 && row < h && row >= 0)
    {
        float value = max(env[row * w + col] - decay_amount, 0.0);
        env[row * w + col] = value;
        value = min(value, 255.0);
#ifndef DISPLAY_SLIME_TRAIL
        value = 0;
#endif
#ifdef DISPLAY_PARTICLE_LOCATION
        value = max(value, (occupied_d[row * w + col] > 0) ? 255.0 : 0);
#endif
        result[row * w + col] = (0u << 24) |
                                ((unsigned int)(value) << 16) |
                                ((unsigned int)(value) << 8) |
                                ((unsigned int)(value));
    }
}