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

// libraries from NVIDIA samples
#include "libs/helper_gl.h"
#include "libs/helper_timer.h"
#include "libs/helper_image.h"
#include "libs/helper_string.h"

// OpenGL libraries
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#define BLOCK_SIZE 32
#define BLOCK_SIZE_PARTICLE 512
#define DEBUG_LEVEL 0
#define HARD_FAIL false

#define RA M_PI / 4 // = rotation angle
#define SA M_PI / 4 // = sensor angle
#define SO 9        // pixel(s) = sensor offset (original = 9)
#define SW 1        // pixel(s) = sensor width
#define SS 1        // pixel(s) = step size
#define depT 20     // how much chemoattractant is deposited (original = 5)
#define decayT 0.5  // decay rate of chemoattractant
#define ENV_WIDTH 400
#define ENV_HEIGHT 300

#define REFRESH_DELAY 10 // ms

struct SlimeParticle
{
    float x;
    float y;
    float orientation;
    curandState_t rng;
};

#define MASK_WIDTH 3
__constant__ float K[MASK_WIDTH][MASK_WIDTH];

void display();

StopWatchInterface *timer;
StopWatchInterface *kernel_timer;

// handle error macro
inline void HandleError(cudaError_t err, const char *file, const int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

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
        while (atomicAdd(&(occupied[ry * w + rx]), 1) != 0)
        {
            rx = ((int)ceilf(curand_uniform(&state) * w)) - 1;
            ry = ((int)ceilf(curand_uniform(&state) * h)) - 1;
        }
        particles[i].x = (float)rx;
        particles[i].y = (float)ry;
        particles[i].orientation = r_theta;
        particles[i].rng = state;
    }
}

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

__device__ float sample_chemoattractant(SlimeParticle *p, float *env, int w, int h, float rotation_offset, float sensor_offset)
{
    float angle = p->orientation + rotation_offset;
    if (angle < 0)
        angle += 2 * M_PI;
    if (angle > 2 * M_PI)
        angle -= 2 * M_PI;
    int s_x = (int)round(p->x + sensor_offset * cos(angle));
    int s_y = (int)round(p->y + sensor_offset + sin(angle));
    if (s_y >= 0 && s_y < ENV_HEIGHT && s_x >= 0 && s_x < ENV_WIDTH)
        return env[s_y * w + s_x];
    else
        p->orientation += curand_uniform(&(p->rng)) * M_PI_2 + M_PI_4; // keep it in bounds
    if (p->orientation > 2 * M_PI)
        p->orientation -= 2 * M_PI;
    return 0;
}

__global__ void sensor_stage_kernel(SlimeParticle *particles, int n, float *env, int w, int h, float sensor_angle, float rotation_angle)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        SlimeParticle *p = &particles[i];
        float F = sample_chemoattractant(p, env, w, h, 0, SO);
        float FR = sample_chemoattractant(p, env, w, h, -sensor_angle, SO);
        float FL = sample_chemoattractant(p, env, w, h, sensor_angle, SO);
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
        if (p->orientation < 0)
            p->orientation += 2 * M_PI;
        if (p->orientation > 2 * M_PI)
            p->orientation -= 2 * M_PI;
    }
}

__global__ void motor_stage_kernel(SlimeParticle *particles, int n, float *env, int *occupied, int w, int h, float sensor_angle, float rotation_angle)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        SlimeParticle *p = &particles[i];
        float n_x = p->x + cos(p->orientation) * SS;
        float n_y = p->y + sin(p->orientation) * SS;
        int p_x_i = (int)round(p->x);
        int p_y_i = (int)round(p->y);
        int n_x_i = (int)round(n_x);
        int n_y_i = (int)round(n_y);
        if (atomicAdd(&(occupied[n_y_i * w + n_x_i]), 1) == 0)
        {                                                  // not occupied
            atomicExch(&(occupied[p_y_i * w + p_x_i]), 0); // clear previous location
            p->x = n_x;
            p->y = n_y;
            if (p_y_i < h && p_y_i >= 0 && p_x_i < w && p_x_i >= 0)
                atomicAdd(&(env[p_y_i * w + p_x_i]), depT); // deposit trail in new location
        }
        else
        {
            p->orientation += curand_uniform(&(p->rng)) * M_PI_2 - M_PI_4; // choose random new orientation
            if (p->orientation < 0)
                p->orientation += 2 * M_PI;
            if (p->orientation > 2 * M_PI)
                p->orientation -= 2 * M_PI;
        }
    }
}

__global__ void decay_chemoattractant_kernel(float *env, uint * result, int w, int h)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < w && col >= 0 && row < h && row >= 0)
    {
        float value = max(env[row * w + col] - decayT, 0.0);
        env[row * w + col] = value;
        value = min(value, 255.0);
        result[row * w + col] = (255u << 24) |
            ((unsigned int)(value) << 16) |
            ((unsigned int)(value) << 8) |
            ((unsigned int)(value));
    }
}

void gray_scale_image_to_file(const char *cpu_output_file, float *env)
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
            fputc((int)(min(255.0, env[i * ENV_WIDTH + j])), output_file_handle); // TODO
        }
    }
    fflush(output_file_handle);
    fclose(output_file_handle);
}

cudaArray *device_array;
cudaArray *device_tmp_array;
cudaTextureObject_t rgba_texture_object;
cudaTextureObject_t rgba_tmp_texture_object;

void initTexture(int width, int height, void *pImage)
{
    // RGBA channel setup
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    // allocate CUDA array
    HANDLE_ERROR(cudaMallocArray(&device_array, &channelDesc, width, height));
    HANDLE_ERROR(cudaMallocArray(&device_tmp_array, &channelDesc, width, height));

    // copy image over to array (but I don't have an image)
    // HANDLE_ERROR(cudaMemcpy2DToArray(device_array, 0, 0, pImage, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

    // setting up a texture resource and descriptor
    cudaResourceDesc texture_resource;
    cudaTextureDesc texture_descriptor;

    memset(&texture_resource, 0, sizeof(cudaResourceDesc));
    texture_resource.resType = cudaResourceTypeArray;
    texture_resource.res.array.array = device_array;

    memset(&texture_descriptor, 0, sizeof(cudaTextureDesc));
    texture_descriptor.normalizedCoords = false;               // what does this mean?
    texture_descriptor.filterMode = cudaFilterModeLinear;      // what does this mean?
    texture_descriptor.addressMode[0] = cudaAddressModeWrap;   // what does this mean?
    texture_descriptor.addressMode[1] = cudaAddressModeWrap;   // what does this mean?
    texture_descriptor.readMode = cudaReadModeNormalizedFloat; // what does this mean?

    // create the texture object (uses the resource and descriptor)
    HANDLE_ERROR(cudaCreateTextureObject(&rgba_texture_object, &texture_resource, &texture_descriptor, NULL));

    // resetting texture resource and descriptor
    memset(&texture_resource, 0, sizeof(cudaResourceDesc));
    texture_resource.resType = cudaResourceTypeArray;
    texture_resource.res.array.array = device_tmp_array;

    memset(&texture_descriptor, 0, sizeof(cudaTextureDesc));
    texture_descriptor.normalizedCoords = false;               // what does this mean?
    texture_descriptor.filterMode = cudaFilterModeLinear;      // what does this mean?
    texture_descriptor.addressMode[0] = cudaAddressModeWrap;   // what does this mean?
    texture_descriptor.addressMode[1] = cudaAddressModeWrap;   // what does this mean?
    texture_descriptor.readMode = cudaReadModeNormalizedFloat; // what does this mean?

    // create the temporary texture object (uses the resource and descriptor)
    HANDLE_ERROR(cudaCreateTextureObject(&rgba_tmp_texture_object, &texture_resource, &texture_descriptor, NULL));
}

void freeTextures()
{
    HANDLE_ERROR(cudaDestroyTextureObject(rgba_texture_object));
    HANDLE_ERROR(cudaDestroyTextureObject(rgba_tmp_texture_object));
    HANDLE_ERROR(cudaFreeArray(device_array));
    HANDLE_ERROR(cudaFreeArray(device_tmp_array));
}

// Keyboard callback function for OpenGL (GLUT)
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case 27:
        glutDestroyWindow(glutGetWindow());
        return;
        break;

    default:
        break;
    }
}

// Resizing the window
void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

// Timer Event so we can refresh the display
void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA); // only allow RGBA (can | GLUT_FLOAT to get other)
    glutInitWindowSize(768, 768);
    glutCreateWindow("CUDA Slime Mold Simulation");
    glutDisplayFunc(display);

    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // if (!isGLVersionSupported(2, 0) ||
    //     !areGLExtensionsSupported(
    //         "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    // {
    //     printf("Error: failed to get minimal extensions for demo\n");
    //     printf("This sample requires:\n");
    //     printf("  OpenGL version 2.0\n");
    //     printf("  GL_ARB_vertex_buffer_object\n");
    //     printf("  GL_ARB_pixel_buffer_object\n");
    //     exit(EXIT_FAILURE);
    // }
}

unsigned int *img_host;
unsigned int *img_device;
unsigned int *tmp_img_device;

SlimeParticle *particles_d;
int *occupied_d;
float *env_h;
float *env_d;

#define N_PARTICLES 1600
dim3 dg((ENV_WIDTH - 1) / BLOCK_SIZE + 1, (ENV_WIDTH - 1) / BLOCK_SIZE + 1, 1);
dim3 db(32, 32, 1);

void initCuda()
{
    HANDLE_ERROR(cudaMalloc((void **)&img_device, ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **)&tmp_img_device, ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int)));

    HANDLE_ERROR(cudaMalloc((void **)&occupied_d, ENV_WIDTH * ENV_HEIGHT * sizeof(int)));
    HANDLE_ERROR(cudaMemset(occupied_d, 0, ENV_WIDTH * ENV_HEIGHT * sizeof(int)));

    env_h = (float *)malloc(ENV_WIDTH * ENV_HEIGHT * sizeof(float));
    HANDLE_ERROR(cudaMalloc((void **)&env_d, ENV_WIDTH * ENV_HEIGHT * sizeof(float)));
    HANDLE_ERROR(cudaMemset(env_d, 0, ENV_WIDTH * ENV_HEIGHT * sizeof(float)));

    // const int N_PARTICLES = 1600;
    HANDLE_ERROR(cudaMalloc((void **)&particles_d, N_PARTICLES * sizeof(SlimeParticle)));
    printf("Dims: %d %d\n", (N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE);
    init_particle_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, occupied_d, ENV_WIDTH, ENV_HEIGHT);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaFree(particles_d));
    HANDLE_ERROR(cudaFree(env_d));
    HANDLE_ERROR(cudaFree(occupied_d));

    initTexture(ENV_WIDTH, ENV_HEIGHT, img_host);

    // creating timers
    sdkCreateTimer(&timer);
    sdkCreateTimer(&kernel_timer);
}

unsigned int frameCount = 0;
unsigned int fpsCount = 0;
unsigned int fpsLimit = 8;
float avgFPS = 0.0f;

// Calculate the Frames per second and print in the title bar
void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.0f);
        sdkResetTimer(&timer);
    }
}

// what is a PBO?
cudaGraphicsResource *cuda_pbo_resource;
GLuint pbo;
GLuint texture_id;
GLuint shader;

void display()
{
    sdkStartTimer(&timer);

    // execute filter, writing results to pbo
    unsigned int *d_result;

    HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_result, &num_bytes, cuda_pbo_resource)); // should come back saying ENV_WIDTH * ENV_HEIGHT * 4 bytes
    // printf("Bytes accessible: %zu\n", num_bytes);
    // boxFilterRGBA(d_img, d_temp, d_result, ENV_WIDTH, ENV_HEIGHT, filter_radius,
    //               iterations, nthreads, kernel_timer);

    // update step
    sensor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, ENV_WIDTH, ENV_HEIGHT, SA, RA);
    motor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, occupied_d, ENV_WIDTH, ENV_HEIGHT, SA, RA);
    decay_chemoattractant_kernel<<<dg, db>>>(env_d, d_result, ENV_WIDTH, ENV_HEIGHT);

    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // OpenGL display code path
    {
        glClear(GL_COLOR_BUFFER_BIT);

        // load texture from pbo
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ENV_WIDTH, ENV_HEIGHT, GL_RGBA,
                        GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        // fragment program is required to display floating point texture
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(0.0f, 0.0f);
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(1.0f, 0.0f);
            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(1.0f, 1.0f);
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(0.0f, 1.0f);
        }
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);
    }

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

// shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
                       (GLsizei)strlen(code), (GLubyte *)code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        printf("Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

// This is where we create the OpenGL PBOs, FBOs, and texture resources
void initGLResources()
{
    // create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, ENV_WIDTH * ENV_HEIGHT * sizeof(GLubyte) * 4,
                 img_host, GL_STREAM_DRAW_ARB);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ENV_WIDTH, ENV_HEIGHT, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void cleanup()
{
    // deleting timers
    //   sdkDeleteTimer(&timer);
    //   sdkDeleteTimer(&kernel_timer);

    if (img_host)
    {
        free(img_host);
        img_host = NULL;
    }

    if (img_device)
    {
        cudaFree(img_device);
        img_device = NULL;
    }

    if (tmp_img_device)
    {
        cudaFree(tmp_img_device);
        tmp_img_device = NULL;
    }

    // Refer to boxFilter_kernel.cu for implementation
    freeTextures();

    cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture_id);
    glDeleteProgramsARB(1, &shader);
}

int main(int argc, char *argv[])
{

    // char buffer[300];
    // const int N_STEPS = 1000;
    // for (int i = 0; i < N_STEPS; ++i)
    // {
    //     // update step
    //     sensor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, ENV_WIDTH, ENV_HEIGHT, SA, RA);
    //     motor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, occupied_d, ENV_WIDTH, ENV_HEIGHT, SA, RA);
    //     decay_chemoattractant_kernel<<<dg, db>>>(env_d, ENV_WIDTH, ENV_HEIGHT);

    //     // save image to file
    //     cudaMemcpy(env_h, env_d, ENV_WIDTH * ENV_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    //     sprintf(buffer, "./Desktop/Projects/slime-mold-simulation-gpu-programming/frames/frame_%d.ppm", i);
    //     gray_scale_image_to_file(buffer, env_h);
    // }

    // printf("Done!\n");

    // command: nvcc -lGL -lGLU -lglut


    /*
    
        Compile Command: nvcc slime_gpu_gl2.cu -L libs -o slimeGL -lGL -lGLU -lglut
    
    */

    // OpenGL version
    initGL(&argc, argv);
    initCuda();
    initGLResources();
    glutCloseFunc(cleanup);
    glutMainLoop();

    return 0;
}
