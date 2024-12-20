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
#include "libs/helper_cuda.h"

// OpenGL libraries
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

// slime simulation kernels
#include "slime_kernels.h"
#include "utils.h"

// #define SAVE_FRAME_PPM          // if last frame should be saved off to assets/tmp.ppm
#define BLOCK_SIZE_PARTICLE 256 // block size for particle kernels (could be improved for occupancy)

#define RA M_PI / 4        // = rotation angle
#define SA M_PI / 4        // = sensor angle
#define SO 9               // pixel(s) = sensor offset (original = 9)
#define SW 1               // pixel(s) = sensor width
#define SS 2               // pixel(s) = step size
#define depT 20            // how much chemoattractant is deposited (original = 5)
#define decayT 0.5         // decay rate of chemoattractant
#define DISPLAY_WIDTH 600  // how big to make display for rendering (width)
#define DISPLAY_HEIGHT 450 // how big to make display for rendering (height)
#define REFRESH_DELAY 20   // ms

int PARTICLE_PERCENT = 33;
int N_PARTICLES;
int ENV_WIDTH = 2000;  // how big to make environment (width)
int ENV_HEIGHT = 1500; // how big to make environment (height)
const int BENCHMARK = 0;
const int DISPLAY_RANDOM = 1;
const int DISPLAY_CIRCLE = 2;
const int DISPLAY_MSOE = 3;
int mode = DISPLAY_MSOE;

void display();
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void reshape(int x, int y);
void timerEvent(int value);
void initGL(int *argc, char **argv);
void initCuda();
void computeFPS();
GLuint compileASMShader(GLenum program_type, const char *code);
void initGLResources();
void cleanup();

StopWatchInterface *timer;
StopWatchInterface *kernel_timer;

unsigned int *img_host;
unsigned int *img_device;
unsigned int *tmp_img_device;

SlimeParticle *particles_d;
int *occupied_d;
float *env_h;
float *env_d;
float *env_dest_d;
float *tmp_d;
float *food_d;
float *food_pattern_h;
float *food_pattern_d;

dim3 dg((ENV_WIDTH - 1) / BLOCK_SIZE + 1, (ENV_WIDTH - 1) / BLOCK_SIZE + 1, 1);
dim3 db(BLOCK_SIZE, BLOCK_SIZE, 1);
dim3 db_diffusion(BLOCK_SIZE + diffK - 1, BLOCK_SIZE + diffK - 1, 1);

unsigned int frameCount = 0;
unsigned int fpsCount = 0;
unsigned int fpsLimit = 8;
float avgFPS = 0.0f;

// what is a PBO? -> pixel buffer object
cudaGraphicsResource *cuda_pbo_resource;
GLuint pbo;
GLuint texture_id;
GLuint shader;

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
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE); // only allow RGBA (can | GLUT_FLOAT to get other)
    glutInitWindowSize(DISPLAY_WIDTH, DISPLAY_HEIGHT);
    glutCreateWindow("CUDA Slime Mold Simulation");
    glutDisplayFunc(display);

    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void initCuda(int mode)
{
    // img
    HANDLE_ERROR(cudaMalloc((void **)&img_device, ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc((void **)&tmp_img_device, ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int)));

    // occupied
    HANDLE_ERROR(cudaMalloc((void **)&occupied_d, ENV_WIDTH * ENV_HEIGHT * sizeof(int)));
    // HANDLE_ERROR(cudaMemset(occupied_d, 0, ENV_WIDTH * ENV_HEIGHT * sizeof(int)));

    // creating environment
    env_h = (float *)malloc(N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float));
    HANDLE_ERROR(cudaMalloc((void **)&env_d, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float)));
    HANDLE_ERROR(cudaMemset(env_d, 0, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&env_dest_d, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float)));
    HANDLE_ERROR(cudaMemset(env_dest_d, 0, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float)));

    // food
    HANDLE_ERROR(cudaMalloc((void **)&food_d, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float)));
    HANDLE_ERROR(cudaMemset(food_d, 0, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float)));
    food_pattern_h = (float *)malloc(N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float));
    int *occupied_h = (int *)malloc(ENV_WIDTH * ENV_HEIGHT * sizeof(int));

    if (mode == DISPLAY_MSOE)
    {
        unsigned char *food_pattern_unscaled_h;
        unsigned int pattern_w;
        unsigned int pattern_h;
        sdkLoadPPM4<unsigned char>((const char *)"assets/MSOE.ppm", &food_pattern_unscaled_h, &pattern_w, &pattern_h);
        memset(food_pattern_h, 0, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float));
        memset(occupied_h, 0, ENV_WIDTH * ENV_HEIGHT * sizeof(int));
        int scale_factor = 6;
        for (int i = 0; i < pattern_h * scale_factor; ++i)
        {
            for (int j = 0; j < pattern_w * scale_factor; ++j)
            {
                int i_scaled = i / scale_factor;
                int j_scaled = j / scale_factor;
                int r = ENV_HEIGHT / 2 - pattern_h * scale_factor / 2 + i;
                int c = ENV_WIDTH / 2 - pattern_w * scale_factor / 2 + j;
                float value = (float)food_pattern_unscaled_h[((pattern_h - i_scaled) * pattern_w + j_scaled) * 4];
                food_pattern_h[(r * ENV_WIDTH + c) * N_FOOD_TYPES] = (value > 128) ? 3 : -1;
                food_pattern_h[(r * ENV_WIDTH + c) * N_FOOD_TYPES + 1] = (value < 128) ? 1 : -2;
                // occupied_h[r * ENV_WIDTH + c] = value > 128 ? 0 : 1;
            }
        }
        HANDLE_ERROR(cudaMalloc((void **)&food_pattern_d, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float)));
        HANDLE_ERROR(cudaMemcpy(food_pattern_d, food_pattern_h, N_FOOD_TYPES * ENV_WIDTH * ENV_HEIGHT * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(occupied_d, occupied_h, ENV_WIDTH * ENV_HEIGHT * sizeof(int), cudaMemcpyHostToDevice));
        add_food_kernel<<<dg, db>>>(food_d, food_pattern_d, ENV_WIDTH, ENV_HEIGHT);
    }

    // creating array of particles
    HANDLE_ERROR(cudaMalloc((void **)&particles_d, N_PARTICLES * sizeof(SlimeParticle)));
    printf("Dims: %d %d\n", (N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE);
    if (mode == DISPLAY_CIRCLE || mode == DISPLAY_MSOE)
    {
        init_circle_particle_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, occupied_d, ENV_WIDTH, ENV_HEIGHT);
    }
    else if (mode == DISPLAY_RANDOM || mode == BENCHMARK)
    {
        init_particle_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, occupied_d, ENV_WIDTH, ENV_HEIGHT);
    }
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // creating timers
    sdkCreateTimer(&timer);
    sdkCreateTimer(&kernel_timer);
}

// Calculate the Frames per second and print in the title bar
void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.0f / (sdkGetTimerValue(&timer) / 1000.0f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.0f);
        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps,
            "CUDA Slime Mold Simulation: "
            "%3.1f fps",
            avgFPS);
    glutSetWindowTitle(fps);
}

// Key Function - Simulates the update step and renders the image
void display()
{
    sdkStartTimer(&timer);

    // execute filter, writing results to pbo
    unsigned int *d_result;

    // get mapped pointer for result to render
    HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_result, &num_bytes, cuda_pbo_resource)); // should come back saying ENV_WIDTH * ENV_HEIGHT * 4 bytes
    // printf("Bytes accessible: %zu\n", num_bytes);

    // update step
    const float BASE_FOOD = mode == DISPLAY_MSOE ? 0.0 : 1.0;
    sensor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, food_d, ENV_WIDTH, ENV_HEIGHT, SA, RA, SO);
    motor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, food_d, occupied_d, ENV_WIDTH, ENV_HEIGHT, SA, RA, SS, depT, deltaT, BASE_FOOD);
    diffusion_kernel<<<dg, db_diffusion>>>(env_d, env_dest_d, ENV_WIDTH, ENV_HEIGHT);

    // swap env_d and env_dest_d
    tmp_d = env_d;
    env_d = env_dest_d;
    env_dest_d = tmp_d;
    decay_chemoattractant_kernel<<<dg, db>>>(env_d, food_d, occupied_d, d_result, ENV_WIDTH, ENV_HEIGHT, decayT);
    // add_food_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(food_d, food_pattern_d, ENV_WIDTH, ENV_HEIGHT); // TODO

    // HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

#ifdef SAVE_FRAME_PPM
    unsigned int *h_result = (unsigned int *)malloc(ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int));
    cudaMemcpy((unsigned char *)h_result, (unsigned char *)d_result, ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(cudaMemcpy(h_result, d_result, ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    sdkSavePPM4ub((const char *)"assets/tmp.ppm", (unsigned char *)h_result, ENV_WIDTH, ENV_HEIGHT);
#endif

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
    sdkDeleteTimer(&timer);
    sdkDeleteTimer(&kernel_timer);

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

    if (particles_d)
    {
        cudaFree(particles_d);
        particles_d = NULL;
    }

    if (env_d)
    {
        cudaFree(env_d);
        env_d = NULL;
    }

    if (env_dest_d)
    {
        cudaFree(env_dest_d);
        env_dest_d = NULL;
    }

    if (occupied_d)
    {
        cudaFree(occupied_d);
        occupied_d = NULL;
    }

    if (env_h)
    {
        free(env_h);
        env_h = NULL;
    }

    if (food_d)
    {
        cudaFree(food_d);
        food_d = NULL;
    }

    if (food_pattern_h)
    {
        free(food_pattern_h);
        food_pattern_h = NULL;
    }

    if (food_pattern_d)
    {
        cudaFree(food_pattern_d);
        food_pattern_d = NULL;
    }

    cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture_id);
    glDeleteProgramsARB(1, &shader);
}

// bencmarking tests
void benchmark(int n_trials, int n_init_steps)
{
    unsigned int *d_result;
    HANDLE_ERROR(cudaMalloc((void **)&d_result, ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int)));

    cudaEvent_t start, stop; // declare a start and stop event
    cudaEventCreate(&start); // create both events
    cudaEventCreate(&stop);

    // initalization steps
    for (int trial = 0; trial < n_init_steps; ++trial)
    {
        sensor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, food_d, ENV_WIDTH, ENV_HEIGHT, SA, RA, SO);
        motor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, food_d, occupied_d, ENV_WIDTH, ENV_HEIGHT, SA, RA, SS, depT, deltaT, 1.0);
        diffusion_kernel<<<dg, db_diffusion>>>(env_d, env_dest_d, ENV_WIDTH, ENV_HEIGHT);

        // swap env_d and env_dest_d
        tmp_d = env_d;
        env_d = env_dest_d;
        env_dest_d = tmp_d;
        decay_chemoattractant_kernel<<<dg, db>>>(env_d, food_d, occupied_d, d_result, ENV_WIDTH, ENV_HEIGHT, decayT);
    }

    // benchmarking trials
    for (int trial = 0; trial < n_trials; ++trial)
    {
        printf("\n*** Trial %d ***\n", trial + 1);

        // update step
        cudaEventRecord(start); // insert the start event into the stream
        sensor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, food_d, ENV_WIDTH, ENV_HEIGHT, SA, RA, SO);
        HANDLE_ERROR(cudaPeekAtLastError());
        cudaEventRecord(stop); // insert the stop event into the stream
        report_kernel_benchmark("sensor_stage_kernel", start, stop);

        // motor stage
        cudaEventRecord(start); // insert the start event into the stream
        motor_stage_kernel<<<(N_PARTICLES - 1) / BLOCK_SIZE_PARTICLE + 1, BLOCK_SIZE_PARTICLE>>>(particles_d, N_PARTICLES, env_d, food_d, occupied_d, ENV_WIDTH, ENV_HEIGHT, SA, RA, SS, depT, deltaT, 1.0);
        HANDLE_ERROR(cudaPeekAtLastError());
        cudaEventRecord(stop); // insert the stop event into the stream
        report_kernel_benchmark("motor_stage_kernel", start, stop);

        // diffusion
        cudaEventRecord(start); // insert the start event into the stream
        diffusion_kernel<<<dg, db_diffusion>>>(env_d, env_dest_d, ENV_WIDTH, ENV_HEIGHT);
        HANDLE_ERROR(cudaPeekAtLastError());
        cudaEventRecord(stop); // insert the stop event into the stream
        report_kernel_benchmark("diffusion_kernel", start, stop);

        // swap env_d and env_dest_d
        tmp_d = env_d;
        env_d = env_dest_d;
        env_dest_d = tmp_d;

        // decay chemoattractant
        cudaEventRecord(start); // insert the start event into the stream
        decay_chemoattractant_kernel<<<dg, db>>>(env_d, food_d, occupied_d, d_result, ENV_WIDTH, ENV_HEIGHT, decayT);
        HANDLE_ERROR(cudaPeekAtLastError());
        cudaEventRecord(stop); // insert the stop event into the stream
        report_kernel_benchmark("decay_chemoattractant_kernel", start, stop);
    }
    HANDLE_ERROR(cudaPeekAtLastError());

    // saves last frame to assets/tmp.ppm to validate working qualitatively
    unsigned int *h_result = (unsigned int *)malloc(ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int));
    HANDLE_ERROR(cudaMemcpy(h_result, d_result, ENV_WIDTH * ENV_HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < ENV_HEIGHT / 2; ++i)
    {
        for (int j = 0; j < ENV_WIDTH; ++j)
        {
            unsigned int tmp = h_result[(ENV_HEIGHT - 1 - i) * ENV_WIDTH + j];            // ENV_WIDTH - 1 -
            h_result[(ENV_HEIGHT - 1 - i) * ENV_WIDTH + j] = h_result[i * ENV_WIDTH + j]; // ENV_WIDTH - 1 -
            h_result[i * ENV_WIDTH + j] = tmp;
        }
    }
    sdkSavePPM4ub((const char *)"assets/tmp.ppm", (unsigned char *)h_result, ENV_WIDTH, ENV_HEIGHT);
    free(h_result);
    HANDLE_ERROR(cudaFree(d_result));
}

int main(int argc, char *argv[])
{
    // parsing command line arguments
    if (argc > 1)
        mode = atoi(argv[1]);
    if (argc > 2)
        ENV_HEIGHT = atoi(argv[2]);
    if (argc > 3)
        ENV_WIDTH = atoi(argv[3]);
    if (argc > 4)
        PARTICLE_PERCENT = atoi(argv[4]);
    N_PARTICLES = (int)((float)PARTICLE_PERCENT / 100.0 * (ENV_WIDTH * ENV_HEIGHT));
    printf("# Particles: %d\n", N_PARTICLES);
    printf("Mode: %s\n", mode > BENCHMARK ? "DISPLAY" : "BENCHMARK");

    // need for setting up where to display to on Linux
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    // intializing key arrays
    findCudaDevice(argc, (const char **)argv);
    initCuda(mode);

    // display modes (else benchmark mode)
    if (mode > BENCHMARK)
    {
        initGL(&argc, argv);
        initGLResources();
        glutCloseFunc(cleanup);
        glutMainLoop();
    }
    else
    {
        benchmark(10, 2000);
        cleanup();
    }
    return 0;
}
