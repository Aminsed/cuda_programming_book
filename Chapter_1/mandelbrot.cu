#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <ctime>

#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n",            \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

#define MAX_COLOR_VALUE 255

// Image dimensions (can be adjusted as needed)
#define DEFAULT_IMAGE_WIDTH 4096
#define DEFAULT_IMAGE_HEIGHT 4096

struct MandelbrotParams {
    int width;
    int height;
    int max_iterations;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
};

__global__ void mandelbrotKernel(unsigned char *output, MandelbrotParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Pixel's x coordinate
    int idy = blockIdx.y * blockDim.y + threadIdx.y;  // Pixel's y coordinate

    if (idx >= params.width || idy >= params.height) return;

    // Map pixel to complex plane
    double real = params.x_min + ((double)idx / params.width) * (params.x_max - params.x_min);
    double imag = params.y_min + ((double)idy / params.height) * (params.y_max - params.y_min);

    double c_real = real;
    double c_imag = imag;
    int iterations = 0;

    // Mandelbrot iteration
    for (; iterations < params.max_iterations; iterations++) {
        double real2 = real * real;
        double imag2 = imag * imag;

        // Check for divergence
        if (real2 + imag2 > 4.0) break;

        imag = 2.0 * real * imag + c_imag;
        real = real2 - imag2 + c_real;
    }

    // Smooth coloring
    double t;
    if (iterations < params.max_iterations) {
        double log_zn = log(real * real + imag * imag) / 2.0;
        double nu = log(log_zn / log(2.0)) / log(2.0);
        t = iterations + 1 - nu;
    } else {
        t = iterations;
    }

    // Normalize t
    t = t / params.max_iterations;

    // Color mapping (R, G, B)
    unsigned char r = (unsigned char)(9 * (1 - t) * t * t * t * MAX_COLOR_VALUE);
    unsigned char g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * MAX_COLOR_VALUE);
    unsigned char b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * MAX_COLOR_VALUE);

    // Store color in output array
    int pixel_index = 3 * (idy * params.width + idx);
    output[pixel_index] = r;
    output[pixel_index + 1] = g;
    output[pixel_index + 2] = b;
}

void writePPM(const char *filename, unsigned char *data, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "P6\n%d %d\n%d\n", width, height, MAX_COLOR_VALUE);
    size_t written = fwrite(data, sizeof(unsigned char), width * height * 3, fp);
    if (written != width * height * 3) {
        fprintf(stderr, "Error writing image data to file: %s\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    fclose(fp);
}

// Determine optimal block and grid dimensions
void getOptimalLaunchParameters(int width, int height, dim3 &block_dim, dim3 &grid_dim) {
    // Set block dimensions
    const int block_size_x = 16;
    const int block_size_y = 16;

    block_dim = dim3(block_size_x, block_size_y);

    // Calculate grid dimensions
    grid_dim = dim3((width + block_dim.x - 1) / block_dim.x,
                    (height + block_dim.y - 1) / block_dim.y);
}

// CPU multi-thread implementation
void mandelbrotCPU(unsigned char *output, MandelbrotParams params, int num_threads) {
    printf("CPU is using %d thread(s).\n", num_threads);

    int progress = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int idy = 0; idy < params.height; idy++) {
        for (int idx = 0; idx < params.width; idx++) {

            double real = params.x_min + ((double)idx / params.width) * (params.x_max - params.x_min);
            double imag = params.y_min + ((double)idy / params.height) * (params.y_max - params.y_min);

            double c_real = real;
            double c_imag = imag;
            int iterations = 0;

            for (; iterations < params.max_iterations; iterations++) {
                double real2 = real * real;
                double imag2 = imag * imag;
                if (real2 + imag2 > 4.0) break;
                imag = 2.0 * real * imag + c_imag;
                real = real2 - imag2 + c_real;
            }

            double t;
            if (iterations < params.max_iterations) {
                double log_zn = log(real * real + imag * imag) / 2.0;
                double nu = log(log_zn / log(2.0)) / log(2.0);
                t = iterations + 1 - nu;
            } else {
                t = iterations;
            }

            t = t / params.max_iterations;

            unsigned char r = (unsigned char)(9 * (1 - t) * t * t * t * MAX_COLOR_VALUE);
            unsigned char g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * MAX_COLOR_VALUE);
            unsigned char b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * MAX_COLOR_VALUE);

            int pixel_index = 3 * (idy * params.width + idx);
            output[pixel_index] = r;
            output[pixel_index + 1] = g;
            output[pixel_index + 2] = b;
        }

        // Update progress
        #pragma omp atomic
        progress++;

        if (progress % (params.height / 100) == 0 || progress == params.height) {
            int percent_complete = (int)((progress * 100.0) / params.height);
            printf("\rProgress: %d%%", percent_complete);
            fflush(stdout);
        }
    }
    printf("\n");
}

// GPU implementation
void computeMandelbrotGPU(unsigned char *h_output, MandelbrotParams params) {
    // Get optimal block and grid dimensions
    dim3 block_dim, grid_dim;
    getOptimalLaunchParameters(params.width, params.height, block_dim, grid_dim);

    // Report parameters being used
    printf("Using device 0\n");
    printf("Image Dimensions: %d x %d\n", params.width, params.height);
    printf("Block Dimensions: %d x %d\n", block_dim.x, block_dim.y);
    printf("Grid Dimensions:  %d x %d\n", grid_dim.x, grid_dim.y);

    size_t image_size = params.width * params.height * 3 * sizeof(unsigned char);

    // Allocate device memory
    unsigned char *d_output = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_output, image_size));

    printf("\nRunning Mandelbrot set on GPU...\n");
    {
        auto start_gpu = std::chrono::high_resolution_clock::now();

        mandelbrotKernel<<<grid_dim, block_dim>>>(d_output, params);
        CUDA_CHECK(cudaGetLastError());

        printf("Computing Mandelbrot set on GPU. Please wait...\n");
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_gpu = end_gpu - start_gpu;
        printf("GPU execution time: %f seconds\n", elapsed_gpu.count());

        CUDA_CHECK(cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaDeviceReset());
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(NULL));

    // Initialize parameters
    MandelbrotParams params;
    params.width = DEFAULT_IMAGE_WIDTH;
    params.height = DEFAULT_IMAGE_HEIGHT;
    params.max_iterations = 1000; // Default value, will be set by user input

    // Plane boundaries (can be adjusted for zooming)
    params.x_min = -0.74877;
    params.x_max = -0.74872;
    params.y_min = 0.06505;
    params.y_max = 0.06510;

    // Interactive selection
    int hardware_choice;

    std::cout << "Choose hardware:\n";
    std::cout << "1 - Single Thread CPU\n";
    std::cout << "2 - Multi Thread CPU (Use all available threads)\n";
    std::cout << "3 - Use GPU\n";
    std::cout << "Your choice: ";
    std::cin >> hardware_choice;

    std::cout << "Number of Iterations: ";
    std::cin >> params.max_iterations;

    // Validate user inputs
    if (hardware_choice < 1 || hardware_choice > 3) {
        std::cerr << "Invalid hardware choice. Exiting.\n";
        exit(EXIT_FAILURE);
    }
    if (params.max_iterations <= 0) {
        std::cerr << "Invalid number of iterations. Exiting.\n";
        exit(EXIT_FAILURE);
    }

    // Allocate memory for output
    size_t image_size = params.width * params.height * 3 * sizeof(unsigned char);
    unsigned char *h_output = (unsigned char *)malloc(image_size);
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory of size %zu bytes\n", image_size);
        exit(EXIT_FAILURE);
    }

    // Variables to store execution time
    std::chrono::duration<double> elapsed;
    const char *output_filename = nullptr;

    // Execute based on hardware choice
    switch (hardware_choice) {
        case 1: { // Single Thread CPU
            printf("\nRunning Mandelbrot set on CPU with a single thread...\n");
            auto start_cpu_single = std::chrono::high_resolution_clock::now();
            mandelbrotCPU(h_output, params, 1);
            auto end_cpu_single = std::chrono::high_resolution_clock::now();
            elapsed = end_cpu_single - start_cpu_single;
            printf("CPU (single-threaded) execution time: %f seconds\n", elapsed.count());

            output_filename = "mandelbrot_cpu_single.ppm";
            break;
        }
        case 2: { // Multi Thread CPU
            int num_cores = omp_get_max_threads();
            printf("\nRunning Mandelbrot set on CPU with %d threads...\n", num_cores);
            auto start_cpu_multi = std::chrono::high_resolution_clock::now();
            mandelbrotCPU(h_output, params, num_cores);
            auto end_cpu_multi = std::chrono::high_resolution_clock::now();
            elapsed = end_cpu_multi - start_cpu_multi;
            printf("CPU (multi-threaded) execution time: %f seconds\n", elapsed.count());

            output_filename = "mandelbrot_cpu_multi.ppm";
            break;
        }
        case 3: { // GPU
            computeMandelbrotGPU(h_output, params);
            // Note: Execution time is already printed inside computeMandelbrotGPU
            output_filename = "mandelbrot_gpu.ppm";
            break;
        }
        default:
            fprintf(stderr, "Invalid hardware choice.\n");
            free(h_output);
            exit(EXIT_FAILURE);
    }

    // Save the result
    if (output_filename) {
        writePPM(output_filename, h_output, params.width, params.height);
        printf("Saved output image to %s\n", output_filename);
    }

    free(h_output);

    return 0;
}