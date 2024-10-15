#include <stdio.h>
#include <stdarg.h>
#include <cuda_runtime.h>

#define LABEL_WIDTH 45

int EstimateTensorCores(int major, int minor, int smCount) {
    int tensorCoresPerSM = 0;

    if (major == 7 && minor == 0) {      // Volta (V100)
        tensorCoresPerSM = 8;
    } else if (major == 7 && minor == 5) { // Turing (RTX 20 Series)
        tensorCoresPerSM = 8;
    } else if (major == 8 && minor == 0) { // Ampere (A100)
        tensorCoresPerSM = 8;
    } else if (major == 8 && minor == 6) { // Ampere (RTX 30 Series)
        tensorCoresPerSM = 4;
    } else {
        tensorCoresPerSM = 0; // Unknown or unsupported architecture
    }

    int totalTensorCores = tensorCoresPerSM * smCount;
    return totalTensorCores;
}

void PrintBoth(FILE* fp, const char* format, ...) {
    va_list args1;
    va_start(args1, format);
    vfprintf(fp, format, args1);
    va_end(args1);

    va_list args2;
    va_start(args2, format);
    vprintf(format, args2);
    va_end(args2);
}

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    FILE *fp = fopen("device_info.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file 'device_info.txt' for writing.\n");
        return -1;
    }

    PrintBoth(fp, "Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        int smCount = deviceProp.multiProcessorCount;
        int tensorCores = EstimateTensorCores(deviceProp.major, deviceProp.minor, smCount);

        PrintBoth(fp, "=============================================================\n");
        PrintBoth(fp, "Device %d: %s\n", i, deviceProp.name);
        PrintBoth(fp, "=============================================================\n\n");

        PrintBoth(fp, "%-*s %d.%d\n", LABEL_WIDTH, "CUDA Capability Major/Minor version number:", deviceProp.major, deviceProp.minor);
        PrintBoth(fp, "%-*s %.2f GB\n", LABEL_WIDTH, "Total Global Memory:", (double)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        PrintBoth(fp, "%-*s %d\n", LABEL_WIDTH, "Streaming Multiprocessor (SM count):", smCount);

        if (tensorCores > 0) {
            PrintBoth(fp, "%-*s %d (Estimated)\n", LABEL_WIDTH, "Total Tensor Cores:", tensorCores);
        } else {
            PrintBoth(fp, "%-*s %s\n", LABEL_WIDTH, "Total Tensor Cores:", "Not Available");
        }

        PrintBoth(fp, "%-*s %d\n", LABEL_WIDTH, "Max Threads per Block:", deviceProp.maxThreadsPerBlock);
        PrintBoth(fp, "%-*s %d\n", LABEL_WIDTH, "Max Threads per Multiprocessor:", deviceProp.maxThreadsPerMultiProcessor);
        PrintBoth(fp, "%-*s %zu bytes\n", LABEL_WIDTH, "Max Shared Memory per Block:", deviceProp.sharedMemPerBlock);
        PrintBoth(fp, "%-*s %zu bytes\n", LABEL_WIDTH, "Max Shared Memory per Multiprocessor:", deviceProp.sharedMemPerMultiprocessor);

        PrintBoth(fp, "\nExample Compilation Command:\n");
        PrintBoth(fp, "  nvcc -arch=sm_%d%d -o example example.cu\n", deviceProp.major, deviceProp.minor);

        PrintBoth(fp, "\nAdditional Device Information:\n");
        PrintBoth(fp, "%-*s %d bytes\n", LABEL_WIDTH, "L2 Cache Size:", deviceProp.l2CacheSize);
        PrintBoth(fp, "%-*s %zu bytes\n", LABEL_WIDTH, "Total Constant Memory:", deviceProp.totalConstMem);
        PrintBoth(fp, "%-*s %d\n", LABEL_WIDTH, "Registers per Block:", deviceProp.regsPerBlock);
        PrintBoth(fp, "%-*s %s\n", LABEL_WIDTH, "Managed Memory:", deviceProp.managedMemory ? "Yes" : "No");
        PrintBoth(fp, "%-*s %d\n", LABEL_WIDTH, "Concurrent Kernels:", deviceProp.concurrentKernels);

        PrintBoth(fp, "\n");
    }

    fclose(fp);

    printf("Device information has been written to 'device_info.txt'\n");
    return 0;
}
