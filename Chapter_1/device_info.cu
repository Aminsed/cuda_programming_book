#include <stdio.h>
#include <cuda_runtime.h>

#define LABEL_WIDTH 45

// Helper function to estimate the number of Tensor Cores
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

    fprintf(fp, "Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        int smCount = deviceProp.multiProcessorCount;
        int tensorCores = EstimateTensorCores(deviceProp.major, deviceProp.minor, smCount);

        fprintf(fp, "=============================================================\n");
        fprintf(fp, "Device %d: %s\n", i, deviceProp.name);
        fprintf(fp, "=============================================================\n\n");

        // Essential Information
        fprintf(fp, "%-*s %d.%d\n", LABEL_WIDTH, "CUDA Capability Major/Minor version number:", deviceProp.major, deviceProp.minor);
        fprintf(fp, "%-*s %.2f GB\n", LABEL_WIDTH, "Total Global Memory:", (double)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        fprintf(fp, "%-*s %d\n", LABEL_WIDTH, "Multiprocessors:", smCount);

        if (tensorCores > 0) {
            fprintf(fp, "%-*s %d (Estimated)\n", LABEL_WIDTH, "Total Tensor Cores:", tensorCores);
        } else {
            fprintf(fp, "%-*s %s\n", LABEL_WIDTH, "Total Tensor Cores:", "Not Available");
        }

        fprintf(fp, "%-*s %d\n", LABEL_WIDTH, "Max Threads per Block:", deviceProp.maxThreadsPerBlock);
        fprintf(fp, "%-*s %d\n", LABEL_WIDTH, "Max Threads per Multiprocessor:", deviceProp.maxThreadsPerMultiProcessor);
        fprintf(fp, "%-*s %zu bytes\n", LABEL_WIDTH, "Max Shared Memory per Block:", deviceProp.sharedMemPerBlock);
        fprintf(fp, "%-*s %zu bytes\n", LABEL_WIDTH, "Max Shared Memory per Multiprocessor:", deviceProp.sharedMemPerMultiprocessor);

        // Example Compilation Command
        fprintf(fp, "\nExample Compilation Command:\n");
        fprintf(fp, "  nvcc -arch=sm_%d%d -o example example.cu\n", deviceProp.major, deviceProp.minor);

        // Additional Information (Keep or Remove based on book's focus)
        fprintf(fp, "\nAdditional Device Information:\n");
        fprintf(fp, "%-*s %d bytes\n", LABEL_WIDTH, "L2 Cache Size:", deviceProp.l2CacheSize);
        fprintf(fp, "%-*s %zu bytes\n", LABEL_WIDTH, "Total Constant Memory:", deviceProp.totalConstMem);
        fprintf(fp, "%-*s %d\n", LABEL_WIDTH, "Registers per Block:", deviceProp.regsPerBlock);
        fprintf(fp, "%-*s %s\n", LABEL_WIDTH, "Managed Memory:", deviceProp.managedMemory ? "Yes" : "No");
        fprintf(fp, "%-*s %d\n", LABEL_WIDTH, "Concurrent Kernels:", deviceProp.concurrentKernels);

        fprintf(fp, "\n");
    }

    fclose(fp);

    printf("Device information has been written to 'device_info.txt'\n");
    return 0;
}