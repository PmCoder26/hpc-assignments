#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

void printCudaDeviceInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cout << "Failed to get CUDA device count: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "\n===== CUDA Device Information =====" << std::endl;
    std::cout << "Detected CUDA Devices: " << deviceCount << std::endl;

    for (int d = 0; d < deviceCount; d++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);

        std::cout << "\nDevice " << d << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Global Memory (MB): " << (prop.totalGlobalMem / (1024 * 1024)) << std::endl;
        std::cout << "  Shared Memory per Block (KB): " << (prop.sharedMemPerBlock / 1024) << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Memory Clock (MHz): " << (prop.memoryClockRate / 1000) << std::endl;
        std::cout << "  Core Clock (MHz): " << (prop.clockRate / 1000) << std::endl;
    }

    std::cout << "===================================\n" << std::endl;
}

__global__ void sum_reduction(float* d_out, float* d_in, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? d_in[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

// -------- FULL REDUCTION (IMPORTANT FIX) --------
float gpu_sum(float* d_in, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float *d_out;
    cudaMalloc(&d_out, blocks * sizeof(float));

    sum_reduction<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    // Copy partial sums to CPU
    std::vector<float> h_out(blocks);
    cudaMemcpy(h_out.data(), d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float final_sum = 0;
    for (int i = 0; i < blocks; i++)
        final_sum += h_out[i];

    cudaFree(d_out);
    return final_sum;
}

// -------- TEST FUNCTION --------
void run_test(int n, std::ofstream& outfile) {
    size_t bytes = n * sizeof(float);
    std::vector<float> h_in(n, 1.0f);

    float *d_in;
    cudaMalloc(&d_in, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // -------- SERIAL --------
    float h_sum_cpu = 0;
    auto start_s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
        h_sum_cpu += h_in[i];
    auto end_s = std::chrono::high_resolution_clock::now();
    double time_s = std::chrono::duration<double, std::milli>(end_s - start_s).count();

    // -------- PARALLEL --------
    auto start_p = std::chrono::high_resolution_clock::now();
    float h_sum_gpu = gpu_sum(d_in, n);
    auto end_p = std::chrono::high_resolution_clock::now();
    double time_p = std::chrono::duration<double, std::milli>(end_p - start_p).count();

    int cores = 2560; // RTX 3060 GPU cores

    double speedup = time_s / time_p;
    double efficiency = speedup / cores;

    std::cout << "N=" << n
              << " CPU=" << h_sum_cpu
              << " GPU=" << h_sum_gpu
              << " Speedup=" << speedup << std::endl;

    outfile << n << ","
            << time_s << ","
            << time_p << ","
            << speedup << ","
            << efficiency << std::endl;

    cudaFree(d_in);
}

// -------- MAIN --------
int main() {

    printCudaDeviceInfo();

    std::ofstream outfile("reduction_result.txt");
    outfile << "N,SERIAL,PARALLEL,SPEEDUP,EFFICIENCY\n";

    int sizes[] = {1000, 10000, 100000, 1000000, 5000000};

    for (int n : sizes)
        run_test(n, outfile);

    outfile.close();

    return 0;
}
