#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <vector>

using namespace std;

#define TILE 16

void printCudaDeviceInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        cout << "Failed to get CUDA device count: " << cudaGetErrorString(err) << endl;
        return;
    }

    cout << "\n===== CUDA Device Information =====" << endl;
    cout << "Detected CUDA Devices: " << deviceCount << endl;

    for (int d = 0; d < deviceCount; d++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);

        cout << "\nDevice " << d << ": " << prop.name << endl;
        cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Multiprocessors: " << prop.multiProcessorCount << endl;
        cout << "  Global Memory (MB): " << (prop.totalGlobalMem / (1024 * 1024)) << endl;
        cout << "  Shared Memory per Block (KB): " << (prop.sharedMemPerBlock / 1024) << endl;
        cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
        cout << "  Warp Size: " << prop.warpSize << endl;
        cout << "  Memory Clock (MHz): " << (prop.memoryClockRate / 1000) << endl;
        cout << "  Core Clock (MHz): " << (prop.clockRate / 1000) << endl;
    }

    cout << "===================================\n" << endl;
}

// ---------------- VECTOR ADD ----------------
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

// ---------------- MATRIX MUL (GLOBAL) ----------------
__global__ void matMulGlobal(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

// ---------------- MATRIX MUL (SHARED) ----------------
__global__ void matMulShared(float *A, float *B, float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0;

    for (int t = 0; t < N / TILE; t++) {
        sA[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*N + col];

        __syncthreads();

        for (int i = 0; i < TILE; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    C[row*N + col] = sum;
}

// ---------------- CPU MATRIX ----------------
void cpuMatMul(vector<float>& A, vector<float>& B, vector<float>& C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i*N+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

// ---------------- MAIN ----------------
int main() {

    printCudaDeviceInfo();

    ofstream file("vector_matrix_result.txt");
    file << "SIZE,CPU_TIME,GLOBAL_TIME,SHARED_TIME,SPEEDUP_GLOBAL,SPEEDUP_SHARED\n";

    vector<int> sizes = {256, 512, 1024, 2048};

    for (int N : sizes) {

        int vecSize = N * 1024;

        // ---------- VECTOR ----------
        float *A, *B, *C;
        cudaMallocManaged(&A, vecSize*sizeof(float));
        cudaMallocManaged(&B, vecSize*sizeof(float));
        cudaMallocManaged(&C, vecSize*sizeof(float));

        for (int i = 0; i < vecSize; i++) {
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < vecSize; i++)
            C[i] = A[i] + B[i];
        auto end = chrono::high_resolution_clock::now();
        double cpuVec = chrono::duration<double, milli>(end - start).count();

        start = chrono::high_resolution_clock::now();
        vectorAdd<<<(vecSize+255)/256,256>>>(A,B,C,vecSize);
        cudaDeviceSynchronize();
        end = chrono::high_resolution_clock::now();
        double gpuVec = chrono::duration<double, milli>(end - start).count();

        cudaFree(A); cudaFree(B); cudaFree(C);

        // ---------- MATRIX ----------
        int size = N*N;
        vector<float> h_A(size,1.0f), h_B(size,2.0f), h_C(size);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size*sizeof(float));
        cudaMalloc(&d_B, size*sizeof(float));
        cudaMalloc(&d_C, size*sizeof(float));

        cudaMemcpy(d_A, h_A.data(), size*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), size*sizeof(float), cudaMemcpyHostToDevice);

        // CPU
        start = chrono::high_resolution_clock::now();
        cpuMatMul(h_A, h_B, h_C, N);
        end = chrono::high_resolution_clock::now();
        double cpuTime = chrono::duration<double, milli>(end - start).count();

        dim3 threads(TILE,TILE);
        dim3 blocks(N/TILE,N/TILE);

        // Global
        start = chrono::high_resolution_clock::now();
        matMulGlobal<<<blocks,threads>>>(d_A,d_B,d_C,N);
        cudaDeviceSynchronize();
        end = chrono::high_resolution_clock::now();
        double globalTime = chrono::duration<double, milli>(end - start).count();

        // Shared
        start = chrono::high_resolution_clock::now();
        matMulShared<<<blocks,threads>>>(d_A,d_B,d_C,N);
        cudaDeviceSynchronize();
        end = chrono::high_resolution_clock::now();
        double sharedTime = chrono::duration<double, milli>(end - start).count();

        double speedupGlobal = cpuTime / globalTime;
        double speedupShared = cpuTime / sharedTime;

        cout << "N=" << N
             << " CPU=" << cpuTime
             << " Global=" << globalTime
             << " Shared=" << sharedTime << endl;

        file << N << ","
             << cpuTime << ","
             << globalTime << ","
             << sharedTime << ","
             << speedupGlobal << ","
             << speedupShared << "\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    file.close();
    return 0;
}
