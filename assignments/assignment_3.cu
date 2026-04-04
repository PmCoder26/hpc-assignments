#include <iostream>          // For input/output (cout, etc.)
#include <cuda_runtime.h>    // CUDA runtime functions
using namespace std;


// ==========================================================
// MAX REDUCTION KERNEL
// ==========================================================
__global__ void maxReductionKernel(int *d_arr, int n) {

    // Shared memory (fast memory shared within a block)
    __shared__ int s_data[256];

    // Local thread ID (within block)
    int tid = threadIdx.x;

    // Global index (unique for each thread across all blocks)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ------------------------------------------------------
    // Step 1: Load data from global memory → shared memory
    // ------------------------------------------------------

    if (idx < n)
        s_data[tid] = d_arr[idx];   // valid data
    else
        s_data[tid] = INT_MIN;      // neutral element for MAX

    // Wait until ALL threads finish loading
    __syncthreads();


    // ------------------------------------------------------
    // Step 2: Parallel Reduction (Tree-like reduction)
    // ------------------------------------------------------

    for(int stride = 1; stride < blockDim.x; stride *= 2) {

        /*
        Only specific threads participate:
        Example:
        stride = 1 → threads 0,2,4,6...
        stride = 2 → threads 0,4,8...
        */

        if(tid % (2 * stride) == 0 && tid + stride < blockDim.x) {

            // Compare and store maximum
            s_data[tid] = max(s_data[tid], s_data[tid + stride]);
        }

        // Synchronize before next iteration
        __syncthreads();
    }

    // ------------------------------------------------------
    // Step 3: Final result writing
    // ------------------------------------------------------

    if(tid == 0) {
        // Only one thread writes final result
        d_arr[0] = max(d_arr[0], s_data[0]);
    }
}


// ==========================================================
// MIN REDUCTION KERNEL (same as MAX but using min())
// ==========================================================
__global__ void minReductionKernel(int *d_arr, int n) {

    __shared__ int s_data[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data with neutral element for MIN
    if (idx < n)
        s_data[tid] = d_arr[idx];
    else
        s_data[tid] = INT_MAX;

    __syncthreads();

    // Reduction loop
    for(int stride = 1; stride < blockDim.x; stride *= 2) {

        if(tid % (2 * stride) == 0 && tid + stride < blockDim.x) {

            // Take minimum instead of maximum
            s_data[tid] = min(s_data[tid], s_data[tid + stride]);
        }

        __syncthreads();
    }

    if(tid == 0) {
        d_arr[0] = min(d_arr[0], s_data[0]);
    }
}


// ==========================================================
// SUM REDUCTION KERNEL
// ==========================================================
__global__ void sumReductionKernel(int *d_arr, int *sum_arr, int n) {

    __shared__ int s_data[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    if(idx < n)
        s_data[tid] = d_arr[idx];
    else
        s_data[tid] = 0;   // neutral element for SUM

    __syncthreads();

    // Reduction loop
    for(int stride = 1; stride < blockDim.x; stride *= 2) {

        if(tid % (2 * stride) == 0 && tid + stride < blockDim.x) {

            // Add pair elements
            s_data[tid] += s_data[tid + stride];
        }

        __syncthreads();
    }

    // Each block writes its partial sum
    if(tid == 0) {
        sum_arr[blockIdx.x] = s_data[0];
    }
}


// ==========================================================
// MAIN FUNCTION (HOST CODE)
// ==========================================================
int main() {

    // Input array (stored in CPU memory)
    int arr[10] = {1,2,3,4,5,6,7,8,9,10};

    int n = 10;
    int size = n * sizeof(int);

    int *d_arr;   // Pointer for GPU memory

    // ------------------------------------------------------
    // Allocate memory on GPU
    // ------------------------------------------------------
    cudaMalloc((void**)&d_arr, size);

    // ------------------------------------------------------
    // Copy data from CPU → GPU
    // ------------------------------------------------------
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // Print input
    printf("Array: ");
    for(int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");


    // ======================================================
    // MAX OPERATION
    // ======================================================

    // Launch kernel: 1 block, 10 threads
    maxReductionKernel<<<1, 10>>>(d_arr, n);

    int result[10];

    // Copy result back to CPU
    cudaMemcpy(result, d_arr, size, cudaMemcpyDeviceToHost);

    printf("Max: %d\n", result[0]);


    // ======================================================
    // MIN OPERATION
    // ======================================================

    // Reset data
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    minReductionKernel<<<1, 10>>>(d_arr, n);

    cudaMemcpy(result, d_arr, size, cudaMemcpyDeviceToHost);

    printf("Min: %d\n", result[0]);


    // ======================================================
    // SUM OPERATION
    // ======================================================

    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int *d_sum;

    // Allocate space for 2 blocks result
    cudaMalloc((void**)&d_sum, 2 * sizeof(int));

    // Launch: 2 blocks × 5 threads
    sumReductionKernel<<<2, 5>>>(d_arr, d_sum, n);

    int h_sum[2];

    // Copy partial sums
    cudaMemcpy(h_sum, d_sum, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Final sum on CPU
    int finalSum = h_sum[0] + h_sum[1];

    printf("Sum: %d\n", finalSum);


    // ======================================================
    // AVERAGE
    // ======================================================

    float avg = (float)finalSum / n;

    printf("Average: %f\n", avg);


    // ------------------------------------------------------
    // Free GPU memory
    // ------------------------------------------------------
    cudaFree(d_arr);
    cudaFree(d_sum);

    return 0;
}