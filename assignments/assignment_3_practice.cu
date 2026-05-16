#include<iostream>
#include<cuda_runtime.h>
#include<chrono>
#include<cstdlib>
using namespace std;
using namespace chrono;



__global__ void minReductionKernel(int* d_arr, int* d_partial, int n) {
    
    __shared__ int sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        sdata[tid] = d_arr[idx];
    } else {
        sdata[tid] = INT_MAX;
    }

    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        
        if(tid % (2 * stride) == 0 && tid + stride < blockDim.x) {
            sdata[tid] = min(sdata[tid], sdata[tid + stride]);
        }

        __syncthreads();
    }

    if(tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

void minFunc(int* arr, int n) {
  int result = INT_MAX;

  for(int x = 0; x < n; x++) {
    result = min(result, arr[x]);
  }
}

__global__ void maxReductionKernel(int* d_arr, int* d_partial, int n) {

  __shared__ int sdata[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < n) {
    sdata[tid] = d_arr[idx];
  } else {
    sdata[tid] = INT_MIN;
  }

  __syncthreads();

  for(int stride = 1; stride < blockDim.x; stride *= 2) {

    if(tid % (2 * stride) == 0 && tid + stride < blockDim.x) {
      sdata[tid] = max(sdata[tid], sdata[tid + stride]);
    }    

    __syncthreads();
  }

  if(tid == 0) {    
    d_partial[blockIdx.x] = sdata[0];
  }
}

void maxFunc(int* arr, int n) {
  int result = INT_MIN;

  for(int x = 0; x < n; x++) {
    result = max(result, arr[x]);
  }
}

__global__ void sumReductionKernel(int* d_arr, int* sum_arr, int n) {

  __shared__ int sdata[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < n) {
    sdata[tid] = d_arr[idx];
  } else {
    sdata[tid] = 0;
  }

  __syncthreads();

  for(int stride = 1; stride < blockDim.x; stride *= 2) {
    if(tid % (2 * stride) == 0 && tid + stride < blockDim.x) {
      sdata[tid] += sdata[tid + stride];
    }

    __syncthreads();
  }

  if(tid == 0) {
    sum_arr[blockIdx.x] = sdata[0];
  }

}

void sumFunc(int* arr, int n) {
  int sum = 0;
  for(int x = 0; x < n; x++) {
    sum += arr[x];
  }
}


int main() {

    int n = 100000;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int size = n * sizeof(int);

    int* d_arr;
    int* d_partial;

    int arr[n];
    int pass_1_result[blocks];
    int pass_2_result[1];

    for(int x = 0; x < n; x++) {
      arr[x] = rand() % 1000;
    }
    

    // Min operation.
    cudaMalloc((void**) & d_arr, size);
    cudaMalloc((void**) & d_partial, blocks * sizeof(int));
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);  

    // pass-1.
    auto start = high_resolution_clock::now();

    minReductionKernel<<<blocks, threads>>>(d_arr, d_partial, n);
    cudaMemcpy(pass_1_result, d_partial, blocks * sizeof(int), cudaMemcpyDeviceToHost);    
    cudaFree(d_arr);
    cudaMalloc((void**) & d_arr, blocks * sizeof(int));

    // pass-2.
    minReductionKernel<<<1, blocks>>>(d_partial, d_arr, blocks);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto parDuration = duration_cast<milliseconds>(stop - start);

    cudaMemcpy(pass_2_result, d_arr, blocks * sizeof(int), cudaMemcpyDeviceToHost);    
    printf("Minimum element is: %d\n", pass_2_result[0]);

    cudaFree(d_arr);
    cudaFree(d_partial);

    cudaDeviceSynchronize();

    start = high_resolution_clock::now();
    int min = minFunc(arr, n);
    stop = high_resolution_clock::now();
    auto seqDuration = duration_cast<milliseconds>(stop - start);

    printf("Sequential Time(ms): %ld\n", seqDuration.count());
    printf("Paralle Time(ms): %ld\n", parDuration.count());


    // Max operation.
    cudaMalloc((void**) & d_arr, size);
    cudaMalloc((void**) & d_partial, blocks * sizeof(int));
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // pass-1
    maxReductionKernel<<<blocks, threads>>>(d_arr, d_partial, n);
    cudaMemcpy(pass_1_result, d_partial, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaMalloc((void**) & d_arr, blocks * sizeof(int));

    // pass-2.
    maxReductionKernel<<<1, blocks>>>(d_partial, d_arr, blocks);
    cudaMemcpy(pass_2_result, d_arr, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Maximum element is: %d\n", pass_2_result[0]);
    
    cudaFree(d_arr);    
  
  
    // Sum operation.
    int* sum_arr;
    int sum_result_1[blocks];
    int sum_result_2[1];

    cudaMalloc((void**) & d_arr, size);
    cudaMalloc((void**) & sum_arr, blocks * sizeof(int));
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // pass-1.
    sumReductionKernel<<<blocks, threads>>>(d_arr, sum_arr, n);
    cudaMemcpy(sum_result_1, sum_arr, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // pass-2.
    cudaFree(d_arr);
    cudaMalloc((void**) & d_arr, sizeof(int));
    sumReductionKernel<<<1, blocks>>>(sum_arr, d_arr, blocks);
    cudaMemcpy(sum_result_2, d_arr, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Total Sum: %d\n", sum_result_2[0]);

    cudaFree(d_arr);
    cudaFree(sum_arr);




    


}
 