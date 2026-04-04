#include <iostream>

// ==========================================================
// CUDA KERNEL: Runs on GPU
// __global__ means this function is executed on GPU
// and called from CPU (host)
// ==========================================================
__global__ void hello() {

  // threadIdx.x → index of thread within a block
  // Each thread executes this line independently
  printf("Hello from GPU thread: %d\n", threadIdx.x);
}


// ==========================================================
// KERNEL: Demonstrates thread indexing in CUDA
// ==========================================================
__global__ void printId() {

  // Global thread ID calculation:
  // blockIdx.x → which block
  // blockDim.x → threads per block
  // threadIdx.x → thread inside block

  int globalId = blockIdx.x * blockDim.x + threadIdx.x;

  // Printing different thread information:
  printf(
    "Block ID: %d, BlockDim: %d, Local Thread ID: %d, Global Thread ID: %d\n",
    blockIdx.x, blockDim.x, threadIdx.x, globalId
  );
}


// ==========================================================
// KERNEL: Accessing array stored in GPU memory
// ==========================================================
__global__ void printArray(int *arr) {

  // Calculate global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // ⚠️ IMPORTANT:
  // No boundary check here → may cause out-of-bounds access
  // Always use: if (id < size)

  printf("Element %d = %d\n", id, arr[id]);
}


// ==========================================================
// KERNEL: Parallel array addition
// ==========================================================
__global__ void addArray(int *a, int *b, int *c) {

  // Each thread handles one element
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // ⚠️ IMPORTANT (Best Practice):
  // Always check bounds
  // if (id < size)

  // Perform element-wise addition in parallel
  c[id] = a[id] + b[id];
}


// ==========================================================
// MAIN FUNCTION (CPU / Host code)
// ==========================================================
int main() {

  // --------------------------------------------------------
  // 1. Basic kernel execution
  // --------------------------------------------------------
  // hello<<<1, 5>>>();
  // Launch configuration:
  // <<<blocks, threadsPerBlock>>>
  // 1 block, 5 threads → total 5 threads execute kernel

  // --------------------------------------------------------
  // 2. Understanding thread indexing
  // --------------------------------------------------------
  // printId<<<3, 5>>>();
  // 3 blocks × 5 threads = 15 threads
  // Each thread has:
  // - blockIdx.x (block number)
  // - threadIdx.x (thread number in block)
  // - globalId (unique ID across all threads)

  // --------------------------------------------------------
  // 3. Array handling in CUDA
  // --------------------------------------------------------

  // Host array (CPU memory)
  // int arr[5] = {10, 20, 30, 40, 50};

  // Device pointer (GPU memory)
  // int *d_arr;

  // Allocate memory on GPU
  // cudaMalloc((void**) &d_arr, 5 * sizeof(int));

  // Copy data from CPU → GPU
  // cudaMemcpy(d_arr, arr, 5 * sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel to print array elements
  // printArray<<<1, 5>>>(d_arr);

  // --------------------------------------------------------
  // 4. Parallel array addition
  // --------------------------------------------------------

  // int a[5] = {1, 2, 3, 4, 5};
  // int b[5] = {10, 20, 30, 40, 50};
  // int c[5];  // result array (host)

  // Device memory pointers
  // int *d_a, *d_b, *d_c;

  // Allocate memory on GPU
  // cudaMalloc((void**) &d_a, 5 * sizeof(int));
  // cudaMalloc((void**) &d_b, 5 * sizeof(int));
  // cudaMalloc((void**) &d_c, 5 * sizeof(int));

  // Copy input arrays to GPU
  // cudaMemcpy(d_a, a, 5 * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_b, b, 5 * sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel for parallel addition
  // addArray<<<1, 5>>>(d_a, d_b, d_c);

  // Copy result back from GPU → CPU
  // cudaMemcpy(c, d_c, 5 * sizeof(int), cudaMemcpyDeviceToHost);

  // Print result on CPU
  // for(int x = 0; x < 5; x++) {
  //   std::cout << c[x] << " ";
  // }

  // --------------------------------------------------------
  // 5. Synchronization
  // --------------------------------------------------------

  // cudaDeviceSynchronize()
  // Ensures all GPU operations are completed before CPU continues
  // CUDA kernels run asynchronously (non-blocking)

  cudaDeviceSynchronize();

  return 0;
}