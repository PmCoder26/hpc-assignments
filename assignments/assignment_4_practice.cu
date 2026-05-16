#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include<cuda_runtime.h>

using namespace std;
using namespace chrono;

// Sequential Matrix Multiplication
void matrixMultiSequential(const vector<vector<int>>& A,
                           const vector<vector<int>>& B,
                           vector<vector<int>>& C, int N) {

    for(int i = 0; i < N; i++) {

        for(int j = 0; j < N; j++) {

            int sum = 0;

            for(int k = 0; k < N; k++) {

                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }
}

__global__ void matrixMultiGlobal(int* A, int* B, int* C, int N) {

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if(row < N && col < N) {

    int sum = 0;

    for(int k = 0; k < N; k++) {

      sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
  }
}

int main() {

    int sizes[4] = { 1000, 1100, 1200, 1300 };

    for(int N : sizes) {

      // Dynamic matrices
      vector<vector<int>> A(N, vector<int>(N));
      vector<vector<int>> B(N, vector<int>(N));
      vector<vector<int>> C(N, vector<int>(N));

      // Fill random values
      for(int i = 0; i < N; i++) {

          for(int j = 0; j < N; j++) {

              A[i][j] = rand() % 100;
              B[i][j] = rand() % 100;
          }
      }

      auto start = high_resolution_clock::now();

      matrixMultiSequential(A, B, C, N);

      auto stop = high_resolution_clock::now();

      auto duration =
          duration_cast<milliseconds>(stop - start);

      cout << "N=" << N << " Sequential Time (ms): "
          << duration.count() << endl;

      vector<int> h_A(N * N);
      vector<int> h_B(N * N);    

      for(int i = 0; i < N; i++) {

        for(int j = 0; j < N; j++) {

          h_A[i * N + j] = A[i][j];
          h_B[i * N + j] = B[i][j];
        }
      }

      dim3 blocks((N + 15) / 16, (N + 15) / 16);
      dim3 threads(16, 16);

      int *d_A, *d_B, *d_C;

      cudaMalloc((void**) & d_A, N * N * sizeof(int));
      cudaMalloc((void**) & d_B, N * N * sizeof(int));
      cudaMalloc((void**) & d_C, N * N * sizeof(int));

      cudaMemcpy(d_A, h_A.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

      start = high_resolution_clock::now();
      matrixMultiGlobal<<<blocks, threads>>>(d_A, d_B, d_C, N);
      
      cudaDeviceSynchronize();
      stop = high_resolution_clock::now();

      duration = duration_cast<milliseconds>(stop - start);

      cout << "N=" << N << " Parallel Time (ms): " << duration.count() << endl;

      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);

    }



    return 0;
}