#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


#define TILE_SIZE 16

template <int BLOCK_SIZE> __global__ void MatrixMulMultiOut(float *A, float *B, float *C, int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x * 2; // 2 wyniki na wątek
 
    float Cvalue0 = 0.0f, Cvalue1 = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; ++t)
    {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        Bs[threadIdx.y][threadIdx.x + 1] = B[(t * TILE_SIZE + threadIdx.y) * N + col + 1];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            float a = As[threadIdx.y][k];
            Cvalue0 += a * Bs[k][threadIdx.x];
            Cvalue1 += a * Bs[k][threadIdx.x + 1];
        }
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = Cvalue0;
    if (row < N && col + 1 < N)
        C[row * N + col + 1] = Cvalue1;
}

int main() {
    return 0;
}