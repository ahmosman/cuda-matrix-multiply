/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

/**
 * Changes made to the original code:
 * - Added a parameter to specify how many results each thread computes.
 * - Implemented a templated kernel that can handle different numbers of results per thread.
 * - Adjusted the grid configuration to account for the number of results per thread.
 * - Added random matrix generation instead of constant values.
 * - Added CPU matrix multiplication for validation.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// Universal version of matrix multiplication kernel
// RESULTS_PER_THREAD is a parameter specifying how many results each thread computes
template <int BLOCK_SIZE, int RESULTS_PER_THREAD>
__global__ void MatrixMulKernel(float *C, float *A, float *B, int wA, int wB)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * BLOCK_SIZE + ty;
  int col = (bx * BLOCK_SIZE + tx) * RESULTS_PER_THREAD; // Each thread calculates RESULTS_PER_THREAD elements horizontally

  // Array for results in thread registers
  float Csub[RESULTS_PER_THREAD] = {0.0f};

  // Loop over all tiles of matrices A and B
  for (int m = 0; m < wA / BLOCK_SIZE; ++m)
  {
    // Shared memory declarations
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE * RESULTS_PER_THREAD]; // x RESULTS_PER_THREAD because each thread needs that many columns

    // Indices for matrix A elements
    int aRow = row;
    int aCol = m * BLOCK_SIZE + tx;

    // Matrix B row index
    int bRow = m * BLOCK_SIZE + ty;

    // Load matrix A data to shared memory
    As[ty][tx] = A[aRow * wA + aCol];

// Load matrix B data to shared memory
// Each thread loads RESULTS_PER_THREAD elements
#pragma unroll
    for (int i = 0; i < RESULTS_PER_THREAD; i++)
    {
      int bCol = col + i;
      Bs[ty][tx * RESULTS_PER_THREAD + i] = B[bRow * wB + bCol];
    }

    __syncthreads();

// Matrix multiplication
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k)
    {
      float aElement = As[ty][k];
#pragma unroll
      for (int i = 0; i < RESULTS_PER_THREAD; i++)
      {
        Csub[i] += aElement * Bs[k][tx * RESULTS_PER_THREAD + i];
      }
    }

    __syncthreads();
  }

// Save results to global memory
#pragma unroll
  for (int i = 0; i < RESULTS_PER_THREAD; i++)
  {
    C[row * wB + col + i] = Csub[i];
  }
}

// Initialize array with random values between 0 and 10000 with 2 decimal places
void RandomInit(float *data, int size)
{
  for (int i = 0; i < size; ++i)
  {
    // Generate random value between 0 and 500 with 2 decimal places
    data[i] = static_cast<float>(rand() % 50000) / 100.0f;
  }
}

// CPU matrix multiplication using ikj loop order for best cache performance
void MatrixMulCPU(float *C, const float *A, const float *B, int hA, int wA, int wB)
{
  // Initialize C to zeros
  for (int i = 0; i < hA * wB; ++i)
    C[i] = 0.0f;

  // Matrix multiplication with ikj loop order
  for (int i = 0; i < hA; ++i)
  {
    for (int k = 0; k < wA; ++k)
    {
      float temp = A[i * wA + k];
      for (int j = 0; j < wB; ++j)
      {
        C[i * wB + j] += temp * B[k * wB + j];
      }
    }
  }
}

// Function testing matrix multiplication for a specific number of results per thread
template <int RESULTS_PER_THREAD>
bool RunMatrixMultiplyTest(int block_size, float *h_A, float *h_B, float *h_C_cpu,
                           const dim3 &dimsA, const dim3 &dimsB)
{
  printf("\n-------------------------------------------------\n");
  printf("Testing matrix multiplication with %d results per thread:\n", RESULTS_PER_THREAD);
  printf("-------------------------------------------------\n");

  // Calculate sizes
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  cudaStream_t stream;

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL)
  {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    return false;
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // copy host memory to device
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

  // Initialize device memory for result to zeros
  checkCudaErrors(cudaMemset(d_C, 0, mem_size_C));

  // Setup execution parameters
  dim3 threads(block_size, block_size);

  // Grid configuration dependent on the number of results per thread
  dim3 grid((dimsB.x + block_size * RESULTS_PER_THREAD - 1) / (block_size * RESULTS_PER_THREAD),
            (dimsA.y + block_size - 1) / block_size);

  printf("Grid configuration: [%d x %d], threads/block: %d\n", grid.x, grid.y, threads.x * threads.y);
  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using MatrixMul CUDA kernel
  if (block_size == 16)
  {
    MatrixMulKernel<16, RESULTS_PER_THREAD><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }
  else
  {
    MatrixMulKernel<32, RESULTS_PER_THREAD><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }

  printf("Warmup completed\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  // Execute the kernel
  int nIter = 10; // Increased for more accurate timing
  printf("Executing %d iterations...\n", nIter);

  for (int j = 0; j < nIter; j++)
  {
    if (block_size == 16)
    {
      MatrixMulKernel<16, RESULTS_PER_THREAD><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else
    {
      MatrixMulKernel<32, RESULTS_PER_THREAD><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance for %d results per thread = %.2f GFlop/s, Time = %.3f ms, Operations = %.0f, Threads/block = %u\n",
      RESULTS_PER_THREAD, gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking results correctness for %d results per thread: ", RESULTS_PER_THREAD);
  bool correct = true;

  // test relative error by comparing with CPU result
  double eps = 1.e-4; // machine zero
  int errorCount = 0;
  const int MAX_ERRORS_TO_PRINT = 10;

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++)
  {
    double abs_err = fabs(h_C[i] - h_C_cpu[i]);
    double abs_val = fabs(h_C_cpu[i]);
    double rel_err = abs_err / (abs_val > 1e-10 ? abs_val : 1.0);

    if (rel_err > eps)
    {
      if (errorCount < MAX_ERRORS_TO_PRINT)
      {
        printf("\nError! Matrix[%05d]=%.8f, CPU Result=%.8f, relative diff > %E",
               i, h_C[i], h_C_cpu[i], eps);
      }
      errorCount++;
      correct = false;
    }
  }

  if (errorCount > MAX_ERRORS_TO_PRINT)
  {
    printf("\n...and %d more errors", errorCount - MAX_ERRORS_TO_PRINT);
  }

  printf("\nResult for %d results per thread = %s\n", RESULTS_PER_THREAD,
         correct ? "CORRECT" : "INCORRECT");

  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return correct;
}

/**
 * Program main - runs tests for different configurations
 */
int main(int argc, char **argv)
{
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  // Seed the random number generator
  srand(time(NULL));

  // This will pick the best possible CUDA capable device
  int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = 32;

  dim3 dimsA(50 * 2 * block_size, 50 * 2 * block_size, 1);
  dim3 dimsB(50 * 2 * block_size, 50 * 2 * block_size, 1);

  if (dimsA.x != dimsB.y)
  {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));

  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

  // Initialize matrices with random values
  printf("Initializing matrices with random values...\n");
  RandomInit(h_A, size_A);
  RandomInit(h_B, size_B);

  // Calculate reference CPU result
  printf("Calculating reference CPU result... (this may take a while)\n");
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C_cpu;
  checkCudaErrors(cudaMallocHost(&h_C_cpu, mem_size_C));

  // Calculate reference result on CPU
  MatrixMulCPU(h_C_cpu, h_A, h_B, dimsA.y, dimsA.x, dimsB.x);
  printf("CPU calculation complete.\n");

  checkCudaErrors(cudaProfilerStart());

  // Run tests for different numbers of results per thread
  bool result1 = RunMatrixMultiplyTest<1>(block_size, h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result2 = RunMatrixMultiplyTest<2>(block_size, h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result4 = RunMatrixMultiplyTest<4>(block_size, h_A, h_B, h_C_cpu, dimsA, dimsB);
  
  // TODO: Explain why doesn't work with 8 or more results per thread

  // Display summary
  printf("\n== SUMMARY ==\n");
  printf("Test with 1 result per thread: %s\n", result1 ? "CORRECT" : "INCORRECT");
  printf("Test with 2 results per thread: %s\n", result2 ? "CORRECT" : "INCORRECT");
  printf("Test with 4 results per thread: %s\n", result4 ? "CORRECT" : "INCORRECT");

  // Free host memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C_cpu));

  checkCudaErrors(cudaProfilerStop());
  checkCudaErrors(cudaDeviceSynchronize());

  if (result1 && result2 && result4 && result8)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}