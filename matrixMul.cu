// System includes
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// File paths for cached data
const char *MATRIX_A_FILE = "matrix_A.bin";
const char *MATRIX_B_FILE = "matrix_B.bin";
const char *MATRIX_C_REF_FILE = "matrix_C_ref.bin";

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

// Function to save matrix data to a file
bool SaveMatrixToFile(const char *filename, float *data, size_t size)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    printf("Failed to open file %s for writing\n", filename);
    return false;
  }

  file.write(reinterpret_cast<char *>(data), size * sizeof(float));
  file.close();
  printf("Matrix saved to %s\n", filename);
  return true;
}

// Function to load matrix data from a file
bool LoadMatrixFromFile(const char *filename, float *data, size_t size)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    printf("File %s not found\n", filename);
    return false;
  }

  file.read(reinterpret_cast<char *>(data), size * sizeof(float));
  if (!file)
  {
    printf("Error reading from file %s\n", filename);
    file.close();
    return false;
  }

  file.close();
  printf("Matrix loaded from %s\n", filename);
  return true;
}

// Special version of matrix multiplication kernel for 4 results per thread in 2x2 pattern
template <int BLOCK_SIZE>
__global__ void MatrixMulKernel2x2(float *C, float *A, float *B, int wA, int wB)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Each thread computes a 2x2 block of results
  // Calculate top-left corner of the 2x2 block this thread computes
  int row = by * BLOCK_SIZE * 2 + ty * 2;
  int col = bx * BLOCK_SIZE * 2 + tx * 2;

  // Each thread accumulates results for 4 elements in a 2x2 block
  float Csub00 = 0.0f; // top-left element [row][col]
  float Csub01 = 0.0f; // top-right element [row][col+1]
  float Csub10 = 0.0f; // bottom-left element [row+1][col]
  float Csub11 = 0.0f; // bottom-right element [row+1][col+1]

  // Loop over all tiles of matrices A and B
  for (int m = 0; m < (wA / BLOCK_SIZE); ++m)
  {
    // Shared memory declarations
    __shared__ float As[BLOCK_SIZE * 2][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE * 2];

    // Each thread loads two elements fA (row and row+1)or matrix
    As[ty * 2][tx] = A[row * wA + (m * BLOCK_SIZE + tx)];
    As[ty * 2 + 1][tx] = A[(row + 1) * wA + (m * BLOCK_SIZE + tx)];

    // Each thread loads two elements for matrix B (col and col+1)
    Bs[ty][tx * 2] = B[(m * BLOCK_SIZE + ty) * wB + col];
    Bs[ty][tx * 2 + 1] = B[(m * BLOCK_SIZE + ty) * wB + (col + 1)];

    __syncthreads();

// Compute partial dot products for the 2x2 block
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k)
    {
      float a_row = As[ty * 2][k];
      float a_row1 = As[ty * 2 + 1][k];

      float b_col = Bs[k][tx * 2];
      float b_col1 = Bs[k][tx * 2 + 1];

      Csub00 += a_row * b_col;   // C[row][col]
      Csub01 += a_row * b_col1;  // C[row][col+1]
      Csub10 += a_row1 * b_col;  // C[row+1][col]
      Csub11 += a_row1 * b_col1; // C[row+1][col+1]
    }

    __syncthreads();
  }

  // Write the 2x2 block results to global memory
  C[row * wB + col] = Csub00;
  C[row * wB + (col + 1)] = Csub01;
  C[(row + 1) * wB + col] = Csub10;
  C[(row + 1) * wB + (col + 1)] = Csub11;
}

// Modified function to run the 2x2 pattern for 4 results per thread
bool RunMatrixMultiply4ResultsTest(int block_size, float *h_A, float *h_B, float *h_C_cpu,
                                   const dim3 &dimsA, const dim3 &dimsB)
{
  printf("\n-------------------------------------------------\n");
  printf("Testing matrix multiplication with 4 results per thread (2x2 pattern):\n");
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
  dim3 threads(block_size / 2, block_size / 2); // Half the threads since each handles 4 elements

  // Grid configuration for 2x2 results per thread
  dim3 grid(dimsB.x / (block_size), dimsA.y / (block_size));

  printf("Grid configuration: [%d x %d], threads/block: %d\n", grid.x, grid.y, threads.x * threads.y);
  printf("Computing result using CUDA Kernel with 2x2 pattern...\n");

  // Performs warmup operation
  if (block_size == 16)
  {
    MatrixMulKernel2x2<8><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }
  else
  {
    MatrixMulKernel2x2<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }

  printf("Warmup completed\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  // Execute the kernel
  printf("Executing kernel...\n");

  if (block_size == 16)
  {
    MatrixMulKernel2x2<8><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }
  else
  {
    MatrixMulKernel2x2<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance for 2x2 pattern = %.2f GFlop/s, Time = %.3f ms, Operations = %.0f, Threads/block = %u\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking results correctness for 2x2 pattern: ");
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

  printf("\nResult for 2x2 pattern = %s\n", correct ? "CORRECT" : "INCORRECT");

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

  // Calculate reference CPU result
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C_cpu;
  checkCudaErrors(cudaMallocHost(&h_C_cpu, mem_size_C));

  // Try to load matrices from files first
  bool filesLoaded = true;

  // Try to load matrix A
  if (!LoadMatrixFromFile(MATRIX_A_FILE, h_A, size_A))
  {
    filesLoaded = false;
  }

  // Try to load matrix B
  if (!LoadMatrixFromFile(MATRIX_B_FILE, h_B, size_B))
  {
    filesLoaded = false;
  }

  // Try to load reference result
  if (!LoadMatrixFromFile(MATRIX_C_REF_FILE, h_C_cpu, dimsC.x * dimsC.y))
  {
    filesLoaded = false;
  }

  // If any file wasn't loaded successfully, generate new matrices and calculate reference
  if (!filesLoaded)
  {
    printf("Files not found or incomplete - generating new random matrices...\n");

    // Initialize matrices with random values
    RandomInit(h_A, size_A);
    RandomInit(h_B, size_B);

    // Save the generated matrices
    SaveMatrixToFile(MATRIX_A_FILE, h_A, size_A);
    SaveMatrixToFile(MATRIX_B_FILE, h_B, size_B);

    // Calculate reference CPU result
    printf("Calculating reference CPU result... (this may take a while)\n");

    // Calculate reference result on CPU
    MatrixMulCPU(h_C_cpu, h_A, h_B, dimsA.y, dimsA.x, dimsB.x);
    printf("CPU calculation complete.\n");

    // Save reference result
    SaveMatrixToFile(MATRIX_C_REF_FILE, h_C_cpu, dimsC.x * dimsC.y);
  }
  else
  {
    printf("All matrices loaded from files successfully.\n");
  }

  checkCudaErrors(cudaProfilerStart());

  // Run tests for different numbers of results per thread
  bool result4 = RunMatrixMultiply4ResultsTest(block_size, h_A, h_B, h_C_cpu, dimsA, dimsB);

  // Display summary
  printf("\n== SUMMARY ==\n");
  printf("Test with 4 results per thread: %s\n", result4 ? "CORRECT" : "INCORRECT");

  // Free host memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C_cpu));

  checkCudaErrors(cudaProfilerStop());
  checkCudaErrors(cudaDeviceSynchronize());

  if (result4)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}