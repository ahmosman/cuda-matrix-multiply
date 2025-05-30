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

#define BLOCK_SIZE 32

template <int RESULTS_PER_THREAD_X = 1, int RESULTS_PER_THREAD_Y = 1>
__global__ void MatrixMulCUDA(float *A, float *B, float *C, int size)
{
  int baseRow = (blockIdx.y * blockDim.y + threadIdx.y) * RESULTS_PER_THREAD_Y;
  int baseCol = (blockIdx.x * blockDim.x + threadIdx.x) * RESULTS_PER_THREAD_X;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Define shared memory arrays
  __shared__ float As[BLOCK_SIZE * RESULTS_PER_THREAD_Y][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE * RESULTS_PER_THREAD_X];

  // Result accumulators - we'll use a 2D array for clarity
  float sum[RESULTS_PER_THREAD_Y][RESULTS_PER_THREAD_X] = {0.0f};

  int numTiles = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int m = 0; m < numTiles; ++m)
  {
    // Load A tile to shared memory
    for (int y = 0; y < RESULTS_PER_THREAD_Y; ++y)
    {
      int row = baseRow + y;
      if (row < size && (m * BLOCK_SIZE + tx) < size)
      {
        As[ty * RESULTS_PER_THREAD_Y + y][tx] = A[row * size + m * BLOCK_SIZE + tx];
      }
      else
      {
        As[ty * RESULTS_PER_THREAD_Y + y][tx] = 0.0f;
      }
    }

    // Load B tile to shared memory
    for (int x = 0; x < RESULTS_PER_THREAD_X; ++x)
    {
      int col = baseCol + x;
      if ((m * BLOCK_SIZE + ty) < size && col < size)
      {
        Bs[ty][tx * RESULTS_PER_THREAD_X + x] = B[(m * BLOCK_SIZE + ty) * size + col];
      }
      else
      {
        Bs[ty][tx * RESULTS_PER_THREAD_X + x] = 0.0f;
      }
    }

    __syncthreads();

    // Compute partial dot products
    for (int k = 0; k < BLOCK_SIZE; ++k)
    {
      for (int y = 0; y < RESULTS_PER_THREAD_Y; ++y)
      {
        float a_val = As[ty * RESULTS_PER_THREAD_Y + y][k];

        for (int x = 0; x < RESULTS_PER_THREAD_X; ++x)
        {
          float b_val = Bs[k][tx * RESULTS_PER_THREAD_X + x];
          sum[y][x] += a_val * b_val;
        }
      }
    }

    __syncthreads();
  }

  // Write results
  for (int y = 0; y < RESULTS_PER_THREAD_Y; ++y)
  {
    int row = baseRow + y;
    if (row < size)
    {
      for (int x = 0; x < RESULTS_PER_THREAD_X; ++x)
      {
        int col = baseCol + x;
        if (col < size)
        {
          C[row * size + col] = sum[y][x];
        }
      }
    }
  }
}

// ================ HELPERS ================

void randomInit(float *data, int size)
{
  for (int i = 0; i < size; ++i)
    data[i] = static_cast<float>(rand() % 50000) / 100.0f;
}

void matrixMulCPU(float *C, const float *A, const float *B, int hA, int wA, int wB)
{
  for (int i = 0; i < hA * wB; ++i)
    C[i] = 0.0f;
  for (int i = 0; i < hA; ++i)
    for (int k = 0; k < wA; ++k)
    {
      float temp = A[i * wA + k];
      for (int j = 0; j < wB; ++j)
        C[i * wB + j] += temp * B[k * wB + j];
    }
}

bool saveMatrixToFile(const char *filename, float *data, size_t size)
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

bool loadMatrixFromFile(const char *filename, float *data, size_t size)
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

bool verifyResults(float *gpu_result, float *cpu_reference, int size)
{
  const double epsilon = 1.e-4;
  int errorCount = 0, maxPrint = 10;
  bool correct = true;
  for (int i = 0; i < size; i++)
  {
    double abs_err = fabs(gpu_result[i] - cpu_reference[i]);
    double abs_val = fabs(cpu_reference[i]);
    double rel_err = abs_err / (abs_val > 1e-10 ? abs_val : 1.0);
    if (rel_err > epsilon)
    {
      if (errorCount < maxPrint)
        printf("\nError! Element[%05d]=%.8f, CPU Result=%.8f, relative diff > %E", i, gpu_result[i], cpu_reference[i], epsilon);
      errorCount++;
      correct = false;
    }
  }
  if (errorCount > maxPrint)
    printf("\n...and %d more errors", errorCount - maxPrint);
  return correct;
}

bool prepareMatrices(float *h_A, float *h_B, float *h_C_cpu, int size)
{
  char MATRIX_A_FILE[64], MATRIX_B_FILE[64], MATRIX_C_REF_FILE[64];
  sprintf(MATRIX_A_FILE, "matrixA_%d.bin", size);
  sprintf(MATRIX_B_FILE, "matrixB_%d.bin", size);
  sprintf(MATRIX_C_REF_FILE, "matrixC_ref_%d.bin", size);

  size_t elements = size * size;
  bool filesLoaded = true;
  if (!loadMatrixFromFile(MATRIX_A_FILE, h_A, elements))
    filesLoaded = false;
  if (!loadMatrixFromFile(MATRIX_B_FILE, h_B, elements))
    filesLoaded = false;
  if (!loadMatrixFromFile(MATRIX_C_REF_FILE, h_C_cpu, elements))
    filesLoaded = false;
  if (!filesLoaded)
  {
    printf("Files not found or incomplete - generating new random matrices...\n");
    randomInit(h_A, elements);
    randomInit(h_B, elements);
    saveMatrixToFile(MATRIX_A_FILE, h_A, elements);
    saveMatrixToFile(MATRIX_B_FILE, h_B, elements);
    printf("Calculating reference result on CPU... (this may take a while)\n");
    matrixMulCPU(h_C_cpu, h_A, h_B, size, size, size);
    printf("CPU calculations completed.\n");
    saveMatrixToFile(MATRIX_C_REF_FILE, h_C_cpu, elements);
  }
  else
  {
    printf("All matrices loaded successfully from files.\n");
  }
  return true;
}

// Function testing matrix multiplication for a specific number of results per thread
template <int RESULTS_PER_THREAD_X = 1, int RESULTS_PER_THREAD_Y = 1>
bool RunMatrixMultiplyTest(float *h_A, float *h_B, float *h_C_cpu,
                           const dim3 &dimsA, const dim3 &dimsB)
{
  printf("\n-------------------------------------------------\n");
  printf("Testing matrix multiplication with %d results per thread (%dx%d)...\n",
         RESULTS_PER_THREAD_X * RESULTS_PER_THREAD_Y, RESULTS_PER_THREAD_X, RESULTS_PER_THREAD_Y);
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
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  // Grid configuration dependent on the number of results per thread
  dim3 grid(
      (dimsB.x + BLOCK_SIZE * RESULTS_PER_THREAD_X - 1) / (BLOCK_SIZE * RESULTS_PER_THREAD_X),
      (dimsA.y + BLOCK_SIZE * RESULTS_PER_THREAD_Y - 1) / (BLOCK_SIZE * RESULTS_PER_THREAD_Y));

  printf("Grid configuration: [%d x %d], threads/block: %d\n", grid.x, grid.y, threads.x * threads.y);
  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using MatrixMul CUDA kernel

  MatrixMulCUDA<RESULTS_PER_THREAD_X, RESULTS_PER_THREAD_Y><<<grid, threads, 0, stream>>>(d_A, d_B, d_C, dimsA.x);

  printf("Warmup completed\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  MatrixMulCUDA<RESULTS_PER_THREAD_X, RESULTS_PER_THREAD_Y><<<grid, threads, 0, stream>>>(d_A, d_B, d_C, dimsA.x);

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  int nIter = 1; // Number of iterations for performance measurement
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance for %d results per thread = %.2f GFlop/s, Time = %.3f ms, Operations = %.0f, Threads/block = %u\n",
      RESULTS_PER_THREAD_X * RESULTS_PER_THREAD_Y, gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking results correctness for %d results per threads...\n",
         RESULTS_PER_THREAD_X * RESULTS_PER_THREAD_Y);
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

  printf("\nResult for %d results per threads %dx%d: %s\n",
         RESULTS_PER_THREAD_X * RESULTS_PER_THREAD_Y, RESULTS_PER_THREAD_X, RESULTS_PER_THREAD_Y,
         correct ? "PASS" : "FAIL");

  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return correct;
}

// ================= MAIN =================

int main(int argc, char **argv)
{
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  // Seed the random number generator
  srand(time(NULL));

  // This will pick the best possible CUDA capable device
  int dev = findCudaDevice(argc, (const char **)argv);

  dim3 dimsA(50 * 2 * BLOCK_SIZE, 50 * 2 * BLOCK_SIZE, 1);
  dim3 dimsB(50 * 2 * BLOCK_SIZE, 50 * 2 * BLOCK_SIZE, 1);

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

  prepareMatrices(h_A, h_B, h_C_cpu, dimsA.x);

  checkCudaErrors(cudaProfilerStart());

  // Run tests for different numbers of results per thread
  bool result_1x1 = RunMatrixMultiplyTest<1, 1>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_2x1 = RunMatrixMultiplyTest<2, 1>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_1x2 = RunMatrixMultiplyTest<1, 2>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_2x2 = RunMatrixMultiplyTest<2, 2>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_2x3 = RunMatrixMultiplyTest<2, 3>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_2x4 = RunMatrixMultiplyTest<2, 4>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_3x3 = RunMatrixMultiplyTest<3, 3>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_4x4 = RunMatrixMultiplyTest<4, 4>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_5x6 = RunMatrixMultiplyTest<5, 6>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  bool result_6x6 = RunMatrixMultiplyTest<6, 6>(h_A, h_B, h_C_cpu, dimsA, dimsB);

  /*bool result_8x8;
  bool result_16x16;
  bool result_23x23;
  bool result_23x24;
  bool result_24x23;
  bool result_24x24;
*/	
  // for 16x16 block size, we can also test these configurations:
  /*if (BLOCK_SIZE == 16)
  {

    result_8x8 = RunMatrixMultiplyTest<8, 8>(h_A, h_B, h_C_cpu, dimsA, dimsB);
    result_16x16 = RunMatrixMultiplyTest<16, 16>(h_A, h_B, h_C_cpu, dimsA, dimsB);
    result_23x23 = RunMatrixMultiplyTest<23, 23>(h_A, h_B, h_C_cpu, dimsA, dimsB);
    result_23x24 = RunMatrixMultiplyTest<23, 24>(h_A, h_B, h_C_cpu, dimsA, dimsB);
    result_24x23 = RunMatrixMultiplyTest<24, 23>(h_A, h_B, h_C_cpu, dimsA, dimsB);
    result_24x24 = RunMatrixMultiplyTest<24, 24>(h_A, h_B, h_C_cpu, dimsA, dimsB);
  }*/

  // Display summary
  printf("\n== SUMMARY ==\n");
  printf(" 1x1 results per thread: %s\n", result_1x1 ? "PASS" : "FAIL");
  //printf(" 2x1 results per thread: %s\n", result_2x1 ? "PASS" : "FAIL");
  //printf(" 1x2 results per thread: %s\n", result_1x2 ? "PASS" : "FAIL");
  //printf(" 2x2 results per thread: %s\n", result_2x2 ? "PASS" : "FAIL");
  //printf(" 2x3 results per thread: %s\n", result_2x3 ? "PASS" : "FAIL");
  //printf(" 2x4 results per thread: %s\n", result_2x4 ? "PASS" : "FAIL");
  //printf(" 3x3 results per thread: %s\n", result_3x3 ? "PASS" : "FAIL");
  //printf(" 4x4 results per thread: %s\n", result_4x4 ? "PASS" : "FAIL");
  //printf(" 5x6 results per thread: %s\n", result_5x6 ? "PASS" : "FAIL");
  //printf(" 6x6 results per thread: %s\n", result_6x6 ? "PASS" : "FAIL");

  /*if (BLOCK_SIZE == 16)
  {
    printf(" 8x8 results per thread: %s\n", result_8x8 ? "PASS" : "FAIL");
    printf("16x16 results per thread: %s\n", result_16x16 ? "PASS" : "FAIL");
    printf("23x23 results per thread: %s\n", result_23x23 ? "PASS" : "FAIL");
    printf("23x24 results per thread: %s\n", result_23x24 ? "PASS" : "FAIL");
    printf("24x23 results per thread: %s\n", result_24x23 ? "PASS" : "FAIL");
    printf("24x24 results per thread: %s\n", result_24x24 ? "PASS" : "FAIL");
  }*/

  // Free host memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C_cpu));

  checkCudaErrors(cudaProfilerStop());
  checkCudaErrors(cudaDeviceSynchronize());

  return 0;
}
