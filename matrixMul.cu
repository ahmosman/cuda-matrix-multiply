#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <iostream>

// Define tile sizes
#define TILE_WIDTH_16 16
#define TILE_WIDTH_32 32

// Structure to store test results
typedef struct {
  float time_ms;
  float gflops;
  bool correct;
  const char *name;
} TestResult;

// ============================================================================
// CUDA KERNELS
// ============================================================================

// Template kernel for 1 result per thread
template <int TILE_WIDTH>
__global__ void matrixMul1Result(float *A, float *B, float *C, int size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  float sum = 0.0f;

  for (int m = 0; m < ceil(size / (float)TILE_WIDTH); ++m) {
    // Load data to shared memory
    if (row < size && (m * TILE_WIDTH + tx) < size) {
      As[ty][tx] = A[row * size + m * TILE_WIDTH + tx];
    } else {
      As[ty][tx] = 0.0f;
    }

    if ((m * TILE_WIDTH + ty) < size && col < size) {
      Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * size + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot product
    for (int k = 0; k < TILE_WIDTH; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  // Write result
  if (row < size && col < size) {
    C[row * size + col] = sum;
  }
}

// Template kernel for 2 results per thread
template <int TILE_WIDTH>
__global__ void matrixMul2Results(float *A, float *B, float *C, int size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int baseCol = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH * 2];

  float sum0 = 0.0f, sum1 = 0.0f;

  for (int m = 0; m < ceil(size / (float)TILE_WIDTH); ++m) {
    // Load data to shared memory
    if (row < size && (m * TILE_WIDTH + tx) < size) {
      As[ty][tx] = A[row * size + m * TILE_WIDTH + tx];
    } else {
      As[ty][tx] = 0.0f;
    }

    if ((m * TILE_WIDTH + ty) < size && baseCol < size) {
      Bs[ty][tx * 2] = B[(m * TILE_WIDTH + ty) * size + baseCol];
      if (baseCol + 1 < size) {
        Bs[ty][tx * 2 + 1] = B[(m * TILE_WIDTH + ty) * size + baseCol + 1];
      } else {
        Bs[ty][tx * 2 + 1] = 0.0f;
      }
    } else {
      Bs[ty][tx * 2] = 0.0f;
      Bs[ty][tx * 2 + 1] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot products
    for (int k = 0; k < TILE_WIDTH; ++k) {
      float a_val = As[ty][k];
      sum0 += a_val * Bs[k][tx * 2];
      sum1 += a_val * Bs[k][tx * 2 + 1];
    }

    __syncthreads();
  }

  // Write results
  if (row < size && baseCol < size) {
    C[row * size + baseCol] = sum0;
  }
  if (row < size && (baseCol + 1) < size) {
    C[row * size + (baseCol + 1)] = sum1;
  }
}

// Template kernel for 4 results per thread
template <int TILE_WIDTH>
__global__ void matrixMul4Results(float *A, float *B, float *C, int size) {
  int baseRow = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  int baseCol = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float As[TILE_WIDTH * 2][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH * 2];

  float sum00 = 0.0f, sum01 = 0.0f;
  float sum10 = 0.0f, sum11 = 0.0f;

  for (int m = 0; m < ceil(size / (float)TILE_WIDTH); ++m) {
    // Load data to shared memory
    if (baseRow < size && (m * TILE_WIDTH + tx) < size) {
      As[ty * 2][tx] = A[baseRow * size + m * TILE_WIDTH + tx];
      if (baseRow + 1 < size) {
        As[ty * 2 + 1][tx] = A[(baseRow + 1) * size + m * TILE_WIDTH + tx];
      } else {
        As[ty * 2 + 1][tx] = 0.0f;
      }
    } else {
      As[ty * 2][tx] = 0.0f;
      As[ty * 2 + 1][tx] = 0.0f;
    }

    if ((m * TILE_WIDTH + ty) < size && baseCol < size) {
      Bs[ty][tx * 2] = B[(m * TILE_WIDTH + ty) * size + baseCol];
      if (baseCol + 1 < size) {
        Bs[ty][tx * 2 + 1] = B[(m * TILE_WIDTH + ty) * size + baseCol + 1];
      } else {
        Bs[ty][tx * 2 + 1] = 0.0f;
      }
    } else {
      Bs[ty][tx * 2] = 0.0f;
      Bs[ty][tx * 2 + 1] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot products
    for (int k = 0; k < TILE_WIDTH; ++k) {
      float a00 = As[ty * 2][k];
      float a10 = As[ty * 2 + 1][k];
      float b00 = Bs[k][tx * 2];
      float b01 = Bs[k][tx * 2 + 1];

      sum00 += a00 * b00;
      sum01 += a00 * b01;
      sum10 += a10 * b00;
      sum11 += a10 * b01;
    }

    __syncthreads();
  }

  // Write results
  if (baseRow < size && baseCol < size) {
    C[baseRow * size + baseCol] = sum00;
  }
  if (baseRow < size && (baseCol + 1) < size) {
    C[baseRow * size + (baseCol + 1)] = sum01;
  }
  if ((baseRow + 1) < size && baseCol < size) {
    C[(baseRow + 1) * size + baseCol] = sum10;
  }
  if ((baseRow + 1) < size && (baseCol + 1) < size) {
    C[(baseRow + 1) * size + (baseCol + 1)] = sum11;
  }
}

// Template kernel for 6 results per thread
template <int TILE_WIDTH>
__global__ void matrixMul6Results(float *A, float *B, float *C, int size) {
  int baseRow = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  int baseCol = (blockIdx.x * blockDim.x + threadIdx.x) * 3;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float As[TILE_WIDTH * 2][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH * 3];

  float sum00 = 0.0f, sum01 = 0.0f, sum02 = 0.0f;
  float sum10 = 0.0f, sum11 = 0.0f, sum12 = 0.0f;

  for (int m = 0; m < ceil(size / (float)TILE_WIDTH); ++m) {
    // Load data to shared memory
    if (baseRow < size && (m * TILE_WIDTH + tx) < size) {
      As[ty * 2][tx] = A[baseRow * size + m * TILE_WIDTH + tx];
      if (baseRow + 1 < size) {
        As[ty * 2 + 1][tx] = A[(baseRow + 1) * size + m * TILE_WIDTH + tx];
      } else {
        As[ty * 2 + 1][tx] = 0.0f;
      }
    } else {
      As[ty * 2][tx] = 0.0f;
      As[ty * 2 + 1][tx] = 0.0f;
    }

    if ((m * TILE_WIDTH + ty) < size && baseCol < size) {
      Bs[ty][tx * 3] = B[(m * TILE_WIDTH + ty) * size + baseCol];
      if (baseCol + 1 < size) {
        Bs[ty][tx * 3 + 1] = B[(m * TILE_WIDTH + ty) * size + baseCol + 1];
      } else {
        Bs[ty][tx * 3 + 1] = 0.0f;
      }
      if (baseCol + 2 < size) {
        Bs[ty][tx * 3 + 2] = B[(m * TILE_WIDTH + ty) * size + baseCol + 2];
      } else {
        Bs[ty][tx * 3 + 2] = 0.0f;
      }
    } else {
      Bs[ty][tx * 3] = 0.0f;
      Bs[ty][tx * 3 + 1] = 0.0f;
      Bs[ty][tx * 3 + 2] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot products
    for (int k = 0; k < TILE_WIDTH; ++k) {
      float a00 = As[ty * 2][k];
      float a10 = As[ty * 2 + 1][k];

      float b00 = Bs[k][tx * 3];
      float b01 = Bs[k][tx * 3 + 1];
      float b02 = Bs[k][tx * 3 + 2];

      sum00 += a00 * b00;
      sum01 += a00 * b01;
      sum02 += a00 * b02;
      sum10 += a10 * b00;
      sum11 += a10 * b01;
      sum12 += a10 * b02;
    }

    __syncthreads();
  }

  // Write results
  if (baseRow < size && baseCol < size) {
    C[baseRow * size + baseCol] = sum00;
  }
  if (baseRow < size && (baseCol + 1) < size) {
    C[baseRow * size + (baseCol + 1)] = sum01;
  }
  if (baseRow < size && (baseCol + 2) < size) {
    C[baseRow * size + (baseCol + 2)] = sum02;
  }
  if ((baseRow + 1) < size && baseCol < size) {
    C[(baseRow + 1) * size + baseCol] = sum10;
  }
  if ((baseRow + 1) < size && (baseCol + 1) < size) {
    C[(baseRow + 1) * size + (baseCol + 1)] = sum11;
  }
  if ((baseRow + 1) < size && (baseCol + 2) < size) {
    C[(baseRow + 1) * size + (baseCol + 2)] = sum12;
  }
}

// ================ HELPERS ================

const char *MATRIX_A_FILE = "matrix_A.bin";
const char *MATRIX_B_FILE = "matrix_B.bin";
const char *MATRIX_C_REF_FILE = "matrix_C_ref.bin";

void randomInit(float *data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = static_cast<float>(rand() % 50000) / 100.0f;
}

void matrixMulCPU(float *C, const float *A, const float *B, int hA, int wA, int wB) {
    for (int i = 0; i < hA * wB; ++i) C[i] = 0.0f;
    for (int i = 0; i < hA; ++i)
        for (int k = 0; k < wA; ++k) {
            float temp = A[i * wA + k];
            for (int j = 0; j < wB; ++j)
                C[i * wB + j] += temp * B[k * wB + j];
        }
}

bool saveMatrixToFile(const char *filename, float *data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Failed to open file %s for writing\n", filename);
        return false;
    }
    file.write(reinterpret_cast<char *>(data), size * sizeof(float));
    file.close();
    printf("Matrix saved to %s\n", filename);
    return true;
}

bool loadMatrixFromFile(const char *filename, float *data, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("File %s not found\n", filename);
        return false;
    }
    file.read(reinterpret_cast<char *>(data), size * sizeof(float));
    if (!file) {
        printf("Error reading from file %s\n", filename);
        file.close();
        return false;
    }
    file.close();
    printf("Matrix loaded from %s\n", filename);
    return true;
}

bool verifyResults(float *gpu_result, float *cpu_reference, int size) {
    const double epsilon = 1.e-4;
    int errorCount = 0, maxPrint = 10;
    bool correct = true;
    for (int i = 0; i < size; i++) {
        double abs_err = fabs(gpu_result[i] - cpu_reference[i]);
        double abs_val = fabs(cpu_reference[i]);
        double rel_err = abs_err / (abs_val > 1e-10 ? abs_val : 1.0);
        if (rel_err > epsilon) {
            if (errorCount < maxPrint)
                printf("\nError! Element[%05d]=%.8f, CPU Result=%.8f, relative diff > %E", i, gpu_result[i], cpu_reference[i], epsilon);
            errorCount++; correct = false;
        }
    }
    if (errorCount > maxPrint)
        printf("\n...and %d more errors", errorCount - maxPrint);
    return correct;
}

bool prepareMatrices(float *h_A, float *h_B, float *h_C_cpu, int size) {
    size_t elements = size * size;
    bool filesLoaded = true;
    if (!loadMatrixFromFile(MATRIX_A_FILE, h_A, elements)) filesLoaded = false;
    if (!loadMatrixFromFile(MATRIX_B_FILE, h_B, elements)) filesLoaded = false;
    if (!loadMatrixFromFile(MATRIX_C_REF_FILE, h_C_cpu, elements)) filesLoaded = false;
    if (!filesLoaded) {
        printf("Files not found or incomplete - generating new random matrices...\n");
        randomInit(h_A, elements);
        randomInit(h_B, elements);
        saveMatrixToFile(MATRIX_A_FILE, h_A, elements);
        saveMatrixToFile(MATRIX_B_FILE, h_B, elements);
        printf("Calculating reference result on CPU... (this may take a while)\n");
        matrixMulCPU(h_C_cpu, h_A, h_B, size, size, size);
        printf("CPU calculations completed.\n");
        saveMatrixToFile(MATRIX_C_REF_FILE, h_C_cpu, elements);
    } else {
        printf("All matrices loaded successfully from files.\n");
    }
    return true;
}

template <typename KernelFunc>
TestResult runTest(KernelFunc kernel, dim3 gridSize, dim3 blockSize,
                   float *d_A, float *d_B, float *d_C,
                   float *h_C_gpu, float *h_C_cpu,
                   int size, const char *testName) {
    TestResult result; result.name = testName;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.time_ms, start, stop);
    size_t bytes = size * size * sizeof(float);
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    result.correct = verifyResults(h_C_gpu, h_C_cpu, size * size);
    double operations = 2.0 * size * size * size;
    result.gflops = operations / (result.time_ms * 1e6);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return result;
}

// ================= MAIN =================

int main() {
    int size = 4096;
    printf("=== Matrix multiplication test with different results per thread ===\n\n");
    printf("=== Testing for size %dx%d ===\n", size, size);

    size_t bytes = size * size * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);

    if (!prepareMatrices(h_A, h_B, h_C_cpu, size)) {
        printf("An error occurred while preparing matrices!\n");
        return -1;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    TestResult results[8];
    int testCount = 0;

    printf("--- Tests for 16x16 block ---\n");
    {
        dim3 block(TILE_WIDTH_16, TILE_WIDTH_16);
        dim3 grid((size + TILE_WIDTH_16 - 1) / TILE_WIDTH_16, (size + TILE_WIDTH_16 - 1) / TILE_WIDTH_16);
        results[testCount++] = runTest(matrixMul1Result<TILE_WIDTH_16>, grid, block, d_A, d_B, d_C, h_C_gpu, h_C_cpu, size, "1 result/thread (16x16)");
    }
    {
        dim3 block(TILE_WIDTH_16, TILE_WIDTH_16);
        dim3 grid((size + 2 * TILE_WIDTH_16 - 1) / (2 * TILE_WIDTH_16), (size + TILE_WIDTH_16 - 1) / TILE_WIDTH_16);
        results[testCount++] = runTest(matrixMul2Results<TILE_WIDTH_16>, grid, block, d_A, d_B, d_C, h_C_gpu, h_C_cpu, size, "2 results/thread (16x16)");
    }
    {
        dim3 block(TILE_WIDTH_16, TILE_WIDTH_16);
        dim3 grid((size + 2 * TILE_WIDTH_16 - 1) / (2 * TILE_WIDTH_16), (size + 2 * TILE_WIDTH_16 - 1) / (2 * TILE_WIDTH_16));
        results[testCount++] = runTest(matrixMul4Results<TILE_WIDTH_16>, grid, block, d_A, d_B, d_C, h_C_gpu, h_C_cpu, size, "4 results/thread (16x16)");
    }
    {
        dim3 block(TILE_WIDTH_16, TILE_WIDTH_16);
        dim3 grid((size + 3 * TILE_WIDTH_16 - 1) / (3 * TILE_WIDTH_16), (size + 2 * TILE_WIDTH_16 - 1) / (2 * TILE_WIDTH_16));
        results[testCount++] = runTest(matrixMul6Results<TILE_WIDTH_16>, grid, block, d_A, d_B, d_C, h_C_gpu, h_C_cpu, size, "6 results/thread (16x16)");
    }

    printf("\n--- Tests for 32x32 block ---\n");
    {
        dim3 block(TILE_WIDTH_32, TILE_WIDTH_32);
        dim3 grid((size + TILE_WIDTH_32 - 1) / TILE_WIDTH_32, (size + TILE_WIDTH_32 - 1) / TILE_WIDTH_32);
        results[testCount++] = runTest(matrixMul1Result<TILE_WIDTH_32>, grid, block, d_A, d_B, d_C, h_C_gpu, h_C_cpu, size, "1 result/thread (32x32)");
    }
    {
        dim3 block(TILE_WIDTH_32, TILE_WIDTH_32);
        dim3 grid((size + 2 * TILE_WIDTH_32 - 1) / (2 * TILE_WIDTH_32), (size + TILE_WIDTH_32 - 1) / TILE_WIDTH_32);
        results[testCount++] = runTest(matrixMul2Results<TILE_WIDTH_32>, grid, block, d_A, d_B, d_C, h_C_gpu, h_C_cpu, size, "2 results/thread (32x32)");
    }
    {
        dim3 block(TILE_WIDTH_32, TILE_WIDTH_32);
        dim3 grid((size + 2 * TILE_WIDTH_32 - 1) / (2 * TILE_WIDTH_32), (size + 2 * TILE_WIDTH_32 - 1) / (2 * TILE_WIDTH_32));
        results[testCount++] = runTest(matrixMul4Results<TILE_WIDTH_32>, grid, block, d_A, d_B, d_C, h_C_gpu, h_C_cpu, size, "4 results/thread (32x32)");
    }
    {
        dim3 block(TILE_WIDTH_32, TILE_WIDTH_32);
        dim3 grid((size + 3 * TILE_WIDTH_32 - 1) / (3 * TILE_WIDTH_32), (size + 2 * TILE_WIDTH_32 - 1) / (2 * TILE_WIDTH_32));
        results[testCount++] = runTest(matrixMul6Results<TILE_WIDTH_32>, grid, block, d_A, d_B, d_C, h_C_gpu, h_C_cpu, size, "6 results/thread (32x32)");
    }

    printf("\n=== RESULTS SUMMARY ===\n");
    printf("%-25s %10s %10s %10s\n", "Test", "Time [ms]", "GFLOPS", "Status");
    printf("%-25s %10s %10s %10s\n", "----", "--------", "------", "------");
    for (int i = 0; i < testCount; i++) {
        printf("%-25s %10.2f %10.2f %10s\n", results[i].name, results[i].time_ms, results[i].gflops, results[i].correct ? "OK" : "ERROR");
    }

    printf("\n=== CGMA ANALYSIS ===\n");
    printf("TODO: Add CGMA analysis here.\n");

    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    printf("\n");
    return 0;
}