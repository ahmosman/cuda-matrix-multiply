# CUDA Matrix Multiplication

[![Release Date](https://img.shields.io/badge/release-May%202025-blue)]()

## Project Description

This project implements high-performance matrix multiplication using NVIDIA CUDA. It demonstrates how to leverage GPU parallelism to accelerate the computation of large matrix products, specifically targeting square matrices of size 4096x4096.

## Problem Statement

Matrix multiplication is a fundamental operation in scientific computing, machine learning, and graphics. However, multiplying large matrices is computationally intensive and time-consuming on CPUs. The problem solved here is to efficiently compute the product of two large matrices by exploiting the massive parallelism available on modern GPUs.

## Solution & Algorithm

The solution uses CUDA to parallelize matrix multiplication. The main algorithm divides the computation into blocks and threads, where each thread computes one or more elements of the output matrix. Shared memory is used to cache sub-blocks (tiles) of the input matrices, reducing global memory accesses and improving performance.

- **Tiling:** The input matrices are divided into tiles of size `BLOCK_SIZE` (16x16 by default).
- **Thread Mapping:** Each CUDA thread block computes a tile of the output matrix. Threads within a block cooperate to load tiles of A and B into shared memory.
- **Multiple Results per Thread:** The kernel template allows each thread to compute multiple output elements, further improving efficiency.
- **CPU Reference:** The project also computes the result on the CPU for correctness verification.

## How to Run

### Prerequisites

- CUDA Toolkit (tested with CUDA 12.5)
- NVIDIA GPU (tested with GeForce GTX 1080, Compute Capability 6.1)
- Linux or Windows with a compatible compiler

To **build** the project, run:

```sh
make
```

This will produce the `matrixMul` executable.

To **run** the program:

```sh
./matrixMul
```

The program will:
- Generate random matrices (if not already present)
- Compute the reference result on the CPU
- Run multiple CUDA kernel configurations
- Print performance and correctness summary

To **remove build** artifacts:

```sh
make clean
```

## Technologies Used

- CUDA C++
- C++11

---

**Used GPU:**  
NVIDIA GeForce GTX 1080
For detailed GPU specifications, see [file](config_gpu_GeForce%20GTX%201080.txt).

## Research papers

Our research paper in Polish is available here

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



