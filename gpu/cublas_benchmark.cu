#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define NUM_RUNS 10
#define TILE_SIZE 32

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                             \
        if (status != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(status)           \
                      << " at line " << __LINE__ << std::endl;                  \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error " << static_cast<int>(status)            \
                      << " at line " << __LINE__ << std::endl;                  \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

__global__ void gemm_o3_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_o3(const float* A, const float* B, float* C,
               int M, int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_o3_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_cublas(cublasHandle_t handle, const float* A, const float* B,
                   float* C, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Matrices in this MP are row-major. cuBLAS assumes column-major, so
    // compute C^T = B^T * A^T by swapping A/B and M/N in the call.
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             C, N));
}

template <typename Launch>
float time_kernel(Launch launch) {
    for (int i = 0; i < 2; ++i) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < NUM_RUNS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / NUM_RUNS;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./cublas_benchmark M N K" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    for (float& x : A) x = static_cast<float>(std::rand() % 10);
    for (float& x : B) x = static_cast<float>(std::rand() % 10);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float o3_ms = time_kernel([&] { launch_o3(d_A, d_B, d_C, M, N, K); });
    float cublas_ms = time_kernel([&] { launch_cublas(handle, d_A, d_B, d_C, M, N, K); });

    std::cout << "o3: " << o3_ms << " ms\n";
    std::cout << "cuBLAS: " << cublas_ms << " ms\n";

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
