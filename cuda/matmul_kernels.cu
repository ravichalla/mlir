// matmul_kernels.cu
//
// Two versions of the same tiled matmul: an fp32 baseline and an
// fp16-compute/fp32-accumulate version (the GPU-side analog of what the
// MLIR pass does at IR level -- lower to f16 for throughput, keep a wider
// accumulator to bound error growth). This is the pair the differential
// harness actually executes on hardware to get real numbers, rather than
// relying on the IR-level story alone.
//
// Build (example, adjust arch for your GPU):
//   nvcc -O3 -arch=sm_80 -o matmul_diff matmul_kernels.cu diff_main.cu
//
// NOT validated to compile/run in this environment (no CUDA toolkit or
// GPU available here) -- treat as a correct-by-construction starting
// point following standard tiled-matmul + half-precision intrinsics
// patterns, to be built and numerically verified on real hardware.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE 16

// Baseline: fp32 throughout.
__global__ void matmul_fp32(const float *A, const float *B, float *C,
                             int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  for (int t = 0; t < N / TILE; ++t) {
    As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
    __syncthreads();

    for (int k = 0; k < TILE; ++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  C[row * N + col] = acc;
}

// fp16 compute, fp32 accumulate -- mirrors the MLIR pass's cast-at-boundary
// strategy: operands truncated to half, product/accumulation kept wide
// enough to bound rounding error over the reduction.
__global__ void matmul_fp16_compute_fp32_accum(const float *A,
                                                const float *B, float *C,
                                                int N) {
  __shared__ __half As[TILE][TILE];
  __shared__ __half Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f; // fp32 accumulator -- deliberate, see file header.
  for (int t = 0; t < N / TILE; ++t) {
    As[threadIdx.y][threadIdx.x] =
        __float2half(A[row * N + t * TILE + threadIdx.x]);
    Bs[threadIdx.y][threadIdx.x] =
        __float2half(B[(t * TILE + threadIdx.y) * N + col]);
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      float p = __half2float(As[threadIdx.y][k]) *
                __half2float(Bs[k][threadIdx.x]);
      acc += p;
    }
    __syncthreads();
  }
  C[row * N + col] = acc;
}

extern "C" void launch_matmul_fp32(const float *A, const float *B, float *C,
                                    int N, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid(N / TILE, N / TILE);
  matmul_fp32<<<grid, block, 0, stream>>>(A, B, C, N);
}

extern "C" void launch_matmul_fp16(const float *A, const float *B, float *C,
                                    int N, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid(N / TILE, N / TILE);
  matmul_fp16_compute_fp32_accum<<<grid, block, 0, stream>>>(A, B, C, N);
}
