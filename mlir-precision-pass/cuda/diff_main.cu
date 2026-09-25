// diff_main.cu
//
// Runs matmul_fp32 and matmul_fp16_compute_fp32_accum on the same random
// input, on the same device, and writes both output matrices to disk as
// raw float32 binaries. compare.py loads these and does the actual
// numerical verdict -- kept out of C++ deliberately so the tolerance
// logic lives in one place (compare.py) shared with the MLIR-side IR
// comparison, rather than duplicated across languages.
//
// Usage: ./matmul_diff <N> <out_prefix>
//   writes <out_prefix>_fp32.bin and <out_prefix>_fp16.bin

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

extern "C" void launch_matmul_fp32(const float *, const float *, float *,
                                    int, cudaStream_t);
extern "C" void launch_matmul_fp16(const float *, const float *, float *,
                                    int, cudaStream_t);

static void checkCuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <N (multiple of 16)> <out_prefix>\n", argv[0]);
    return 1;
  }
  int N = atoi(argv[1]);
  const char *prefix = argv[2];

  std::vector<float> hA(N * N), hB(N * N);
  std::mt19937 rng(42);
  // Deliberately wide dynamic range: fp16 has ~3 decimal digits of
  // precision, so a uniform [-1,1] input would under-stress rounding.
  // Mixing magnitudes surfaces the cases that matter for a correctness
  // writeup.
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (auto &v : hA) v = dist(rng);
  for (auto &v : hB) v = dist(rng);

  float *dA, *dB, *dC32, *dC16;
  checkCuda(cudaMalloc(&dA, N * N * sizeof(float)), "malloc A");
  checkCuda(cudaMalloc(&dB, N * N * sizeof(float)), "malloc B");
  checkCuda(cudaMalloc(&dC32, N * N * sizeof(float)), "malloc C32");
  checkCuda(cudaMalloc(&dC16, N * N * sizeof(float)), "malloc C16");

  checkCuda(cudaMemcpy(dA, hA.data(), N * N * sizeof(float),
                        cudaMemcpyHostToDevice), "copy A");
  checkCuda(cudaMemcpy(dB, hB.data(), N * N * sizeof(float),
                        cudaMemcpyHostToDevice), "copy B");

  launch_matmul_fp32(dA, dB, dC32, N, 0);
  launch_matmul_fp16(dA, dB, dC16, N, 0);
  checkCuda(cudaDeviceSynchronize(), "sync");

  std::vector<float> hC32(N * N), hC16(N * N);
  checkCuda(cudaMemcpy(hC32.data(), dC32, N * N * sizeof(float),
                        cudaMemcpyDeviceToHost), "copy C32");
  checkCuda(cudaMemcpy(hC16.data(), dC16, N * N * sizeof(float),
                        cudaMemcpyDeviceToHost), "copy C16");

  char path[256];
  snprintf(path, sizeof(path), "%s_fp32.bin", prefix);
  FILE *f32 = fopen(path, "wb");
  fwrite(hC32.data(), sizeof(float), N * N, f32);
  fclose(f32);

  snprintf(path, sizeof(path), "%s_fp16.bin", prefix);
  FILE *f16 = fopen(path, "wb");
  fwrite(hC16.data(), sizeof(float), N * N, f16);
  fclose(f16);

  printf("Wrote %dx%d results to %s_fp32.bin / %s_fp16.bin\n", N, N, prefix,
         prefix);

  cudaFree(dA); cudaFree(dB); cudaFree(dC32); cudaFree(dC16);
  return 0;
}
