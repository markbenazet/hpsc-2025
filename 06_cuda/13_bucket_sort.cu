#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void countElements(int *input, int *counts, int n, int range) {
  for (int i = threadIdx.x; i < range; i += blockDim.x) {
      counts[i] = 0;
  }
  __syncthreads();
  
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      atomicAdd(&counts[input[i]], 1);
  }
}

__global__ void sortElements(int *input, int *output, int *counts, int n, int range) {
  __shared__ int prefixSum[5];
  
  for (int i = threadIdx.x; i < range; i += blockDim.x) {
      prefixSum[i] = counts[i];
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
      int sum = 0;
      for (int i = 0; i < range; i++) {
          int count = prefixSum[i];
          prefixSum[i] = sum;
          sum += count;
      }
  }
  __syncthreads();
  
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      int value = input[i];
      int pos = atomicAdd(&prefixSum[value], 1);
      output[pos] = value;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int numBlocks = (n + 255) / 256;
  std::vector<int> key(n);

  printf("Unsorted array: ");
  srand(time(0));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  
  int *d_input, *d_output, *d_counts;
  cudaMalloc(&d_input, n * sizeof(int));
  cudaMalloc(&d_output, n * sizeof(int));
  cudaMalloc(&d_counts, range * sizeof(int));
  
  cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  
  countElements<<<numBlocks, 256>>>(d_input, d_counts, n, range);
  
  sortElements<<<numBlocks, 256>>>(d_input, d_output, d_counts, n, range);
  
  cudaDeviceSynchronize();
  
  std::vector<int> output(n);
  cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
  
  printf("Sorted array: ");
  for (int i = 0; i < n; i++) {
      printf("%d ", output[i]);
  }
  printf("\n");
  
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_counts);
  
  return 0;
}
