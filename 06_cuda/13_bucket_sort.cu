#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void countElements(int *key, int n, int range, int *blockCounts) {
  extern __shared__ int localBucket[];
  
  for (int i = threadIdx.x; i < range; i += blockDim.x) {
      localBucket[i] = 0;
  }
  __syncthreads();
  
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      atomicAdd(&localBucket[key[i]], 1);
  }
  __syncthreads();
  
  for (int i = threadIdx.x; i < range; i += blockDim.x) {
      blockCounts[blockIdx.x * range + i] = localBucket[i];
  }
}

__global__ void sortElements(int *key, int n, int range, int *blockCounts, int *output) {
  extern __shared__ int sharedMem[];
  int* localBucket = sharedMem;
  int* globalOffset = &sharedMem[range];
  
  for (int i = threadIdx.x; i < range; i += blockDim.x) {
      localBucket[i] = 0;
      globalOffset[i] = 0;
  }
  __syncthreads();
  
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      atomicAdd(&localBucket[key[i]], 1);
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
      for (int b = 0; b < range; b++) {
          int globalPos = 0;
          for (int blk = 0; blk < blockIdx.x; blk++) {
              globalPos += blockCounts[blk * range + b];
          }
          globalOffset[b] = globalPos;
      }
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
      int sum = 0;
      for (int b = 0; b < range; b++) {
          int temp = localBucket[b];
          localBucket[b] = sum;
          sum += temp;
      }
  }
  __syncthreads();
  
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      int val = key[i];
      int pos = globalOffset[val] + localBucket[val];
      atomicAdd(&localBucket[val], 1);
      output[pos] = val;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int numBlocks = (n + 255) / 256;
  std::vector<int> key(n);

  print("Unsorted array: ");
  srand(time(0));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  
  int *d_key, *d_output, *d_blockCounts;
  cudaMalloc(&d_key, n * sizeof(int));
  cudaMalloc(&d_output, n * sizeof(int));
  cudaMalloc(&d_blockCounts, numBlocks * range * sizeof(int));
  
  cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  
  countElements<<<numBlocks, 256, range * sizeof(int)>>>(d_key, n, range, d_blockCounts);
  
  sortElements<<<numBlocks, 256, 2 * range * sizeof(int)>>>(d_key, n, range, d_blockCounts, d_output);
  
  cudaDeviceSynchronize();
  
  cudaMemcpy(key.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaFree(d_key);
  cudaFree(d_output);
  cudaFree(d_blockCounts);

  printf("\n Sorted array: ");
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
