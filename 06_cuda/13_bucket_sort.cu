#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>


__global__ void countElements(int *input, int *counts, int n, int range) {
    
    extern __shared__ int sharedCounts[];
    
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        sharedCounts[i] = 0;
    }
    __syncthreads();
    
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&sharedCounts[input[i]], 1);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        atomicAdd(&counts[i], sharedCounts[i]);
    }
}

__global__ void sortElements(int *input, int *output, int *counts, int n, int range) {
    
    extern __shared__ int prefixSum[];
    
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
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    std::vector<int> key(n);
    
    printf("Unsorted array: ");
    srand(time(0));
    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");
    
    int *d_input, *d_output, *d_counts;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_counts, range * sizeof(int));
    
    cudaMemset(d_counts, 0, range * sizeof(int));
    
    cudaMemcpy(d_input, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    countElements<<<numBlocks, blockSize, range * sizeof(int)>>>(d_input, d_counts, n, range);
      
    cudaDeviceSynchronize();
    
    sortElements<<<numBlocks, blockSize, range * sizeof(int)>>>(d_input, d_output, d_counts, n, range);
    
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