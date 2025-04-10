#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono> // For timing

int main() {
  int n = 2300000;
  int range = 2300000;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    // printf("%d ",key[i]);
  }
  printf("\n");

  auto start = std::chrono::high_resolution_clock::now(); // Start timing

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  // for (int i=0; i<range; i++) {
  //   printf("Bucket %d: %d\n", i, bucket[i]);
  // }
  // printf("\n");
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
  // for (int i=0; i<range; i++) {
  //   printf("Offset %d: %d\n", i, offset[i]);
  // }
  // printf("\n");

#pragma omp parallel for
for (int i = 0; i < range; i++) {
    int j = offset[i];
    for (; bucket[i] > 0; bucket[i]--) {
        key[j++] = i;
    }
}


  auto end = std::chrono::high_resolution_clock::now(); // End timing
  std::chrono::duration<double> elapsed = end - start; // Calculate elapsed time

  // for (int i=0; i<n; i++) {
  //   printf("%d ",key[i]);
  // }
  // printf("\n");

  printf("Elapsed time: %.6f seconds\n", elapsed.count());
}
