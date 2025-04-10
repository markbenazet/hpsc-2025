#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iostream>

void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) { 
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++]; 
  }
  for (int i=0; i<tmp.size(); i++) 
    vec[begin++] = tmp[i];
}

void merge_sort(std::vector<int>& vec, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;
    if (end - begin > 1000) {
        #pragma omp task shared(vec)
        merge_sort(vec, begin, mid);

        #pragma omp task shared(vec)
        merge_sort(vec, mid + 1, end);

        #pragma omp taskwait
    } else {
        // For small subarrays, perform serial sorting
        merge_sort(vec, begin, mid);
        merge_sort(vec, mid + 1, end);
    }

      // Merge the two halves
      merge(vec, begin, mid, end);
    }
}

int main() {
  int n = 2000000;
  std::vector<int> vec(n);
  // for (int i=0; i<n; i++) {
  //   vec[i] = rand() % (10 * n);
  //   printf("%d ",vec[i]);
  // }
  // printf("\n");

  #pragma omp parallel
  {
      #pragma omp single
      num_threads = omp_get_thread_num();
  }
  std::cout << "Number of threads: " << num_threads << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
    {
        #pragma omp single
        merge_sort(vec, 0, vec.size() - 1);
    }
  auto end = std::chrono::high_resolution_clock::now();

  // for (int i=0; i<n; i++) {
  //   printf("%d ",vec[i]);
  // }
  // printf("\n");

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time spent: " << elapsed.count() << " seconds" << std::endl;
}
