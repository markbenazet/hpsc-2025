#include <cstdio>
#include <omp.h>

int main() {
  int n = 1e8;
  double dx = 1. / n;
  double pi = 0;

  double start_time = omp_get_wtime(); // Start timing
  #pragma omp parallel for reduction(+:pi)
  for (int i = 0; i < n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  double end_time = omp_get_wtime(); // End timing

  #pragma omp parallel
  {
    // Print the number of threads used
    int num_threads = omp_get_num_threads();
    #pragma omp single
    {
      printf("Number of threads: %d\n", num_threads);
    }
  }

  double elapsed_time = end_time - start_time; // Time in seconds
  printf("%17.15f\n", pi);

  // Calculate GFLOPs per second
  double flops = 7.0 * n; // Total GFLOPs
  double flops_per_second = flops / (elapsed_time * 1e9); // Convert to FLOPs per second
  printf("GFLOPs per second: %.9f\n", flops_per_second);

  return 0;
}

