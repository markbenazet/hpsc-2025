#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h> 

int main() {
  const int N = 8;
  alignas(32) float x[N], y[N], m[N], fx[N], fy[N]; 
  
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  
  for(int i=0; i<N; i++) {
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);
  
    __m256 xj = _mm256_load_ps(x);  
    __m256 yj = _mm256_load_ps(y);  
    __m256 mj = _mm256_load_ps(m);  
    
    __m256 rx = _mm256_sub_ps(xi, xj);  
    __m256 ry = _mm256_sub_ps(yi, yj);  
    
    __m256 rx2 = _mm256_mul_ps(rx, rx);  
    __m256 ry2 = _mm256_mul_ps(ry, ry);  
    __m256 r2 = _mm256_add_ps(rx2, ry2); 
    
    __m256 r = _mm256_sqrt_ps(r2);          
    __m256 r3 = _mm256_mul_ps(r2, r);       
    
    __m256 mask = _mm256_cmp_ps(xi, xj, _CMP_NEQ_OQ);  
    __m256 r3_safe = _mm256_blendv_ps(_mm256_set1_ps(1.0f), r3, mask); 
    
    __m256 invr3 = _mm256_div_ps(_mm256_set1_ps(1.0f), r3_safe);  
    __m256 coef = _mm256_mul_ps(mj, invr3);  
    
    coef = _mm256_and_ps(coef, mask);  
    
    __m256 fx_incr = _mm256_mul_ps(rx, coef);  
    __m256 fy_incr = _mm256_mul_ps(ry, coef);  
    
    alignas(32) float fx_array[8], fy_array[8];
    _mm256_store_ps(fx_array, fx_incr);
    _mm256_store_ps(fy_array, fy_incr);
    
    for(int j=0; j<N; j++) {
      fx[i] -= fx_array[j];
      fy[i] -= fy_array[j];
    }
    
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
  
  return 0;
}