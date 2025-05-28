#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h> 

int main() {
  const int N = 16;
  srand48(1234);
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  alignas(64) int indices[N];
  
  for(int i=0; i<N; i++) {
    x[i] = (float)drand48();
    y[i] = (float)drand48();
    m[i] = (float)drand48();
    fx[i] = fy[i] = 0.0f;
    indices[i] = i;
  }

    __m512 x_vec = _mm512_load_ps(x);
    __m512 y_vec = _mm512_load_ps(y);
    __m512 m_vec = _mm512_load_ps(m);

    __m512i index_vec = _mm512_load_epi32(indices);


  
  for(int i=0; i<N; i++) {

    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    
    __m512 rx_vec = _mm512_sub_ps(x_vec, xi);
    __m512 ry_vec = _mm512_sub_ps(y_vec, yi);
    __m512 r2 = _mm512_fmadd_ps(rx_vec, rx_vec, _mm512_mul_ps(ry_vec, ry_vec));
    __m512 r = _mm512_sqrt_ps(r2);
    __m512 r3 = _mm512_mul_ps(r2, r);

    __m512i i_vec = _mm512_set1_epi32(i);
    __mmask16 self_mask = _mm512_cmpeq_epi32_mask(index_vec, i_vec);
    __mmask16 nonzero_mask = _mm512_cmp_ps_mask(r2, _mm512_set1_ps(1e-12f), _CMP_GT_OQ);
    __mmask16 mask = _kandn_mask16(self_mask, nonzero_mask);


    __m512 r3_safe = _mm512_mask_blend_ps(mask, _mm512_set1_ps(1.0f), r3);

    __m512 invr3 = _mm512_div_ps(_mm512_set1_ps(1.0f), r3_safe);
    
    __m512 coef = _mm512_mul_ps(m_vec, invr3);
    
    coef = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), coef);

    __m512 fx_components_vec = _mm512_mul_ps(rx_vec, coef);
    __m512 fy_components_vec = _mm512_mul_ps(ry_vec, coef);

    fx[i] = _mm512_reduce_add_ps(fx_components_vec);
    fy[i] = _mm512_reduce_add_ps(fy_components_vec);
  }


  printf("SIMD N-body simulation results:\n");
  printf("----------------------------------\n");
  for(int i=0; i<N; i++) {
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
  printf("----------------------------------\n");
  printf("Comparison of SIMD and standard results:\n");
  bool results_match = true;
  for(int i=0; i<N; i++) {
    float fx_i = 0.0f, fy_i = 0.0f;
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[j] - x[i];
        float ry = y[j] - y[i];
        float r2 = rx * rx + ry * ry;
        if(r2 > 1e-12f) {
          float r = sqrtf(r2);
          float invr3 = 1.0f / (r2 * r);
          fx_i += m[j] * rx * invr3;
          fy_i += m[j] * ry * invr3;
        }
      }
    }
    if(fabsf(fx[i] - fx_i) > 1e-6f || fabsf(fy[i] - fy_i) > 1e-6f) {
      results_match = false;
      break;
    }
  }
  
  if(results_match) {
    printf("TRUE - SIMD and standard results match\n");
  } else {
    printf("FALSE - SIMD and standard results do not match\n");
  }


  
  return 0;
}