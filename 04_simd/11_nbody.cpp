#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h> 

int main() {
  const int N = 16;
  srand48(1234);
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N]; 
  
  for(int i=0; i<N; i++) {
    x[i] = (float)drand48();
    y[i] = (float)drand48();
    m[i] = (float)drand48();
    fx[i] = fy[i] = 0.0f;
  }

    __m512 x_vec = _mm512_load_ps(x);
    __m512 y_vec = _mm512_load_ps(y);
    __m512 m_vec = _mm512_load_ps(m);

    __m512 zero_vec = _mm512_setzero_ps();
    __m512 epsilon_sq_vec = _mm512_set1_ps(1e-6f); 

  
  for(int i=0; i<N; i++) {

    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    
    __m512 rx_vec = _mm512_sub_ps(x_vec, xi);
    __m512 ry_vec = _mm512_sub_ps(y_vec, yi);
    __m512 r2_vec = _mm512_fmadd_ps(rx_vec, rx_vec, _mm512_mul_ps(ry_vec, ry_vec));

    __mmask16 self_interaction_mask = (1 << i);
    __mmask16 r_too_small_mask = _mm512_cmp_ps_mask(r2_vec, epsilon_sq_vec, _CMP_LT_OQ);
    __mmask16 ignore_force_mask = self_interaction_mask | r_too_small_mask;

    __m512 inv_r_vec = _mm512_rsqrt14_ps(r2_vec);
    
    __m512 inv_r3_vec = _mm512_mul_ps(_mm512_mul_ps(inv_r_vec, inv_r_vec), inv_r_vec);

    __m512 coef_vec = _mm512_mul_ps(m_vec, inv_r3_vec);
    coef_vec = _mm512_mul_ps(coef_vec, _mm512_set1_ps(-1.0f));


    __m512 fx_components_vec = _mm512_mul_ps(rx_vec, coef_vec);
    __m512 fy_components_vec = _mm512_mul_ps(ry_vec, coef_vec);

    fx_components_vec = _mm512_mask_blend_ps(ignore_force_mask, fx_components_vec, zero_vec);
    fy_components_vec = _mm512_mask_blend_ps(ignore_force_mask, fy_components_vec, zero_vec);

    fx[i] = _mm512_reduce_add_ps(fx_components_vec);
    fy[i] = _mm512_reduce_add_ps(fy_components_vec);
  }

  for(int i=0; i<N; i++) {
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
  
  return 0;
}