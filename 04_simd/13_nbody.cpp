#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], index[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    index[i] = i;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 jvec = _mm256_load_ps(index);
  for(int i=0; i<N; i++) {
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);
    __m256 rvec = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));
    rvec = _mm256_mul_ps(rvec, _mm256_mul_ps(rvec, rvec));
    __m256 mrvec = _mm256_mul_ps(mvec, rvec);
    __m256 rxmvec = _mm256_mul_ps(rxvec, mrvec);
    __m256 rymvec = _mm256_mul_ps(ryvec, mrvec);
    
    __m256 mask = _mm256_cmp_ps(_mm256_set1_ps(i), jvec, _CMP_NEQ_OQ);
    __m256 dfxvec = _mm256_setzero_ps();
    __m256 dfyvec = _mm256_setzero_ps();
    dfxvec = _mm256_blendv_ps(dfxvec, rxmvec, mask);
    dfyvec = _mm256_blendv_ps(dfyvec, rymvec, mask);
    
    __m256 dfxvec2 = _mm256_permute2f128_ps(dfxvec, dfxvec, 1);
    __m256 dfyvec2 = _mm256_permute2f128_ps(dfyvec, dfyvec, 1);
    dfxvec2 = _mm256_add_ps(dfxvec2, dfxvec);
    dfyvec2 = _mm256_add_ps(dfyvec2, dfyvec);
    dfxvec2 = _mm256_hadd_ps(dfxvec2, dfxvec2);
    dfyvec2 = _mm256_hadd_ps(dfyvec2, dfyvec2);
    dfxvec2 = _mm256_hadd_ps(dfxvec2, dfxvec2);
    dfyvec2 = _mm256_hadd_ps(dfyvec2, dfyvec2);
    float temp[N];
    _mm256_store_ps(temp, dfxvec2);
    fx[i] -= temp[0];
    _mm256_store_ps(temp, dfyvec2);
    fy[i] -= temp[0];
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
