#include "simd.hpp"
#include <cmath>
#include <iostream>

/*!
 * @brief 内積計算を行う関数
 * @param [in] a  ベクトルその1
 * @param [in] b  ベクトルその2
 * @param [in] n  ベクトルのサイズ
 * @return  内積
 */
#if defined(ENABLE_AVX512)
static inline float innerProductAVX512(const float* a, const float* b, std::size_t n)
{
  static constexpr std::size_t INTERVAL = sizeof(__m512) / sizeof(float);
  __m512 sumx16 = {0};
  for (std::size_t i = 0; i < n; i += INTERVAL) {
    __m512 ax16 = _mm512_load_ps(&a[i]);
    __m512 bx16 = _mm512_load_ps(&b[i]);
    sumx16 = _mm512_fmadd_ps(ax16, bx16, sumx16);
  }

  alignas(ALIGN) float s[INTERVAL] = {0};
  _mm512_store_ps(s, sumx16);

  std::size_t offset = n - n % INTERVAL;
  return std::inner_product(
      a + offset,
      a + n,
      b + offset,
      std::accumulate(std::begin(s), std::end(s), 0.0f));
}
#endif

#if defined(ENABLE_AVX512) || defined(ENABLE_AVX)
static inline float innerProductAVX(const float* a, const float* b, std::size_t n)
{
  static constexpr std::size_t INTERVAL = sizeof(__m256) / sizeof(float);
  __m256 sumx8 = {0};
  for (std::size_t i = 0; i < n; i += INTERVAL) {
    __m256 ax8 = _mm256_load_ps(&a[i]);
    __m256 bx8 = _mm256_load_ps(&b[i]);
    sumx8 = _mm256_fmadd_ps(ax8, bx8, sumx8);
  }

  alignas(ALIGN) float s[INTERVAL] = {0};
  _mm256_store_ps(s, sumx8);

  std::size_t offset = n - n % INTERVAL;
  return std::inner_product(
      a + offset,
      a + n,
      b + offset,
      std::accumulate(std::begin(s), std::end(s), 0.0f));
}
#endif

static inline float innerProductNormal(const float* a, const float* b, std::size_t n)
{
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; i++) {
    // <cmath>のstd::fma関数を用いると，積和演算がハードウェアのサポートを受けることを期待できる
    // 処理としては， sum += a[i] * b[i]; と同じ
    sum = std::fma(a[i], b[i], sum);
  }
  return sum;
}

int main()
{
  static constexpr int N_ELEMENT = 256;

  std::unique_ptr<float[], AlignedDeleter> a(alignedAllocArray<float>(N_ELEMENT, ALIGN));
  std::unique_ptr<float[], AlignedDeleter> b(alignedAllocArray<float>(N_ELEMENT, ALIGN));

  for (int i = 0; i < N_ELEMENT; i++) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
  }

#if defined(ENABLE_AVX512)
  std::cout << "512: \t";
  std::cout << innerProductAVX512(a.get(), b.get(), N_ELEMENT) << std::endl;
#endif

#if defined(ENABLE_AVX512) || defined(ENABLE_AVX)
  std::cout << "avx: \t";
  std::cout << innerProductAVX(a.get(), b.get(), N_ELEMENT) << std::endl;
#endif

  std::cout << "fma: \t";
  std::cout << innerProductNormal(a.get(), b.get(), N_ELEMENT) << std::endl;

  return 0;
}