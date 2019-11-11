#if defined(ENABLE_AVX512) && !defined(__AVX512F__)
#error Macro: ENABLE_AVX512 is defined, but unable to use AVX512F intrinsic functions
#elif defined(ENABLE_AVX) && !defined(__AVX__)
#error Macro: ENABLE_AVX is defined, but unable to use AVX intrinsic functions
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>

// SIMD
#include <x86intrin.h>

/*!
 * @brief アラインメントされたメモリを動的確保する関数．配列向けにalignedMallocの引数指定が簡略化されている
 * @tparam T  確保する配列の要素型．この関数の返却値はT*
 * @param [in] size       確保する要素数．すなわち確保するサイズは size * sizeof(T)
 * @param [in] alignment  アラインメント (2のべき乗を指定すること)
 * @return  アラインメントし，動的確保されたメモリ領域へのポインタ
 */
template <typename T>
static inline T* alignedAllocArray(std::size_t size, std::size_t alignment = alignof(T)) noexcept
{
  void* p;
  return reinterpret_cast<T*>(::posix_memalign(&p, alignment, size * sizeof(T)) == 0 ? p : nullptr);
}

/*!
 * @brief std::unique_ptr で利用するアラインされたメモリ用のカスタムデリータ
 */
struct AlignedDeleter {
  void operator()(void* p) const noexcept
  {
    std::free(p);
  }
};

#if defined(ENABLE_AVX512)
static constexpr int ALIGN = alignof(__m512);
#elif defined(ENABLE_AVX)
static constexpr int ALIGN = alignof(__m256);
#else
static constexpr int ALIGN = 8;
#endif


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

#if defined(ENABLE_AVX)
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
  std::cout << "avx512: \t";
  std::cout << innerProductAVX512(a.get(), b.get(), N_ELEMENT) << std::endl;
#endif

#if defined(ENABLE_AVX)
  std::cout << "avx: \t";
  std::cout << innerProductAVX(a.get(), b.get(), N_ELEMENT) << std::endl;
#endif

  std::cout << "std::fma: \t";
  std::cout << innerProductNormal(a.get(), b.get(), N_ELEMENT) << std::endl;

  return 0;
}