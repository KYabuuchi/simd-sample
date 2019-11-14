#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <type_traits>
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

#if defined(__AVX512F__)
static constexpr int ALIGN = alignof(__m512);  // = 64
#elif defined(__AVX__)
static constexpr int ALIGN = alignof(__m256);  // = 32
#else
static constexpr int ALIGN = 8;
#endif