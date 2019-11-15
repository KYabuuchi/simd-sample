#include "simd.hpp"
#include "timer.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

// cv::Mat1f
std::array<cv::Mat, 40> images;

constexpr int ORDER = 20;
const std::array<float, ORDER> coeff = {
    0.0101, -0.0053, -0.0306, -0.0160, 0.0641, 0.0891, -0.0436, -0.1685, -0.0584, 0.1616, 0.1616, -0.0584, -0.1685, -0.0436, 0.0891, 0.0641, -0.0160, -0.0306, -0.0053, 0.0101};

#if defined(__AVX512F__)
__m512 coeff16[ORDER];
#endif

#if defined(__AVX__)
__m256 coeff8[ORDER];
#endif

void init()
{
  cv::Mat figure = cv::Mat::zeros(600, 800, CV_32FC1);
  for (int i = 0, max = images.size(); i < max; i++) {
    cv::Mat image = cv::Mat::zeros(600, 800, CV_32FC1);
    cv::randn(image, cv::Scalar(100), cv::Scalar(50));
    cv::circle(figure, cv::Point(400, 300), 100, cv::Scalar::all(20 * std::sin(i * 6.28 / 5) + 20), -1);

    image += figure;
    images.at(i) = image;

    cv::Mat show;
    image.convertTo(show, CV_8UC1);
    cv::imshow("image", show);
    cv::waitKey(2);
  }

#if defined(__AVX__)
  for (int i = 0; i < ORDER; i++) {
    alignas(ALIGN) float c[8] = {};
    for (int j = 0; j < 8; j++)
      c[j] = coeff[i];
    __m256 c8 = _mm256_load_ps(c);
    coeff8[i] = c8;
  }
#endif
#if defined(__AVX512F__)
  for (int i = 0; i < ORDER; i++) {
    alignas(ALIGN) float c[16] = {};
    for (int j = 0; j < 16; j++)
      c[j] = coeff[i];
    __m512 c16 = _mm512_load_ps(c);
    coeff16[i] = c16;
  }
#endif
}

int offset = 0;

cv::Mat fir()
{
  Timer t("fir");
  cv::Mat sum = cv::Mat::zeros(600, 800, CV_32FC1);
  for (int i = 0; i < ORDER; i++)
    sum += coeff.at(i) * images.at(i + offset);

  return sum;
}

cv::Mat firForeach()
{
  Timer t("each");
  cv::Mat sum = cv::Mat::zeros(600, 800, CV_32FC1);
  sum.forEach<float>([=](float& p, const int* pos) -> void {
    for (int i = 0; i < ORDER; i++) {
      p = std::fma(coeff.at(i), images.at(i + offset).at<float>(pos[0], pos[1]), p);
    }
  });
  return sum;
}

namespace
{
#if defined(__AVX512F__)

float* float_input[20];

inline void impleAVX512Init(__m512& sum16, int i, int j)
{
  float_input[i] = reinterpret_cast<float*>(images.at(i + offset).data);
  sum16 = _mm512_fmadd_ps(_mm512_load_ps(&float_input[i][j]), coeff16[i], sum16);
}

inline void impleAVX512(__m512& sum16, int i, int j)
{
  sum16 = _mm512_fmadd_ps(_mm512_load_ps(&float_input[i][j]), coeff16[i], sum16);
}

cv::Mat sum_image = cv::Mat::zeros(600, 800, CV_32FC1);


inline cv::Mat firAVX512()
{
  static constexpr std::size_t INTERVAL = sizeof(__m512) / sizeof(float);
  float* sum = reinterpret_cast<float*>(sum_image.data);

  Timer t("512");

  // 初期化
  {
    __m512 sum16 = _mm512_setzero_ps();
    for (int i = 0; i < ORDER; ++i) {
      impleAVX512Init(sum16, i, 0);
    }
    _mm512_store_ps(&sum[0], sum16);
  }

// mainループ
#pragma omp parallel for
  for (int j = INTERVAL; j < 600 * 800; j += INTERVAL) {
    __m512 sum16 = _mm512_setzero_ps();
    for (int i = 0; i < ORDER; i += 10) {
      impleAVX512(sum16, 0 + i, j);
      impleAVX512(sum16, 1 + i, j);
      impleAVX512(sum16, 2 + i, j);
      impleAVX512(sum16, 3 + i, j);
      impleAVX512(sum16, 4 + i, j);
      impleAVX512(sum16, 5 + i, j);
      impleAVX512(sum16, 6 + i, j);
      impleAVX512(sum16, 7 + i, j);
      impleAVX512(sum16, 8 + i, j);
      impleAVX512(sum16, 9 + i, j);
    }
    _mm512_store_ps(&sum[j], sum16);
  }
  return sum_image;
}
#endif
}  // namespace

#if defined(__AVX__)
inline void impleAVX(__m256& sum8, int i, int j)
{
  uchar* uchar_input = images.at(i + offset).data;
  float* float_input = reinterpret_cast<float*>(uchar_input);
  sum8 = _mm256_fmadd_ps(_mm256_load_ps(&float_input[j]), coeff8[i], sum8);
}
inline cv::Mat firAVX()
{
  Timer t("256");
  cv::Mat sum_image = cv::Mat::zeros(600, 800, CV_32FC1);
  uchar* uchar_sum = sum_image.data;
  float* sum = reinterpret_cast<float*>(uchar_sum);

  static constexpr std::size_t INTERVAL = sizeof(__m256) / sizeof(float);
#pragma omp parallel for
  for (int j = 0; j < 600 * 800; j += INTERVAL) {
    __m256 sum8 = _mm256_load_ps(&sum[j]);

    for (int i = 0; i < ORDER; ++i) {
      impleAVX(sum8, i, j);
    }
    _mm256_store_ps(&sum[j], sum8);
  }
  return sum_image;
}
#endif

int main()
{
  std::cout << "init start" << std::endl;
  init();

#if defined(__AVX512F__)
  std::cout << "avx512 start" << std::endl;
  for (int i = 0; i < 10; i++) {
    offset = i;
    cv::Mat sum = firAVX512();
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(50);
  }
#endif

#if defined(__AVX__)
  std::cout << "avx start" << std::endl;
  for (int i = 0; i < 10; i++) {
    offset = i;
    cv::Mat sum = firAVX();
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(50);
  }
#endif

  std::cout << "fir start" << std::endl;
  for (int i = 0; i < 10; i++) {
    offset = i;
    cv::Mat sum = fir();
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(50);
  }

  std::cout << "each start" << std::endl;
  for (int i = 0; i < 10; i++) {
    offset = i;
    cv::Mat sum = firForeach();
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(50);
  }

  return 0;
}