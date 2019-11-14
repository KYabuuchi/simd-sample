#include "simd.hpp"
#include "timer.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

// cv::Mat1f
std::array<cv::Mat, 40> images;

const int ORDER = 20;
const std::array<float, ORDER> coeff = {
    0.0101, -0.0053, -0.0306, -0.0160, 0.0641, 0.0891, -0.0436, -0.1685, -0.0584, 0.1616, 0.1616, -0.0584, -0.1685, -0.0436, 0.0891, 0.0641, -0.0160, -0.0306, -0.0053, 0.0101};

#if defined(ENABLE_AVX512)
__m512 coeff16[ORDER];
#endif

#if defined(ENABLE_AVX512) || defined(ENABLE_AVX)
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

#if defined(ENABLE_AVX) || defined(ENABLE_AVX512)
  for (int i = 0; i < ORDER; i++) {
    alignas(ALIGN) float c[8] = {};
    for (int j = 0; j < 8; j++)
      c[j] = coeff[i];

    __m256 c8 = _mm256_load_ps(c);
    coeff8[i] = c8;
  }
#if defined(ENABLE_AVX512)
  for (int i = 0; i < ORDER; i++) {
    alignas(ALIGN) float c[16] = {};
    for (int j = 0; j < 16; j++)
      c[j] = coeff[i];

    __m512 c16 = _mm512_load_ps(c);
    coeff16[i] = c16;
  }
#endif
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

#if defined(ENABLE_AVX512)
inline void impleAVX512(__m512& sum16, int i, int j)
{
  uchar* uchar_input = images.at(i + offset).data;
  float* float_input = reinterpret_cast<float*>(uchar_input);
  sum16 = _mm512_fmadd_ps(_mm512_load_ps(&float_input[j]), coeff16[i], sum16);
}

inline cv::Mat firAVX512()
{
  Timer t("512");
  cv::Mat sum_image = cv::Mat::zeros(600, 800, CV_32FC1);
  uchar* uchar_sum = sum_image.data;
  float* sum = reinterpret_cast<float*>(uchar_sum);

  static constexpr std::size_t INTERVAL = sizeof(__m512) / sizeof(float);
#pragma omp parallel for
  for (int j = 0; j < 600 * 800; j += INTERVAL) {
    __m512 sum16 = _mm512_load_ps(&sum[j]);

    for (int i = 0; i < ORDER; ++i) {
      impleAVX512(sum16, i, j);
    }
    _mm512_store_ps(&sum[j], sum16);
  }
  return sum_image;
}
#endif

#if defined(ENABLE_AVX)
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

#if defined(ENABLE_AVX512)
  std::cout << "avx512 start" << std::endl;
  for (int i = 0; i < 10; i++) {
    offset = i;
    cv::Mat sum = firAVX512();
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(50);
  }
#endif

#if defined(ENABLE_AVX)
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