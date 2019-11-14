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

__m512 coeff16[ORDER];

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

  // AVX512
  for (int i = 0; i < ORDER; i++) {
    alignas(ALIGN) float c[16] = {};
    for (int j = 0; j < 16; j++)
      c[j] = coeff[i];

    __m512 c16 = _mm512_load_ps(c);
    coeff16[i] = c16;
  }
}

cv::Mat fir(const int offset)
{
  Timer t("fir");
  cv::Mat sum = cv::Mat::zeros(600, 800, CV_32FC1);

  for (int i = 0; i < ORDER; i++) {
    sum += coeff.at(i) * images.at(i + offset);
  }

  return sum;
}

cv::Mat firForeach(const int offset)
{
  Timer t("each");
  cv::Mat sum = cv::Mat::zeros(600, 800, CV_32FC1);

  sum.forEach<float>([offset](float& p, const int* pos) -> void {
    for (int i = 0; i < ORDER; i++) {
      p = std::fma(coeff.at(i), images.at(i + offset).at<float>(pos[0], pos[1]), p);
    }
  });

  return sum;
}

constexpr std::size_t INTERVAL = sizeof(__m512) / sizeof(float);

int offset = 0;
inline void implFir512(float* sum, const int i)
{
  uchar* uchar_input = images.at(i + offset).data;
  float* input = reinterpret_cast<float*>(uchar_input);
#pragma omp parallel for
  for (int j = 0; j < 600 * 800; j += INTERVAL) {
    __m512 sum16 = _mm512_load_ps(&sum[j]);
    __m512 input16 = _mm512_load_ps(&input[j]);
    sum16 = _mm512_fmadd_ps(input16, coeff16[i], sum16);
    _mm512_store_ps(&sum[j], sum16);
  }
}


cv::Mat firAVX512()
{
  Timer t("512");
  cv::Mat sum_image = cv::Mat::zeros(600, 800, CV_32FC1);
  uchar* uchar_sum = sum_image.data;
  float* sum = reinterpret_cast<float*>(uchar_sum);

  implFir512(sum, 0 + 0);
  implFir512(sum, 0 + 1);
  implFir512(sum, 0 + 2);
  implFir512(sum, 0 + 3);
  implFir512(sum, 0 + 4);
  implFir512(sum, 0 + 5);
  implFir512(sum, 0 + 6);
  implFir512(sum, 0 + 7);
  implFir512(sum, 0 + 8);
  implFir512(sum, 0 + 9);
  implFir512(sum, 10 + 0);
  implFir512(sum, 10 + 1);
  implFir512(sum, 10 + 2);
  implFir512(sum, 10 + 3);
  implFir512(sum, 10 + 4);
  implFir512(sum, 10 + 5);
  implFir512(sum, 10 + 6);
  implFir512(sum, 10 + 7);
  implFir512(sum, 10 + 8);
  implFir512(sum, 10 + 9);

  return sum_image;
}

int main()
{
  std::cout << "init start" << std::endl;
  init();

  std::cout << "fir start" << std::endl;
  for (int i = 0; i < 10; i++) {
    cv::Mat sum = fir(i);
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(50);
  }
  std::cout << "each start" << std::endl;
  for (int i = 0; i < 10; i++) {
    cv::Mat sum = firForeach(i);
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(50);
  }
  std::cout << "fir512 start" << std::endl;
  for (int i = 0; i < 10; i++) {
    offset = i;
    cv::Mat sum = firAVX512();
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(50);
  }

  return 0;
}