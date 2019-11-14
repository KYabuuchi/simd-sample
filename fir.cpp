#include "simd.hpp"
#include <cmath>
#include <iostream>
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


cv::Mat fir()
{
  cv::Mat sum = cv::Mat::zeros(600, 800, CV_32FC1);
  uchar* uchar_data = sum.data;
  float* data = reinterpret_cast<float*>(uchar_data);

  for (int i = 0; i < 600 * 800 / 2; i++) {
    data[i] = 255;
    data[i + 600 * 800 / 2] = 120;
  }
  return sum;
}

constexpr std::size_t INTERVAL = sizeof(__m512) / sizeof(float);

cv::Mat firAVX512(const int offset)
{
  cv::Mat sum_image = cv::Mat::zeros(600, 800, CV_32FC1);
  uchar* uchar_sum = sum_image.data;
  float* sum = reinterpret_cast<float*>(uchar_sum);

  for (int i = 0; i < ORDER; i++) {
    uchar* uchar_input = images.at(i + offset).data;
    float* input = reinterpret_cast<float*>(uchar_input);

    for (int j = 0; j < 600 * 800; j += INTERVAL) {
      __m512 sum16 = _mm512_load_ps(&sum[j]);
      __m512 input16 = _mm512_load_ps(&input[j]);
      sum16 = _mm512_fmadd_ps(input16, coeff16[i], sum16);
      _mm512_store_ps(&sum[j], sum16);
    }
  }

  return sum_image;
}


int main()
{
  init();
  std::cout << "init done" << std::endl;

  for (int i = 0; i < 10; i++) {
    cv::Mat sum = firAVX512(i);
    sum.convertTo(sum, CV_8UC1);
    cv::imshow("image", sum);
    cv::waitKey(0);
  }

  return 0;
}