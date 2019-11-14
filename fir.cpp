#include "simd.hpp"
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

// cv::Mat1f
std::array<cv::Mat, 40> images;

void init()
{
  cv::Mat figure = cv::Mat::zeros(600, 800, CV_32FC1);
  for (int i = 0, max = images.size(); i < max; i++) {
    cv::Mat image = cv::Mat::zeros(600, 800, CV_32FC1);
    cv::randn(image, cv::Scalar(100), cv::Scalar(50));
    cv::circle(figure, cv::Point(400, 300), 100, cv::Scalar::all(20 * std::sin(i * 6.28 / 5) + 20), -1);

    image += figure;
    images.at(i) = image;

    cv::imshow("image", image);
    cv::waitKey(10);
  }
}

const int ORDER = 20;
const std::array<float, ORDER> coeff = {
    0.0101, -0.0053, -0.0306, -0.0160, 0.0641, 0.0891, -0.0436, -0.1685, -0.0584, 0.1616, 0.1616, -0.0584, -0.1685, -0.0436, 0.0891, 0.0641, -0.0160, -0.0306, -0.0053, 0.0101};

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


int main()
{
  init();
  cv::Mat sum = fir();
  sum.convertTo(sum, CV_8UC1);
  cv::imshow("image", sum);
  cv::waitKey(0);

  return 0;
}