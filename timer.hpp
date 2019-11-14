#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>

class Timer
{
public:
  Timer(const std::string& id = "")
      : m_id(id), m_raii(id != "")
  {
    m_start = std::chrono::system_clock::now();
  }
  ~Timer()
  {
    if (m_raii) {
      long count = microSeconds();
      std::cout << "\033[33m\033[40m" << m_id << ": \t" << std::fixed << std::setprecision(4) << double(count) / 1000.00 << " [ms]\033[m" << std::endl;
    }
  }
  void reset()
  {
    m_start = std::chrono::system_clock::now();
  }
  long millSeconds()
  {
    auto dur = std::chrono::system_clock::now() - m_start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
  }
  long microSeconds()
  {
    auto dur = std::chrono::system_clock::now() - m_start;
    return std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
  }
  void call()
  {
    m_raii = false;
    long count = microSeconds();
    std::cout << "\033[33m\033[40m" << m_id << ": \t" << std::fixed << std::setprecision(4) << double(count) / 1000.00 << " [ms]\033[m" << std::endl;
  }

private:
  std::chrono::system_clock::time_point m_start;
  const std::string m_id;
  bool m_raii;
};
