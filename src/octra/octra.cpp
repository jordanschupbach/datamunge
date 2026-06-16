#include <iostream>
#include <octra/octra.hpp>

namespace octra {
void hello() {
  std::cout << "Hello octra" << std::endl;
}

double call_with_callback(double x, Callback* cb) {
  return cb ? cb->call(x) : x;
}

std::vector<double> map_dvector_with_callback(const std::vector<double>& values, Callback* cb) {
  if (!cb) {
    return values;
  }
  std::vector<double> out;
  out.reserve(values.size());
  for (double v : values) {
    out.push_back(cb->call(v));
  }
  return out;
}

std::vector<double> make_dvector(double a, double b, double c) {
  return {a, b, c};
}

double sum_dvector(const std::vector<double>& values) {
  double sum = 0.0;
  for (double v : values) {
    sum += v;
  }
  return sum;
}

std::pair<double, double> make_dpair(double a, double b) {
  return {a, b};
}

double sum_dpair(const std::pair<double, double>& values) {
  return values.first + values.second;
}

} // namespace octra
