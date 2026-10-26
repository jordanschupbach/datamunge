#pragma once
#include <benchmark/benchmark.h>

#include <cmath>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
namespace nb {
namespace nn = datamunge::nn;
struct Scope {
  int               threads = nn::get_num_threads();
  nn::LinalgBackend backend = nn::linalg_backend();
  Scope() { nn::set_num_threads(1); }
  ~Scope() {
    nn::set_num_threads(threads);
    nn::set_linalg_backend(backend);
  }
};
inline nn::Tensor random(const nn::Shape& shape, nn::DType dt = nn::DType::Float64) {
  nn::Generator g(42);
  return nn::normal(shape, 0.0, 0.1, g, dt);
}
inline void check(bool ok, const char* msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
inline double error(const nn::Tensor& a, const nn::Tensor& b) {
  return nn::max(nn::abs(nn::sub(a, b))).item();
}
inline void close(const nn::Tensor& a, const nn::Tensor& b, double tol) {
  check(error(a, b) < tol, "numerical reference check failed");
}
inline void consume(const nn::Tensor& t) {
  benchmark::DoNotOptimize(t.impl());
  benchmark::ClobberMemory();
}
inline void finite(const nn::Tensor& t) {
  for (double x : t.to_vector<double>())
    check(std::isfinite(x), "nonfinite output");
}
} // namespace nb
