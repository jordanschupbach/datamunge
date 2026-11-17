// Independent check support; no training or experiment workflow.
#pragma once
#include <cstring>
#include <dm/dm.hpp>
#include <iostream>
namespace nn = dm::nn;
using nn::Tensor;
namespace check {
inline void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
inline void Near(double a, double b, double tol, const std::string& message) {
  Require(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol, message);
}
template<class F>
void Throws(F f, const std::string& message) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception&) {
    caught = true;
  }
  Require(caught, message);
}
} // namespace check
namespace sal {
inline auto F(const Tensor& t) {
  return t.contiguous().to_vector<float>();
}
inline auto FromF(const std::vector<float>& v, const nn::Shape& s) {
  return nn::from_vector(v, s);
}
inline auto FromI(const std::vector<int>& v) {
  return nn::from_vector(v);
}
inline std::string Bytes(const Tensor& t) {
  auto v = F(t);
  return {reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float)};
}
using nn::models::SaliencyCnn;
using nn::models::SaliencyScore;
using nn::models::VanillaInputGradient;
} // namespace sal
