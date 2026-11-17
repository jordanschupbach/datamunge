#pragma once
// Independent numerical and architecture regression checks retained from the
// original executable; canonical operations are called directly. No training
// driver or example-source dependency is retained here.
#include <dm/dm.hpp>
#include <iostream>
namespace {
namespace nn = dm::nn;
using nn::Padding;
using nn::Tensor;
namespace check {
template<class F>
std::string Throws(F&& f, const std::string& message) {
  try {
    f();
  } catch (const std::exception& e) {
    return e.what();
  }
  throw std::runtime_error(message + ": expected exception");
}
inline void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}
inline void Near(double a, double b, double tolerance, const std::string& message) {
  if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance)
    throw std::runtime_error(message);
}
} // namespace check
namespace sentence {
using namespace dm::nn::models;
using Weights = std::vector<std::vector<double>>;
using Rows    = std::vector<std::int64_t>;
inline Rows Range(std::int64_t begin, std::int64_t end) {
  Rows result(end - begin);
  std::iota(result.begin(), result.end(), begin);
  return result;
}
inline auto SentenceCE(const nn::Tensor& z, const nn::Tensor& y) {
  return SequenceCrossEntropy(z, y);
}
inline Weights FixParameters(
    nn::ParameterStore&                                                      p,
    const std::function<float(std::size_t, std::int64_t, const nn::Shape&)>& value) {
  Weights weights;
  for (std::size_t v = 0; v < p.size(); ++v) {
    const auto         shape = p.parameters()[v].shape();
    std::vector<float> data(static_cast<std::size_t>(nn::shape_numel(shape)));
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = value(v, static_cast<std::int64_t>(i), shape);
    p.set(p.names()[v], nn::cast(nn::from_vector(data, shape), p.dtype()));
    weights.emplace_back(data.begin(), data.end());
  }
  return weights;
}

/// max |central difference of `objective` (step 1e-5) - gradient| over every
/// coordinate of `weights`; `coordinates` receives the count.
inline double HostGradientError(
    const Weights&                               weights,
    const std::vector<Tensor>&                   gradients,
    const std::function<double(const Weights&)>& objective,
    int&                                         coordinates) {
  double error = 0;
  coordinates  = 0;
  for (std::size_t v = 0; v < weights.size(); ++v) {
    const auto g = gradients[v].to_vector<double>();
    for (std::size_t i = 0; i < weights[v].size(); ++i) {
      auto plus = weights, minus = weights;
      plus[v][i] += 1e-5;
      minus[v][i] -= 1e-5;
      error = std::max(error, std::abs((objective(plus) - objective(minus)) / 2e-5 - g[i]));
      ++coordinates;
    }
  }
  return error;
}

/// Sum of squares of each gradient.
inline std::vector<double> SquaredNorms(const std::vector<Tensor>& gradients) {
  std::vector<double> out;
  for (const auto& g : gradients) {
    double s = 0;
    for (double x : g.to_vector<double>())
      s += x * x;
    out.push_back(s);
  }
  return out;
}

} // namespace sentence
using sentence::Weights;
namespace charcnn = dm::nn::models;
namespace sensor  = dm::nn::models;
namespace series  = dm::nn::models;
namespace tcn     = dm::nn::models;
namespace wavestack {
using namespace dm::nn::models;
inline int MuLawEncode(double x) {
  return nn::mu_law_encode(nn::full({}, x, nn::DType::Float64)).item<int>();
}
inline double MuLawDecode(int x) {
  return nn::mu_law_decode(nn::full({}, x, nn::DType::Int32), 256, nn::DType::Float64).item();
}
} // namespace wavestack
} // namespace
