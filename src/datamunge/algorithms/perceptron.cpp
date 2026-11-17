/// @file perceptron.cpp
/// @brief Canonical producing implementation and actual-site diagnostics; public contracts are in the paired header.
#include <cmath>
#include <datamunge/algorithms/perceptron.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief File-local validated kernels shared by ordinary and observed entry points.
namespace {
/// @brief Reject nonfinite double arithmetic without silently certifying a fit.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("perceptron arithmetic overflow");
  return x;
}
/// @brief Original ordered additive kernel with validation and actual-site observer.
PerceptronTrace fit(
    const std::vector<std::vector<double>>& x,
    const std::vector<int>&                 y,
    std::size_t                             epochs,
    double                                  rate,
    bool                                    capture,
    std::size_t                             cap) {
  if (x.size() != y.size() || !std::isfinite(rate) || rate <= 0)
    throw std::invalid_argument("perceptron shapes/rate");
  const auto d = x.empty() ? 0 : x.front().size();
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (x[i].size() != d || (y[i] != -1 && y[i] != 1))
      throw std::invalid_argument("perceptron shape/label");
    for (double v : x[i])
      if (!std::isfinite(v))
        throw std::invalid_argument("perceptron nonfinite feature");
  }
  PerceptronTrace t;
  auto&           m = t.model;
  if (x.empty())
    return t;
  m.weights.assign(d, 0);
  std::size_t retained = 0;
  for (std::size_t e = 0; e < epochs; ++e) {
    bool mistake = false;
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (capture) {
        if (d == std::numeric_limits<std::size_t>::max() ||
            d + 1 > std::numeric_limits<std::size_t>::max() - retained ||
            (cap && (retained > cap || d + 1 > cap - retained)))
          throw std::length_error("perceptron history cap");
        retained += d + 1;
      }
      double score = m.bias;
      for (std::size_t j = 0; j < d; ++j)
        score = finite(score + finite(m.weights[j] * x[i][j]));
      const bool update = y[i] * score <= 0;
      if (update) {
        if (m.total_mistakes == std::numeric_limits<std::size_t>::max())
          throw std::overflow_error("perceptron update count");
        for (std::size_t j = 0; j < d; ++j)
          m.weights[j] = finite(m.weights[j] + finite(rate * y[i] * x[i][j]));
        m.bias = finite(m.bias + rate * y[i]);
        ++m.total_mistakes;
        mistake = true;
      }
      if (capture)
        t.visits.push_back({e + 1, i, score, update, m.weights, m.bias});
    }
    m.epochs_run = e + 1;
    if (!mistake) {
      m.converged = true;
      break;
    }
  }
  return t;
}
} // namespace
PerceptronModel perceptron_train(
    const std::vector<std::vector<double>>& x,
    const std::vector<int>&                 y,
    std::size_t                             e,
    double                                  r) {
  return fit(x, y, e, r, false, 0).model;
}
PerceptronTrace perceptron_trace(
    const std::vector<std::vector<double>>& x,
    const std::vector<int>&                 y,
    std::size_t                             e,
    double                                  r,
    std::size_t                             cap) {
  return fit(x, y, e, r, true, cap);
}
int perceptron_predict(const PerceptronModel& m, const std::vector<double>& x) {
  if (x.size() != m.weights.size() || !std::isfinite(m.bias))
    throw std::invalid_argument("perceptron prediction shape/bias");
  for (std::size_t j = 0; j < x.size(); ++j)
    if (!std::isfinite(x[j]) || !std::isfinite(m.weights[j]))
      throw std::invalid_argument("perceptron prediction finite values");
  double s = m.bias;
  for (std::size_t j = 0; j < x.size(); ++j)
    s = finite(s + finite(m.weights[j] * x[j]));
  return s > 0 ? 1 : -1;
}
} // namespace datamunge::algorithms
