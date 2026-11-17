/// @file rvm.cpp
/// @brief Single checked fixed-point producer using canonical widened Cholesky.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/rvm.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native algorithm producers and owned outputs.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Widened dense row storage for model algebra.
using Matrix = std::vector<std::vector<long double>>;
/// @brief Require finite representable public output.
/// @param x Widened computed value.
/// @return Finite double representation.
double narrow(long double x) {
  if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
    throw std::domain_error("RVM: numerical range");
  return static_cast<double>(x);
}
/// @brief Evaluate one finite Gaussian basis using scaled distance terms.
/// @param a First finite input.
/// @param b Second finite input.
/// @param gamma Positive inverse squared-distance scale.
/// @return Actual finite Gaussian kernel; very distant points yield zero.
long double kernel(const std::vector<double>& a, const std::vector<double>& b, double gamma) {
  long double exponent = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double d    = static_cast<long double>(a[i]) - b[i];
    const auto        term = std::sqrt(static_cast<long double>(gamma)) * std::abs(d);
    if (!std::isfinite(term) || term > std::sqrt(std::numeric_limits<long double>::max()))
      return 0;
    exponent += term * term;
    if (!std::isfinite(exponent))
      return 0;
  }
  return std::exp(-exponent);
}
/// @brief Validate complete matrix/data finite geometry.
/// @param x Borrowed inputs.
/// @param t Borrowed targets.
/// @param o Controls and policies.
void validate(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>&              t,
    const RVMOptions&                       o) {
  if (x.empty() || x.front().empty() || t.size() != x.size() || !std::isfinite(o.gamma) ||
      o.gamma <= 0 || !std::isfinite(o.tolerance) || o.tolerance < 0 ||
      !std::isfinite(o.prune_threshold) || o.prune_threshold <= 1 ||
      !std::isfinite(o.prune_threshold * 10))
    throw std::invalid_argument("RVM: inputs/controls");
  const auto n = x.size(), max = std::numeric_limits<std::size_t>::max();
  if ((o.max_samples && n > o.max_samples) || n == max || n + 1 > max / (n + 1) ||
      (o.max_matrix_entries && (n + 1) * (n + 1) > o.max_matrix_entries) ||
      (o.max_updates && o.max_iterations > o.max_updates) || x.front().size() > max / n ||
      (o.max_matrix_entries && n * x.front().size() > o.max_matrix_entries))
    throw std::length_error("RVM: complete geometry/update policy");
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i].size() != x.front().size() || !std::isfinite(t[i]))
      throw std::invalid_argument("RVM: row/target shape");
    for (auto v : x[i])
      if (!std::isfinite(v))
        throw std::invalid_argument("RVM: finite inputs");
  }
}
/// @brief Actual solved mean and covariance for selected basis columns.
struct Posterior {
  std::vector<long double> mean; ///< Actual posterior means.
  Matrix covariance;             ///< Actual covariance from canonical unit-right-hand-side solves.
};
/// @brief Factor and solve one actual selected-column posterior.
/// @param gram Complete widened Gram matrix.
/// @param rhs Complete widened Phi-transpose-target product.
/// @param alpha Complete current prior precisions.
/// @param beta Current noise precision.
/// @param active Actual selected full-basis column indices.
/// @return Actual selected posterior.
Posterior posterior(
    const Matrix&                   gram,
    const std::vector<long double>& rhs,
    const std::vector<long double>& alpha,
    long double                     beta,
    const std::vector<std::size_t>& active) {
  const auto                       m = active.size();
  linalg::DenseMatrix<long double> precision(m, m, 0.L);
  std::vector<long double>         b(m);
  for (std::size_t i = 0; i < m; ++i) {
    b[i] = beta * rhs[active[i]];
    if (!std::isfinite(b[i]))
      throw std::domain_error("RVM: rhs range");
    for (std::size_t j = 0; j < m; ++j) {
      precision(i, j) = beta * gram[active[i]][active[j]] + (i == j ? alpha[active[i]] : 0);
      if (!std::isfinite(precision(i, j)))
        throw std::domain_error("RVM: precision range");
    }
  }
  const auto factor = linalg::cholesky(precision);
  if (!factor.ok)
    throw std::domain_error("RVM: non-SPD posterior");
  Posterior p;
  p.mean = factor.solve(b);
  p.covariance.assign(m, std::vector<long double>(m));
  for (std::size_t j = 0; j < m; ++j) {
    std::vector<long double> unit(m);
    unit[j]     = 1;
    auto column = factor.solve(unit);
    for (std::size_t i = 0; i < m; ++i)
      p.covariance[i][j] = column[i];
  }
  return p;
}
/// @brief Execute actual full-basis updates and a final selected-basis matched refit.
/// @param x Borrowed validated training inputs.
/// @param t Borrowed targets.
/// @param o Controls/policies.
/// @param capture Retain actual updates.
/// @return Owned actual fitted run.
RVMRun train(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>&              t,
    const RVMOptions&                       o,
    bool                                    capture) {
  validate(x, t, o);
  const auto n = x.size(), m = n + 1;
  RVMRun     out;
  out.model.centers = x;
  out.model.gamma   = o.gamma;
  Matrix phi(n, std::vector<long double>(m, 1)), gram(m, std::vector<long double>(m));
  std::vector<long double> rhs(m), alpha(m, 1);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      phi[i][j + 1] = kernel(x[i], x[j], o.gamma);
  for (std::size_t a = 0; a < m; ++a) {
    for (std::size_t i = 0; i < n; ++i)
      rhs[a] += phi[i][a] * t[i];
    for (std::size_t b = 0; b < m; ++b)
      for (std::size_t i = 0; i < n; ++i)
        gram[a][b] += phi[i][a] * phi[i][b];
  }
  long double mean = 0;
  for (auto v : t)
    mean += static_cast<long double>(v) / n;
  long double variance = 0;
  for (auto v : t) {
    const auto d = static_cast<long double>(v) - mean;
    variance += d * d;
  }
  variance /= n > 1 ? n - 1 : 1;
  long double beta = 1 / (0.1L * variance + 1e-9L);
  if (!(beta > 0) || !std::isfinite(beta))
    throw std::domain_error("RVM: initial precision range");
  std::vector<std::size_t> all(m);
  std::iota(all.begin(), all.end(), 0);
  std::size_t retained = 0;
  for (std::size_t it = 0; it < o.max_iterations; ++it) {
    if (capture) {
      if (m > (std::numeric_limits<std::size_t>::max() - 1) / 5)
        throw std::length_error("RVM: capture geometry");
      const auto count = 1 + 5 * m;
      if (count > std::numeric_limits<std::size_t>::max() - retained ||
          (o.max_trace_entries &&
           (retained > o.max_trace_entries || count > o.max_trace_entries - retained)))
        throw std::length_error("RVM: trace policy");
      retained += count;
    }
    const auto p = posterior(gram, rhs, alpha, beta, all);
    ++out.posterior_solves;
    RVMStep step;
    step.iteration = it + 1;
    if (capture) {
      step.beta_before = narrow(beta);
      for (auto a : alpha)
        step.alpha_before.push_back(narrow(a));
    }
    long double change = 0, sum = 0;
    for (std::size_t j = 0; j < m; ++j) {
      auto g = 1 - alpha[j] * p.covariance[j][j];
      if (!std::isfinite(g) || g < -1e-7L || g > 1 + 1e-7L)
        throw std::domain_error("RVM: effective parameter range");
      g = std::clamp(g, 0.L, 1.L);
      sum += g;
      auto next = p.mean[j] * p.mean[j] > 1e-300L
                      ? g / (p.mean[j] * p.mean[j])
                      : static_cast<long double>(o.prune_threshold) * 10;
      if (!(next > 0) || !std::isfinite(next) || next > std::numeric_limits<double>::max())
        next = static_cast<long double>(o.prune_threshold) * 10;
      change = std::max(change, std::abs(std::log(next + 1e-300L) - std::log(alpha[j] + 1e-300L)));
      alpha[j] = next;
      if (capture) {
        step.mean.push_back(narrow(p.mean[j]));
        step.covariance_diagonal.push_back(narrow(p.covariance[j][j]));
        step.effective_parameters.push_back(narrow(g));
        step.alpha_after.push_back(narrow(next));
      }
    }
    long double error = 0;
    for (std::size_t i = 0; i < n; ++i) {
      long double prediction = 0;
      for (std::size_t j = 0; j < m; ++j)
        prediction += phi[i][j] * p.mean[j];
      const auto d = static_cast<long double>(t[i]) - prediction;
      error += d * d;
    }
    beta = (static_cast<long double>(n) - sum) / (error + 1e-12L);
    if (!(beta > 0) || !std::isfinite(beta) || beta > std::numeric_limits<double>::max())
      beta = 1 / (variance + 1e-9L);
    out.model.iterations = it + 1;
    if (capture) {
      step.beta_after         = narrow(beta);
      step.squared_error      = narrow(error);
      step.maximum_log_change = narrow(change);
      out.steps.push_back(std::move(step));
    }
    if (change < o.tolerance && it > 1) {
      out.converged = true;
      break;
    }
  }
  out.model.relevant.assign(n, 0);
  std::vector<std::size_t> active{0};
  for (std::size_t j = 0; j < n; ++j)
    if (alpha[j + 1] < o.prune_threshold) {
      out.model.relevant[j] = 1;
      active.push_back(j + 1);
    }
  out.model.num_relevance_vectors = active.size() - 1;
  out.model.beta                  = narrow(beta);
  if(!(out.model.beta>0))throw std::domain_error("RVM: reported noise precision underflow");
  for (auto a : alpha)
    {const auto value=narrow(a);if(!(value>0))throw std::domain_error("RVM: reported prior precision underflow");out.alpha.push_back(value);}
  // Round reported hyperparameters before final solve so covariance and mean use exactly their published values.
  for (std::size_t j = 0; j < m; ++j)
    alpha[j] = out.alpha[j];
  beta             = out.model.beta;
  const auto final = posterior(gram, rhs, alpha, beta, active);
  ++out.posterior_solves;
  out.model.weights.assign(m, 0);
  out.covariance.assign(m, std::vector<double>(m));
  for (std::size_t i = 0; i < active.size(); ++i) {
    out.model.weights[active[i]] = narrow(final.mean[i]);
    for (std::size_t j = 0; j < active.size(); ++j)
      out.covariance[active[i]][active[j]] =
          narrow((final.covariance[i][j] + final.covariance[j][i]) / 2);
  }
  return out;
}
/// @brief Validate a mutable public mean-model and produce its active basis vector.
/// @param m Borrowed mean model.
/// @param x Borrowed evaluation input.
/// @return Owned bias-first active Gaussian basis vector.
std::vector<long double> features(const RVMModel& m, const std::vector<double>& x) {
  if (m.centers.empty() || m.centers.front().empty() || x.size() != m.centers.front().size() ||
      m.weights.size() != m.centers.size() + 1 || m.relevant.size() != m.centers.size() ||
      !std::isfinite(m.gamma) || m.gamma <= 0 || !std::isfinite(m.beta) || m.beta <= 0)
    throw std::invalid_argument("RVM: model/input shape");
  for (auto v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("RVM: input finite");
  for (auto v : m.weights)
    if (!std::isfinite(v))
      throw std::invalid_argument("RVM: weights finite");
  std::size_t              count = 0;
  std::vector<long double> f(m.weights.size());
  f[0] = 1;
  for (std::size_t j = 0; j < m.centers.size(); ++j) {
    if (m.centers[j].size() != x.size() || (m.relevant[j] != 0 && m.relevant[j] != 1) ||
        (!m.relevant[j] && m.weights[j + 1] != 0))
      throw std::invalid_argument("RVM: center/activity shape");
    for (auto v : m.centers[j])
      if (!std::isfinite(v))
        throw std::invalid_argument("RVM: center finite");
    if (m.relevant[j]) {
      ++count;
      f[j + 1] = kernel(x, m.centers[j], m.gamma);
    }
  }
  if (count != m.num_relevance_vectors)
    throw std::invalid_argument("RVM: active count");
  return f;
}
} // namespace
/// @brief Delegate the checked uncaptured native fixed-point fit.
/// @param x Borrowed training inputs.
/// @param t Borrowed targets.
/// @param o Complete controls and policies.
/// @return Actual final matched posterior.
RVMRun rvm_train_run(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>&              t,
    const RVMOptions&                       o) {
  return train(x, t, o, false);
}
RVMRun rvm_train_trace(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>&              t,
    const RVMOptions&                       o) {
  return train(x, t, o, true);
}
RVMModel rvm_train(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>&              t,
    double                                  gamma,
    std::size_t                             iterations,
    double                                  threshold,
    double                                  tolerance) {
  RVMOptions o;
  o.gamma           = gamma;
  o.max_iterations  = iterations;
  o.prune_threshold = threshold;
  o.tolerance       = tolerance;
  return rvm_train_run(x, t, o).model;
}
double rvm_predict(const RVMModel& m, const std::vector<double>& x) {
  auto        f = features(m, x);
  long double y = 0;
  for (std::size_t j = 0; j < f.size(); ++j)
    y += f[j] * m.weights[j];
  return narrow(y);
}
RVMPrediction rvm_predictive(const RVMRun& r, const std::vector<double>& x) {
  auto       f = features(r.model, x);
  const auto n = f.size();
  if (r.covariance.size() != n)
    throw std::invalid_argument("RVM: covariance shape");
  for (const auto& row : r.covariance)
    if (row.size() != n)
      throw std::invalid_argument("RVM: covariance row shape");
  long double variance = 0, absolute = 0, mean = 0;
  for (std::size_t i = 0; i < n; ++i) {
    mean += f[i] * r.model.weights[i];
    for (std::size_t j = 0; j < n; ++j) {
      const auto c = r.covariance[i][j];
      if (!std::isfinite(c) || c != r.covariance[j][i] ||
          ((!f[i] || !f[j]) &&
           ((i && !r.model.relevant[i - 1]) || (j && !r.model.relevant[j - 1])) && c != 0))
        throw std::invalid_argument("RVM: covariance values");
      const auto term = f[i] * c * f[j];
      variance += term;
      absolute += std::abs(term);
    }
  }
  if (variance < -1e-12L * std::max(1.L, absolute))
    throw std::domain_error("RVM: negative variance");
  variance         = std::max(0.L, variance);
  const auto noise = 1 / static_cast<long double>(r.model.beta);
  return {narrow(mean), narrow(variance), narrow(noise), narrow(variance + noise)};
}
RVMExample rvm_sine_example() {
  RVMExample e;
  for (int i = 0; i <= 40; ++i) {
    const auto x = 2 * std::numbers::pi * i / 40;
    e.inputs.push_back({x});
    e.targets.push_back(std::sin(x));
  }
  for (int i = 0; i <= 200; ++i) {
    const auto x = 2 * std::numbers::pi * i / 200;
    e.grid.push_back({x});
    e.truth.push_back(std::sin(x));
  }
  return e;
}
} // namespace datamunge::algorithms
