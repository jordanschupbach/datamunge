/// @file rbf_network.cpp
/// @brief One checked native Lloyd/Gaussian/normal-system producer for independent regression comparisons.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/rbf_network.hpp>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <utility>
/// @brief Canonical fixed-Gaussian regression implementation.
namespace datamunge::algorithms {
/// @brief Private widened arithmetic and actual-training observer.
namespace {
/// @brief Owned widened normal-system matrix.
using WideMatrix = std::vector<std::vector<long double>>;
/// @brief Narrow one actual scalar without returning infinity/NaN.
/// @param x Actual widened result.
/// @return Finite double.
double narrow(long double x) {
  const auto value = static_cast<double>(x);
  if (!std::isfinite(value))
    throw std::overflow_error("RBF: finite scalar range");
  return value;
}
/// @brief Per-call actual distance/history policy and output accumulator.
struct Work {
  const RBFTrainingOptions& options;    ///< Borrowed complete policy.
  RBFTrainingRun&           run;        ///< Borrowed owned result accumulator.
  bool                      capture;    ///< Whether to copy actual scalar histories.
  std::size_t               retained{}; ///< Actual admitted retained scalar/header entries.

  /// @brief Admit a complete real scalar capture before copying.
  /// @param count Actual scalar/header entries.
  void admit(std::size_t count) {
    if (!capture)
      return;
    if (count > std::numeric_limits<std::size_t>::max() - retained ||
        (options.max_trace_entries &&
         (retained > options.max_trace_entries || count > options.max_trace_entries - retained)))
      throw std::length_error("RBF: history policy");
    retained += count;
  }
  /// @brief Count and evaluate one actual widened squared Euclidean distance.
  /// @param a Finite first point.
  /// @param b Finite matching second point.
  /// @return Actual widened squared distance.
  long double distance(const std::vector<double>& a, const std::vector<double>& b) {
    if (run.distance_tests == std::numeric_limits<std::size_t>::max() ||
        (options.max_distance_tests && run.distance_tests >= options.max_distance_tests))
      throw std::length_error("RBF: distance policy");
    ++run.distance_tests;
    long double value = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const long double delta = static_cast<long double>(a[i]) - b[i];
      value += delta * delta;
    }
    if (!std::isfinite(value))
      throw std::overflow_error("RBF: distance range");
    return value;
  }
};
/// @brief Validate every training row/target/control before placement or allocation.
/// @param X Complete observations.
/// @param y Complete targets.
/// @param ridge Nonnegative ridge coefficient.
/// @param width Finite explicit/automatic selector.
/// @param options Complete input policies.
/// @return Positive matching coordinate dimension.
std::size_t validate(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    double                                  ridge,
    double                                  width,
    const RBFTrainingOptions&               options) {
  if (X.empty() || X.size() != y.size() || X.front().empty() || !std::isfinite(ridge) ||
      ridge < 0 || !std::isfinite(width))
    throw std::invalid_argument("RBF: shape/ridge/width");
  const auto d = X.front().size();
  if ((options.max_points && X.size() > options.max_points) ||
      (options.max_dimensions && d > options.max_dimensions))
    throw std::length_error("RBF: input policy");
  for (std::size_t i = 0; i < X.size(); ++i) {
    if (X[i].size() != d || !std::isfinite(y[i]))
      throw std::invalid_argument("RBF: row/target");
    for (double x : X[i])
      if (!std::isfinite(x))
        throw std::invalid_argument("RBF: coordinate");
  }
  return d;
}
/// @brief Solve the actual widened normal system by checked partial-pivot elimination.
/// @param matrix Owned mutable normal system.
/// @param rhs Owned matching right side.
/// @return Actual finite solution coefficients.
std::vector<double> output_solve(WideMatrix matrix, std::vector<long double> rhs) {
  const auto n = rhs.size();
  for (std::size_t column = 0; column < n; ++column) {
    auto pivot = column;
    for (std::size_t row = column + 1; row < n; ++row)
      if (std::fabs(matrix[row][column]) > std::fabs(matrix[pivot][column]))
        pivot = row;
    if (matrix[pivot][column] == 0)
      throw std::domain_error("RBF: singular normal system");
    std::swap(matrix[pivot], matrix[column]);
    std::swap(rhs[pivot], rhs[column]);
    for (std::size_t row = column + 1; row < n; ++row) {
      const long double factor = matrix[row][column] / matrix[column][column];
      for (std::size_t j = column; j < n; ++j)
        matrix[row][j] -= factor * matrix[column][j];
      rhs[row] -= factor * rhs[column];
    }
  }
  std::vector<double> result(n);
  for (std::size_t i = n; i-- > 0;) {
    long double value = rhs[i];
    for (std::size_t j = i + 1; j < n; ++j)
      value -= matrix[i][j] * result[j];
    result[i] = narrow(value / matrix[i][i]);
  }
  return result;
}
/// @brief Fit the one canonical Gaussian design/normal readout after actual center placement.
/// @param X Complete validated observations.
/// @param y Finite matching targets.
/// @param ridge Finite nonnegative ridge.
/// @param width Explicit positive sigma or finite automatic selector.
/// @param work Actual count/capture accumulator with selected model centers.
void readout(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    double                                  ridge,
    double                                  width,
    Work&                                   work) {
  auto&       net = work.run.model;
  const auto  n = X.size(), k = net.centers.size(), m = k + 1;
  long double sigma = width;
  if (width <= 0) {
    long double largest = 0;
    for (std::size_t a = 0; a < k; ++a)
      for (std::size_t b = a + 1; b < k; ++b)
        largest = std::max(largest, std::sqrt(work.distance(net.centers[a], net.centers[b])));
    sigma = k > 1 && largest > 0 ? largest / std::sqrt(2.L * k) : 1;
  }
  net.width = narrow(sigma);
  if (net.width <= 0)
    throw std::overflow_error("RBF: width underflow");
  if (m > std::numeric_limits<std::size_t>::max() / m ||
      n > std::numeric_limits<std::size_t>::max() / m)
    throw std::length_error("RBF: matrix capacity");
  const auto cells = n * m + m * m;
  if (cells < n * m || (work.options.max_matrix_entries && cells > work.options.max_matrix_entries))
    throw std::length_error("RBF: matrix policy");
  std::vector<std::vector<double>> phi(n, std::vector<double>(m, 1));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < k; ++j) {
      const long double d2 = work.distance(X[i], net.centers[j]);
      phi[i][j + 1]        = narrow(std::exp(-.5L * (d2 / sigma) / sigma));
    }
  WideMatrix               normal(m, std::vector<long double>(m));
  std::vector<long double> rhs(m);
  for (std::size_t a = 0; a < m; ++a) {
    for (std::size_t b = 0; b < m; ++b) {
      long double sum = 0;
      for (std::size_t i = 0; i < n; ++i)
        sum += static_cast<long double>(phi[i][a]) * phi[i][b];
      normal[a][b] = sum + (a == b ? ridge : 0);
    }
    for (std::size_t i = 0; i < n; ++i)
      rhs[a] += static_cast<long double>(phi[i][a]) * y[i];
  }
  if (work.capture) {
    if (cells == std::numeric_limits<std::size_t>::max() ||
        m > std::numeric_limits<std::size_t>::max() - cells - 1)
      throw std::length_error("RBF: history capacity");
    work.admit(cells + m + 1);
    work.run.design = phi;
    work.run.normal_matrix.assign(m, std::vector<double>(m));
    work.run.rhs.resize(m);
    for (std::size_t i = 0; i < m; ++i) {
      work.run.rhs[i] = narrow(rhs[i]);
      for (std::size_t j = 0; j < m; ++j)
        work.run.normal_matrix[i][j] = narrow(normal[i][j]);
    }
  }
  const auto coefficients = output_solve(std::move(normal), std::move(rhs));
  net.bias                = coefficients[0];
  net.weights.assign(coefficients.begin() + 1, coefficients.end());
}
/// @brief Execute the shared native placement/readout producer with optional actual scalar capture.
/// @param X Complete observations.
/// @param y Complete finite targets.
/// @param requested Requested center count.
/// @param ridge Nonnegative finite ridge.
/// @param width Explicit/automatic finite width.
/// @param iterations Actual fixed Lloyd passes.
/// @param seed Native initial-center seed.
/// @param options Execution policies.
/// @param capture Retain actual native scalar histories.
/// @return Owned actual comparison model and selected work.
RBFTrainingRun train(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    std::size_t                             requested,
    double                                  ridge,
    double                                  width,
    std::size_t                             iterations,
    std::uint64_t                           seed,
    const RBFTrainingOptions&               options,
    bool                                    capture) {
  const auto d = validate(X, y, ridge, width, options), n = X.size(), k = std::min(requested, n);
  if ((options.max_centers && k > options.max_centers) ||
      (options.max_iterations && iterations > options.max_iterations) ||
      k > std::numeric_limits<std::size_t>::max() / d)
    throw std::length_error("RBF: center/iteration policy");
  RBFTrainingRun result;
  Work           work{options, result, capture, 0};
  auto&          centers = result.model.centers;
  centers.resize(k);
  std::mt19937_64                            rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);
  for (auto& c : centers)
    c = X[pick(rng)];
  if (capture) {
    work.admit(k * d + 1);
    result.initial_centers = centers;
  }
  if (k)
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      RBFKmeansPass pass;
      pass.iteration = iteration;
      if (capture) {
        if (k * d > (std::numeric_limits<std::size_t>::max() - n - k - 1) / 2)
          throw std::length_error("RBF: pass history capacity");
        work.admit(2 * k * d + n + k + 1);
        pass.before = centers;
        pass.assignments.reserve(n);
      }
      WideMatrix               sums(k, std::vector<long double>(d));
      std::vector<std::size_t> counts(k, 0);
      for (const auto& x : X) {
        std::size_t best     = 0;
        long double distance = std::numeric_limits<long double>::infinity();
        for (std::size_t j = 0; j < k; ++j) {
          const auto candidate = work.distance(x, centers[j]);
          if (candidate < distance) {
            distance = candidate;
            best     = j;
          }
        }
        for (std::size_t i = 0; i < d; ++i)
          sums[best][i] += x[i];
        ++counts[best];
        if (capture)
          pass.assignments.push_back(best);
      }
      for (std::size_t j = 0; j < k; ++j)
        if (counts[j])
          for (std::size_t i = 0; i < d; ++i)
            centers[j][i] = narrow(sums[j][i] / counts[j]);
      if (capture) {
        pass.counts = counts;
        pass.after  = centers;
        result.passes.push_back(std::move(pass));
      }
    }
  readout(X, y, ridge, width, work);
  return result;
}
} // namespace
RBFTrainingRun rbf_train_run(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    std::size_t                             k,
    double                                  ridge,
    double                                  width,
    std::size_t                             iterations,
    std::uint64_t                           seed,
    const RBFTrainingOptions&               o) {
  return train(X, y, k, ridge, width, iterations, seed, o, false);
}
RBFTrainingRun rbf_train_trace(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    std::size_t                             k,
    double                                  ridge,
    double                                  width,
    std::size_t                             iterations,
    std::uint64_t                           seed,
    const RBFTrainingOptions&               o) {
  return train(X, y, k, ridge, width, iterations, seed, o, true);
}
RBFNetwork rbf_train(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    std::size_t                             k,
    double                                  ridge,
    double                                  width,
    std::size_t                             iterations,
    std::uint64_t                           seed) {
  return rbf_train_run(X, y, k, ridge, width, iterations, seed).model;
}
RBFNetwork rbf_fit_output(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              y,
    const std::vector<std::vector<double>>& centers,
    double                                  width,
    double                                  ridge,
    const RBFTrainingOptions&               options) {
  const auto d = validate(X, y, ridge, width, options);
  if (width <= 0)
    throw std::invalid_argument("RBF fixed output: positive width required");
  if (options.max_centers && centers.size() > options.max_centers)
    throw std::length_error("RBF fixed output: center policy");
  for (const auto& c : centers) {
    if (c.size() != d)
      throw std::invalid_argument("RBF fixed output: center dimension");
    for (double v : c)
      if (!std::isfinite(v))
        throw std::invalid_argument("RBF fixed output: nonfinite center");
  }
  RBFTrainingRun run;
  run.model.centers = centers;
  Work work{options, run, false, 0};
  readout(X, y, ridge, width, work);
  return std::move(run.model);
}
double rbf_predict(const RBFNetwork& net, const std::vector<double>& x) {
  if (x.empty() || !std::isfinite(net.width) || net.width <= 0 || !std::isfinite(net.bias) ||
      net.weights.size() != net.centers.size())
    throw std::invalid_argument("RBF predict: model/input shape");
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("RBF predict: input");
  for (std::size_t j = 0; j < net.centers.size(); ++j) {
    if (net.centers[j].size() != x.size() || !std::isfinite(net.weights[j]))
      throw std::invalid_argument("RBF predict: center/weight shape");
    for (double v : net.centers[j])
      if (!std::isfinite(v))
        throw std::invalid_argument("RBF predict: center");
  }
  RBFTrainingOptions o;
  o.max_distance_tests = 0;
  RBFTrainingRun run;
  Work           work{o, run, false, 0};
  long double    value = net.bias;
  for (std::size_t j = 0; j < net.centers.size(); ++j)
    value += net.weights[j] *
             std::exp(-.5L * (work.distance(x, net.centers[j]) / net.width) / net.width);
  return narrow(value);
}
RBFExampleData rbf_example_data(std::uint64_t seed) {
  RBFExampleData                   data;
  std::mt19937_64                  rng(seed);
  std::normal_distribution<double> noise(0, .15);
  for (int i = 0; i <= 60; ++i) {
    const double x = 2 * std::numbers::pi * i / 60;
    data.input.push_back({x});
    data.target.push_back(std::sin(x) + std::sin(3 * x) / 3 + noise(rng));
  }
  for (int i = 0; i <= 200; ++i) {
    const double x = 2 * std::numbers::pi * i / 200;
    data.grid.push_back(x);
    data.truth.push_back(std::sin(x) + std::sin(3 * x) / 3);
  }
  return data;
}
} // namespace datamunge::algorithms
