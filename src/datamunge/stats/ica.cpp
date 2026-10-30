/// @file ica.cpp
/// @brief Population-whitened deflationary FastICA, inspectable updates and benchmark matching.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/hungarian.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/ica.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Private finite arithmetic, contrast and deflation helpers.
namespace {
/// @brief Recycled colours for lexically ordered external plot groups.
const graphics::RGB kSeriesColors[] =
    {{37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178}};
/// @brief Copies sorted distinct group labels.
/// @param labels Borrowed external labels.
/// @return Owned lexically ordered unique labels.
std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
  auto out = labels;
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}
/// @brief Converts extended arithmetic to a representable finite double.
/// @param x Extended intermediate, permitting underflow to zero.
/// @return Finite double.
/// @throws std::invalid_argument Nonfinite or unrepresentable result.
double finite_ica(long double x) {
  const double y = static_cast<double>(x);
  if (!std::isfinite(y))
    throw std::invalid_argument("ICA: nonfinite/unrepresentable arithmetic or retained value");
  return y;
}
/// @brief Verifies all borrowed matrix entries are finite.
/// @param x Matrix to inspect, unchanged.
/// @throws std::invalid_argument Nonfinite entry.
void check_matrix(const linalg::DenseMatrix<double>& x) {
  for (std::size_t i = 0; i < x.rows(); ++i)
    for (std::size_t j = 0; j < x.cols(); ++j)
      (void)finite_ica(x(i, j));
}
/// @brief A contrast's g and g-prime values.
struct ContrastValue {
  double g;  ///< First derivative of G.
  double gp; ///< Derivative of g.
};
/// @brief Function pointer for one scalar contrast derivative pair.
using Contrast = ContrastValue (*)(double);
/// @brief Bounded tanh derivative pair for logcosh G.
/// @param u Finite projection.
/// @return tanh(u),1-tanh(u)^2.
ContrastValue logcosh_contrast(double u) {
  const double t = std::tanh(u);
  return {t, 1 - t * t};
}
/// @brief Exponential contrast derivative pair with stable far-tail zero limits.
/// @param u Finite projection.
/// @return u*exp(-u^2/2),(1-u^2)*exp(-u^2/2), or their zero tail limits.
ContrastValue exp_contrast(double u) {
  if (std::abs(u) > 40)
    return {0, 0};
  const double v = u * u, e = std::exp(-.5 * v);
  return {u * e, (1 - v) * e};
}
/// @brief Cubic contrast derivative pair with finite arithmetic checks.
/// @param u Finite projection.
/// @return u^3,3*u^2.
/// @throws std::invalid_argument Unrepresentable cubic/quadratic output.
ContrastValue cube_contrast(double u) {
  return {finite_ica(static_cast<long double>(u) * u * u), finite_ica(3.L * u * u)};
}
/// @brief Resolves a public contrast name.
/// @param name logcosh, exp or cube.
/// @return Private scalar derivative function.
/// @throws std::invalid_argument Unknown contrast.
Contrast choose_contrast(const std::string& name) {
  if (name == "logcosh")
    return logcosh_contrast;
  if (name == "exp")
    return exp_contrast;
  if (name == "cube")
    return cube_contrast;
  throw std::invalid_argument("ICA: contrast must be logcosh, exp or cube");
}
/// @brief Extended dot product of equal-sized vectors.
/// @param a First vector.
/// @param b Equally sized second vector.
/// @return Finite dot product.
/// @throws std::invalid_argument Nonfinite intermediate/result.
double dot_ica(const std::vector<double>& a, const std::vector<double>& b) {
  long double s = 0;
  for (std::size_t j = 0; j < a.size(); ++j)
    s += static_cast<long double>(a[j]) * b[j];
  return finite_ica(s);
}
/// @brief Projects twice off previously extracted orthonormal rows and normalizes.
/// @param w Owned candidate direction, modified.
/// @param previous Borrowed prior row vectors.
/// @return Norm before normalization, strictly positive.
/// @throws std::runtime_error Zero projected direction.
/// @throws std::invalid_argument Nonfinite arithmetic.
double project_normalize(std::vector<double>& w, const std::vector<std::vector<double>>& previous) {
  for (int pass = 0; pass < 2; ++pass)
    for (const auto& old : previous) {
      const double dot = dot_ica(w, old);
      for (std::size_t j = 0; j < w.size(); ++j)
        w[j] = finite_ica(static_cast<long double>(w[j]) - static_cast<long double>(dot) * old[j]);
    }
  double norm = 0;
  for (double v : w)
    norm = std::hypot(norm, v);
  norm = finite_ica(norm);
  if (norm == 0)
    throw std::runtime_error("ICA: zero/degenerate deflated direction");
  for (auto& v : w)
    v /= norm;
  return norm;
}
/// @brief Performs one already-validated population-expectation update.
/// @param z Whitened rows.
/// @param w Unit current direction.
/// @param contrast Private scalar contrast.
/// @param previous Extracted orthonormal row directions.
/// @return Raw/normalized update and angle evidence.
/// @throws std::runtime_error Zero projected update.
/// @throws std::invalid_argument Nonfinite arithmetic.
ICAFixedPointStep step_ica(
    const linalg::DenseMatrix<double>&      z,
    const std::vector<double>&              w,
    Contrast                                contrast,
    const std::vector<std::vector<double>>& previous) {
  const auto               n = z.rows(), k = z.cols();
  std::vector<long double> mean(k);
  long double              gp = 0;
  for (std::size_t i = 0; i < n; ++i) {
    long double projection = 0;
    for (std::size_t j = 0; j < k; ++j)
      projection += static_cast<long double>(z(i, j)) * w[j];
    const auto v = contrast(finite_ica(projection));
    gp += v.gp;
    for (std::size_t j = 0; j < k; ++j)
      mean[j] += static_cast<long double>(z(i, j)) * v.g;
  }
  std::vector<double> raw(k);
  for (std::size_t j = 0; j < k; ++j)
    raw[j] = finite_ica((mean[j] - gp * w[j]) / n);
  // Keep the projected vector for inspection; normalize a separate copy.
  for (int pass = 0; pass < 2; ++pass)
    for (const auto& old : previous) {
      const double dot = dot_ica(raw, old);
      for (std::size_t j = 0; j < k; ++j)
        raw[j] =
            finite_ica(static_cast<long double>(raw[j]) - static_cast<long double>(dot) * old[j]);
    }
  auto         direction = raw;
  const double norm      = project_normalize(direction, {});
  return {raw, direction, norm, std::max(0., 1 - std::abs(dot_ica(direction, w)))};
}
/// @brief Maximum absolute population covariance deviation from identity.
/// @param x Borrowed zero-mean row matrix with positive row count.
/// @return Maximum absolute entry error in X^T*X/n-I.
double covariance_error(const linalg::DenseMatrix<double>& x) {
  double error = 0;
  for (std::size_t a = 0; a < x.cols(); ++a)
    for (std::size_t b = 0; b < x.cols(); ++b) {
      long double s = 0;
      for (std::size_t i = 0; i < x.rows(); ++i)
        s += static_cast<long double>(x(i, a)) * x(i, b);
      error = std::max(error, std::abs(finite_ica(s / x.rows()) - (a == b ? 1. : 0.)));
    }
  return error;
}
} // namespace

ICAFixedPointStep ica_fixed_point_step(
    const linalg::DenseMatrix<double>& z,
    const std::vector<double>&         w,
    const std::string&                 name,
    const linalg::DenseMatrix<double>& previous) {
  if (z.rows() < 2 || z.cols() == 0 || w.size() != z.cols() || previous.rows() >= z.cols() ||
      (previous.rows() && previous.cols() != z.cols()))
    throw std::invalid_argument("ICA step: invalid dimensions");
  check_matrix(z);
  check_matrix(previous);
  for (double v : w)
    (void)finite_ica(v);
  if (std::abs(dot_ica(w, w) - 1) > 1e-8)
    throw std::invalid_argument("ICA step: w must have unit norm");
  std::vector<std::vector<double>> rows;
  for (std::size_t i = 0; i < previous.rows(); ++i) {
    auto row = previous.row(i);
    if (std::abs(dot_ica(row, row) - 1) > 1e-8 || std::abs(dot_ica(row, w)) > 1e-8)
      throw std::invalid_argument("ICA step: invalid prior directions");
    for (const auto& old : rows)
      if (std::abs(dot_ica(row, old)) > 1e-8)
        throw std::invalid_argument("ICA step: prior directions not orthogonal");
    rows.push_back(row);
  }
  return step_ica(z, w, choose_contrast(name), rows);
}
ICA::ICA(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& columns,
    ICAOptions                      options)
    : feature_columns_(columns), options_(std::move(options)) {
  fit(data);
}
void ICA::fit(const dstruct::DataFrame& data) {
  if (feature_columns_.empty() || options_.n_components == 0 || options_.max_iter == 0 ||
      !std::isfinite(options_.tol) || options_.tol <= 0 || options_.tol >= 1)
    throw std::invalid_argument(
        "ICA: require features, positive components/iterations and finite tol in (0,1)");
  const auto contrast = choose_contrast(options_.contrast);
  auto       names    = feature_columns_;
  std::sort(names.begin(), names.end());
  if (std::adjacent_find(names.begin(), names.end()) != names.end())
    throw std::invalid_argument("ICA: duplicate feature columns");
  for (const auto& col : feature_columns_)
    if (!data.has_column(col) || data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("ICA: missing or non-numeric feature '" + col + "'");
  kept_row_indices_.clear();
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    bool complete = true;
    for (const auto& col : feature_columns_)
      if (data.is_null(col, i)) {
        complete = false;
        break;
      }
    if (complete)
      kept_row_indices_.push_back(i);
  }
  const auto n = kept_row_indices_.size(), p = feature_columns_.size(), k = options_.n_components;
  if (n < 2 || k > p || k >= n)
    throw std::invalid_argument(
        "ICA: need at least two complete rows and components<=min(features,rows-1)");
  observations_ = n;
  mean_.assign(p, 0);
  centered_ = linalg::DenseMatrix<double>(n, p);
  for (std::size_t j = 0; j < p; ++j) {
    long double sum = 0;
    for (auto i : kept_row_indices_)
      sum += finite_ica(data.double_at(feature_columns_[j], i));
    mean_[j] = finite_ica(sum / n);
    for (std::size_t i = 0; i < n; ++i)
      centered_(i, j) = finite_ica(
          static_cast<long double>(data.double_at(feature_columns_[j], kept_row_indices_[i])) -
          mean_[j]);
  }
  double scale = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < p; ++j)
      scale = std::max(scale, std::abs(centered_(i, j)));
  if (scale == 0)
    throw std::invalid_argument("ICA: no nonconstant features");
  linalg::DenseMatrix<double> cov(p, p);
  for (std::size_t a = 0; a < p; ++a)
    for (std::size_t b = a; b < p; ++b) {
      long double sum = 0;
      for (std::size_t i = 0; i < n; ++i)
        sum += (static_cast<long double>(centered_(i, a)) / scale) *
               (static_cast<long double>(centered_(i, b)) / scale);
      cov(a, b) = cov(b, a) = finite_ica(sum / n);
    }
  const auto   eig = linalg::jacobi_eigen(cov);
  const double cut = std::max(n, p) * std::numeric_limits<double>::epsilon() * eig.eigenvalues[0];
  if (!(eig.eigenvalues[k - 1] > cut))
    throw std::invalid_argument("ICA: insufficient numerical covariance rank");
  long double total_eigen = 0, retained_eigen = 0;
  for (std::size_t j = 0; j < p; ++j) {
    total_eigen += std::max(0., eig.eigenvalues[j]);
    if (j < k)
      retained_eigen += eig.eigenvalues[j];
  }
  retained_variance_fraction_ = finite_ica(retained_eigen / total_eigen);
  eigenvalues_.clear();
  for (auto v : eig.eigenvalues)
    eigenvalues_.push_back(finite_ica(static_cast<long double>(v) * scale * scale));
  rank_threshold_ = finite_ica(static_cast<long double>(cut) * scale * scale);
  whitening_      = linalg::DenseMatrix<double>(p, k);
  for (std::size_t j = 0; j < p; ++j)
    for (std::size_t c = 0; c < k; ++c)
      whitening_(j, c) = finite_ica(
          static_cast<long double>(eig.eigenvectors(j, c)) /
          (static_cast<long double>(scale) * std::sqrt(eig.eigenvalues[c])));
  whitened_ = linalg::DenseMatrix<double>(n, k);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t c = 0; c < k; ++c) {
      long double v = 0;
      for (std::size_t j = 0; j < p; ++j)
        v += static_cast<long double>(centered_(i, j)) * whitening_(j, c);
      whitened_(i, c) = finite_ica(v);
    }
  std::mt19937_64                  rng(options_.seed);
  std::normal_distribution<double> normal(0, 1);
  std::vector<std::vector<double>> W;
  initial_ = linalg::DenseMatrix<double>(k, k);
  iterations_.assign(k, 0);
  alignment_errors_.assign(k, 0);
  component_converged_.assign(k, false);
  converged_ = true;
  std::vector<double>              hc, hi, he, hn;
  std::vector<std::vector<double>> hw(k);
  for (std::size_t c = 0; c < k; ++c) {
    std::vector<double> w(k);
    for (auto& v : w)
      v = normal(rng);
    (void)project_normalize(w, W);
    for (std::size_t j = 0; j < k; ++j)
      initial_(c, j) = w[j];
    for (std::size_t it = 1; it <= options_.max_iter; ++it) {
      const auto step      = step_ica(whitened_, w, contrast, W);
      w                    = step.direction;
      iterations_[c]       = it;
      alignment_errors_[c] = step.alignment_error;
      hc.push_back(c + 1);
      hi.push_back(it);
      he.push_back(step.alignment_error);
      hn.push_back(step.update_norm);
      for (std::size_t j = 0; j < k; ++j)
        hw[j].push_back(w[j]);
      if (step.alignment_error < options_.tol) {
        component_converged_[c] = true;
        break;
      }
    }
    converged_ = converged_ && component_converged_[c];
    W.push_back(w);
  }
  history_ = dstruct::DataFrame{};
  history_.add_column("component", hc);
  history_.add_column("iteration", hi);
  history_.add_column("alignment_error", he);
  history_.add_column("update_norm", hn);
  for (std::size_t j = 0; j < k; ++j)
    history_.add_column("w" + std::to_string(j), hw[j]);
  sources_  = linalg::DenseMatrix<double>(n, k);
  rotation_ = linalg::DenseMatrix<double>(k, k);
  source_kurtosis_.assign(k, 0);
  for (std::size_t c = 0; c < k; ++c) {
    double biggest = 0, at = 0;
    for (std::size_t i = 0; i < n; ++i) {
      long double value = 0;
      for (std::size_t j = 0; j < k; ++j)
        value += static_cast<long double>(whitened_(i, j)) * W[c][j];
      sources_(i, c) = finite_ica(value);
      if (std::abs(sources_(i, c)) > biggest) {
        biggest = std::abs(sources_(i, c));
        at      = sources_(i, c);
      }
    }
    const double sign = at < 0 ? -1. : 1.;
    for (std::size_t j = 0; j < k; ++j)
      rotation_(c, j) = sign * W[c][j];
    long double m2 = 0, m4 = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sources_(i, c) *= sign;
      const long double s = sources_(i, c);
      m2 += s * s;
      m4 += s * s * s * s;
    }
    source_kurtosis_[c] = finite_ica(m4 * n / (m2 * m2) - 3);
  }
  unmixing_ = linalg::DenseMatrix<double>(k, p);
  mixing_   = linalg::DenseMatrix<double>(p, k);
  for (std::size_t c = 0; c < k; ++c)
    for (std::size_t j = 0; j < p; ++j) {
      long double b = 0, a = 0;
      for (std::size_t d = 0; d < k; ++d) {
        b += static_cast<long double>(rotation_(c, d)) * whitening_(j, d);
        a += static_cast<long double>(eig.eigenvectors(j, d)) * scale *
             std::sqrt(eig.eigenvalues[d]) * rotation_(c, d);
      }
      unmixing_(c, j) = finite_ica(b);
      mixing_(j, c)   = finite_ica(a);
    }
}
std::vector<double> ICA::component(std::size_t c) const {
  if (c >= n_components())
    throw std::out_of_range("ICA::component index out of range");
  return sources_.col(c);
}
linalg::DenseMatrix<double> ICA::transform(const dstruct::DataFrame& data) const {
  for (const auto& col : feature_columns_)
    if (!data.has_column(col) || data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("ICA::transform: missing/non-numeric feature '" + col + "'");
  linalg::DenseMatrix<double> out(data.nrows(), n_components());
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    std::vector<double> x(feature_columns_.size());
    for (std::size_t j = 0; j < x.size(); ++j) {
      if (data.is_null(feature_columns_[j], i))
        throw std::invalid_argument("ICA::transform: null feature");
      x[j] = finite_ica(
          static_cast<long double>(finite_ica(data.double_at(feature_columns_[j], i))) - mean_[j]);
    }
    for (std::size_t c = 0; c < n_components(); ++c) {
      long double v = 0;
      for (std::size_t j = 0; j < x.size(); ++j)
        v += static_cast<long double>(x[j]) * unmixing_(c, j);
      out(i, c) = finite_ica(v);
    }
  }
  return out;
}
linalg::DenseMatrix<double> ICA::inverse_transform(
    const linalg::DenseMatrix<double>& scores) const {
  if (scores.cols() != n_components())
    throw std::invalid_argument("ICA::inverse_transform: wrong component count");
  check_matrix(scores);
  linalg::DenseMatrix<double> out(scores.rows(), mean_.size());
  for (std::size_t i = 0; i < scores.rows(); ++i)
    for (std::size_t j = 0; j < mean_.size(); ++j) {
      long double v = mean_[j];
      for (std::size_t c = 0; c < n_components(); ++c)
        v += static_cast<long double>(scores(i, c)) * mixing_(j, c);
      out(i, j) = finite_ica(v);
    }
  return out;
}
dstruct::DataFrame ICA::convergence_history() const {
  return history_;
}
dstruct::DataFrame ICA::component_diagnostics() const {
  std::vector<double> component, iterations, converged, error, mean, variance;
  for (std::size_t c = 0; c < n_components(); ++c) {
    long double m = 0, v = 0;
    for (std::size_t i = 0; i < observations_; ++i) {
      m += sources_(i, c);
      v += static_cast<long double>(sources_(i, c)) * sources_(i, c);
    }
    component.push_back(c + 1);
    iterations.push_back(iterations_[c]);
    converged.push_back(component_converged_[c] ? 1 : 0);
    error.push_back(alignment_errors_[c]);
    mean.push_back(finite_ica(m / observations_));
    variance.push_back(finite_ica(v / observations_));
  }
  dstruct::DataFrame out;
  out.add_column("component", component);
  out.add_column("iterations", iterations);
  out.add_column("converged", converged);
  out.add_column("alignment_error", error);
  out.add_column("mean", mean);
  out.add_column("population_variance", variance);
  out.add_column("excess_kurtosis", source_kurtosis_);
  return out;
}
dstruct::DataFrame ICA::diagnostics() const {
  const auto  k = n_components(), p = feature_columns_.size();
  double      orthogonal = 0;
  long double residual   = 0;
  for (std::size_t a = 0; a < k; ++a)
    for (std::size_t b = 0; b < k; ++b) {
      long double v = 0;
      for (std::size_t j = 0; j < k; ++j)
        v += static_cast<long double>(rotation_(a, j)) * rotation_(b, j);
      orthogonal = std::max(orthogonal, std::abs(finite_ica(v) - (a == b ? 1. : 0.)));
    }
  for (std::size_t i = 0; i < observations_; ++i)
    for (std::size_t j = 0; j < p; ++j) {
      long double v = centered_(i, j);
      for (std::size_t c = 0; c < k; ++c)
        v -= static_cast<long double>(sources_(i, c)) * mixing_(j, c);
      residual += v * v;
    }
  dstruct::DataFrame out;
  out.add_column("observations", std::vector<double>{double(observations_)});
  out.add_column("features", std::vector<double>{double(p)});
  out.add_column("components", std::vector<double>{double(k)});
  out.add_column("converged", std::vector<double>{converged_ ? 1. : 0.});
  out.add_column("whitening_error", std::vector<double>{covariance_error(whitened_)});
  out.add_column("source_covariance_error", std::vector<double>{covariance_error(sources_)});
  out.add_column("rotation_error", std::vector<double>{orthogonal});
  out.add_column(
      "reconstruction_rmse",
      std::vector<double>{finite_ica(std::sqrt(residual / (observations_ * p)))});
  out.add_column("retained_variance_fraction", std::vector<double>{retained_variance_fraction_});
  out.add_column("rank_threshold", std::vector<double>{rank_threshold_});
  return out;
}
dstruct::DataFrame ica_benchmark(std::size_t n, std::uint64_t seed, const std::string& kind) {
  if (n < 2 ||
      (kind != "independent" && kind != "gaussian" && kind != "waveforms" && kind != "uniform"))
    throw std::invalid_argument("ica_benchmark: n>=2 and a known kind required");
  const std::size_t                k = kind == "uniform" ? 2 : 3;
  random::Xoshiro256StarStar       rng(seed);
  std::vector<std::vector<double>> s(k, std::vector<double>(n)), x(k, std::vector<double>(n));
  std::vector<double>              t(n);
  const double a3[3][3] = {{.8, .3, .2}, {.4, .9, .5}, {.6, .2, .7}}, a2[2][2] = {{1, .8}, {.3, 1}};
  for (std::size_t i = 0; i < n; ++i) {
    t[i] = i;
    if (kind == "waveforms") {
      s[0][i] = std::sin(.060 * i);
      s[1][i] = std::sin(.093 * i) >= 0 ? 1 : -1;
      s[2][i] = std::fmod(.050 * i, 2) - 1;
    } else if (kind == "gaussian") {
      for (std::size_t c = 0; c < k; ++c)
        s[c][i] = rng.normal();
    } else if (kind == "uniform") {
      for (std::size_t c = 0; c < k; ++c)
        s[c][i] = std::sqrt(3.) * (2 * rng.uniform01() - 1);
    } else {
      const double u = rng.uniform01() - .5;
      s[0][i] =
          -std::copysign(1., u) *
          std::log1p(-2 * std::min(std::abs(u), .5 - std::numeric_limits<double>::epsilon())) /
          std::sqrt(2.);
      s[1][i] = std::sqrt(3.) * (2 * rng.uniform01() - 1);
      s[2][i] = rng.uniform01() < .5 ? -1 : 1;
    }
    for (std::size_t j = 0; j < k; ++j)
      for (std::size_t c = 0; c < k; ++c)
        x[j][i] += (k == 2 ? a2[j][c] : a3[j][c]) * s[c][i];
  }
  dstruct::DataFrame out;
  out.add_column("t", t);
  for (std::size_t c = 0; c < k; ++c)
    out.add_column("s" + std::to_string(c + 1), s[c]);
  for (std::size_t j = 0; j < k; ++j)
    out.add_column("x" + std::to_string(j + 1), x[j]);
  return out;
}
dstruct::DataFrame ica_source_matches(
    const linalg::DenseMatrix<double>& reference,
    const linalg::DenseMatrix<double>& recovered) {
  const auto n = reference.rows(), k = reference.cols();
  if (n < 2 || k == 0 || recovered.rows() != n || recovered.cols() != k)
    throw std::invalid_argument("ICA matching: require equal n-by-k inputs, n>=2, k>=1");
  check_matrix(reference);
  check_matrix(recovered);
  std::vector<double>              mr(k), me(k), sr(k), se(k);
  std::vector<std::vector<double>> corr(k, std::vector<double>(k)), cost = corr;
  for (std::size_t c = 0; c < k; ++c) {
    long double r = 0, e = 0;
    for (std::size_t i = 0; i < n; ++i) {
      r += reference(i, c);
      e += recovered(i, c);
    }
    mr[c]          = finite_ica(r / n);
    me[c]          = finite_ica(e / n);
    long double vr = 0, ve = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const long double a = static_cast<long double>(reference(i, c)) - mr[c],
                        b = static_cast<long double>(recovered(i, c)) - me[c];
      vr += a * a;
      ve += b * b;
    }
    sr[c] = finite_ica(std::sqrt(vr / n));
    se[c] = finite_ica(std::sqrt(ve / n));
    if (sr[c] == 0 || se[c] == 0)
      throw std::invalid_argument("ICA matching: constant column");
  }
  for (std::size_t a = 0; a < k; ++a)
    for (std::size_t b = 0; b < k; ++b) {
      long double v = 0;
      for (std::size_t i = 0; i < n; ++i)
        v += ((static_cast<long double>(reference(i, a)) - mr[a]) / sr[a]) *
             ((static_cast<long double>(recovered(i, b)) - me[b]) / se[b]);
      corr[a][b] = std::clamp(finite_ica(v / n), -1., 1.);
      cost[a][b] = -std::abs(corr[a][b]);
    }
  const auto          match = algorithms::hungarian(cost);
  std::vector<double> source, component, correlation, absolute, offset, multiplier;
  for (std::size_t a = 0; a < k; ++a) {
    const auto   b = match.assignment[a];
    const double v = (corr[a][b] < 0 ? -1. : 1.) * sr[a] / se[b];
    source.push_back(a + 1);
    component.push_back(b + 1);
    correlation.push_back(corr[a][b]);
    absolute.push_back(std::abs(corr[a][b]));
    multiplier.push_back(finite_ica(v));
    offset.push_back(
        finite_ica(static_cast<long double>(mr[a]) - static_cast<long double>(v) * me[b]));
  }
  dstruct::DataFrame out;
  out.add_column("source", source);
  out.add_column("component", component);
  out.add_column("correlation", correlation);
  out.add_column("absolute_correlation", absolute);
  out.add_column("offset", offset);
  out.add_column("multiplier", multiplier);
  return out;
}
std::string ICA::summary() const {
  std::ostringstream oss;
  print_summary(oss);
  return oss.str();
}

void ICA::print_summary() const {
  print_summary(std::cout);
}

void ICA::print_summary(std::ostream& os) const {
  os << "Independent Component Analysis (FastICA, " << options_.contrast << " contrast)\n";
  os << "Number of obs: " << observations_ << ", components: " << n_components() << "\n";
  os << "PCA-whitened to " << n_components() << " dimensions\n";
  os << (converged_ ? "All components converged\n" : "At least one component did NOT converge\n");
  os << std::fixed << std::setprecision(4);
  os << "\n"
     << std::left << std::setw(8) << "IC" << std::right << std::setw(12) << "iterations"
     << std::setw(18) << "excess kurtosis\n";
  for (std::size_t c = 0; c < n_components(); ++c)
    os << std::left << std::setw(8) << ("IC" + std::to_string(c + 1)) << std::right << std::setw(12)
       << iterations_[c] << std::setw(18) << source_kurtosis_[c] << "\n";
}

graphics::RPlot ICA::plot_sources(
    const std::vector<std::string>& group_labels,
    const std::size_t               component_x,
    const std::size_t               component_y) const {
  if (component_x >= n_components() || component_y >= n_components())
    throw std::out_of_range("ICA::plot_sources component index out of range");
  if (group_labels.size() != observations_)
    throw std::invalid_argument(
        "ICA::plot_sources: group_labels must have one entry per fitted observation");

  auto       plot   = graphics::RPlot::create();
  const auto groups = unique_sorted(group_labels);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    std::vector<double> xs, ys;
    for (std::size_t i = 0; i < observations_; ++i) {
      if (group_labels[i] != groups[g])
        continue;
      xs.push_back(sources_(i, component_x));
      ys.push_back(sources_(i, component_y));
    }
    plot.points(
        xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
  }
  plot.title("Independent Component Analysis")
      .x_label("IC" + std::to_string(component_x + 1))
      .y_label("IC" + std::to_string(component_y + 1));
  return plot;
}

} // namespace datamunge::stats
