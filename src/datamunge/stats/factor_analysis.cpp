/// @file factor_analysis.cpp
/// @brief Principal-axis extraction, orthogonal rotation and inspectable covariance checks.
#include <algorithm>
#include <cmath>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/factor_analysis.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Private arithmetic, rotation and presentation helpers.
namespace {

/// @brief Recycled categorical colours for fitted score groups.
const graphics::RGB kSeriesColors[] = {
    {37, 99, 235},
    {220, 38, 38},
    {22, 163, 74},
    {217, 119, 6},
    {124, 58, 237},
    {8, 145, 178},
};

/// @brief Copies distinct group labels in lexical order.
/// @param labels Borrowed group names.
/// @return Owned sorted unique names.
std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
  std::vector<std::string> result(labels.begin(), labels.end());
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

/// @brief Converts extended arithmetic to a finite double, allowing underflow.
/// @param x Intermediate value.
/// @return Representable double.
/// @throws std::invalid_argument Nonfinite or unrepresentable value.
double finite_fa(long double x) {
  const double result = static_cast<double>(x);
  if (!std::isfinite(result))
    throw std::invalid_argument("FactorAnalysis: nonfinite/unrepresentable arithmetic");
  return result;
}
/// @brief Normalizes a symmetric matrix before Jacobi to protect its norm calculation.
/// @param A Finite symmetric matrix.
/// @return Descending eigenpairs in original units.
/// @throws std::invalid_argument Unrepresentable eigenvalue.
linalg::SymmetricEigenResult<double> scaled_eigen(linalg::DenseMatrix<double> A) {
  double scale = 0;
  for (std::size_t i = 0; i < A.rows(); ++i)
    for (std::size_t j = 0; j < A.cols(); ++j)
      scale = std::max(scale, std::abs(A(i, j)));
  if (scale > 0)
    for (std::size_t i = 0; i < A.rows(); ++i)
      for (std::size_t j = 0; j < A.cols(); ++j)
        A(i, j) /= scale;
  auto eig = linalg::jacobi_eigen(A);
  for (auto& e : eig.eigenvalues)
    e = finite_fa(static_cast<long double>(e) * scale);
  return eig;
}
/// @brief Eigenvalue-floored inverse with a relative floor in the fitted feature units.
/// @param A Finite positive-semidefinite sample covariance/correlation.
/// @return V diag(1/max(lambda,1e-8*max(diag(A)))) V^T; not a pseudoinverse.
/// @throws std::invalid_argument Unrepresentable inverse.
linalg::DenseMatrix<double> symmetric_inverse(const linalg::DenseMatrix<double>& A) {
  const auto        eig   = scaled_eigen(A);
  const std::size_t n     = A.rows();
  double            scale = 0;
  for (std::size_t i = 0; i < n; ++i)
    scale = std::max(scale, A(i, i));
  linalg::DenseMatrix<double> inv(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      long double sum = 0;
      for (std::size_t c = 0; c < n; ++c) {
        const long double lambda =
            std::max(static_cast<long double>(eig.eigenvalues[c]), 1e-8L * scale);
        sum += static_cast<long double>(eig.eigenvectors(i, c)) * eig.eigenvectors(j, c) / lambda;
      }
      inv(i, j) = finite_fa(sum);
    }
  return inv;
}
/// @brief Counts material loading variances above their observed variances.
/// @param C Observed covariance.
/// @param L Loading matrix in the same units.
/// @return Count whose excess exceeds 64*epsilon*max(observed,common).
std::size_t heywood_count(
    const linalg::DenseMatrix<double>& C,
    const linalg::DenseMatrix<double>& L) {
  std::size_t count = 0;
  for (std::size_t j = 0; j < L.rows(); ++j) {
    long double h = 0;
    for (std::size_t c = 0; c < L.cols(); ++c)
      h += static_cast<long double>(L(j, c)) * L(j, c);
    if (h - C(j, j) > 64 * std::numeric_limits<double>::epsilon() *
                          std::max(h, static_cast<long double>(C(j, j))))
      ++count;
  }
  return count;
}

/// @brief Kaiser-normalized pairwise varimax, preserving L*L^T up to roundoff.
/// @param L Owned loading matrix changed in place; a single column is unchanged.
/// @param max_iter Maximum pair sweeps.
/// @param tol Absolute change tolerance on the normalized varimax objective.
/// @throws std::invalid_argument Unrepresentable row norm or rotated loading.
/// @see FactorAnalysisOptions
void varimax(linalg::DenseMatrix<double>& L, std::size_t max_iter, double tol) {
  const std::size_t p = L.rows();
  const std::size_t k = L.cols();
  if (k < 2)
    return;

  // Kaiser normalization: scale each row to unit length so high- and low-communality variables
  // influence the rotation equally, then restore afterward.
  std::vector<double> norm(p, 1.0);
  for (std::size_t j = 0; j < p; ++j) {
    long double ss = 0.0;
    for (std::size_t c = 0; c < k; ++c)
      ss += static_cast<long double>(L(j, c)) * L(j, c);
    norm[j] = finite_fa(std::sqrt(ss));
    if (norm[j] > 0.0)
      for (std::size_t c = 0; c < k; ++c)
        L(j, c) /= norm[j];
  }

  double prev_crit = 0.0;
  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    for (std::size_t a = 0; a < k; ++a) {
      for (std::size_t b = a + 1; b < k; ++b) {
        double sum_u = 0.0, sum_v = 0.0, sum_u2mv2 = 0.0, sum_2uv = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
          const double x = L(j, a), y = L(j, b);
          const double u = x * x - y * y;
          const double v = 2.0 * x * y;
          sum_u += u;
          sum_v += v;
          sum_u2mv2 += u * u - v * v;
          sum_2uv += 2.0 * u * v;
        }
        const double num   = sum_2uv - 2.0 * sum_u * sum_v / static_cast<double>(p);
        const double den   = sum_u2mv2 - (sum_u * sum_u - sum_v * sum_v) / static_cast<double>(p);
        const double theta = 0.25 * std::atan2(num, den);
        if (std::abs(theta) < 1e-12)
          continue;
        const double c = std::cos(theta), s = std::sin(theta);
        for (std::size_t j = 0; j < p; ++j) {
          const double x = L(j, a), y = L(j, b);
          L(j, a) = c * x + s * y;
          L(j, b) = -s * x + c * y;
        }
      }
    }

    double crit = 0.0;
    for (std::size_t c = 0; c < k; ++c) {
      double s1 = 0.0, s2 = 0.0;
      for (std::size_t j = 0; j < p; ++j) {
        const double q = L(j, c) * L(j, c);
        s1 += q;
        s2 += q * q;
      }
      crit += s2 - s1 * s1 / static_cast<double>(p);
    }
    if (iter > 0 && std::abs(crit - prev_crit) < tol)
      break;
    prev_crit = crit;
  }

  for (std::size_t j = 0; j < p; ++j)
    for (std::size_t c = 0; c < k; ++c)
      L(j, c) = finite_fa(static_cast<long double>(L(j, c)) * norm[j]);
}

} // namespace

FactorAnalysis::FactorAnalysis(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& feature_columns,
    FactorAnalysisOptions           options)
    : feature_columns_(feature_columns), options_(options) {
  fit(data);
}

void FactorAnalysis::fit(const dstruct::DataFrame& data) {
  if (feature_columns_.empty())
    throw std::invalid_argument("FactorAnalysis: feature_columns must not be empty");
  if (options_.n_factors == 0)
    throw std::invalid_argument("FactorAnalysis: n_factors must be at least 1");
  if (options_.rotation != "none" && options_.rotation != "varimax")
    throw std::invalid_argument("FactorAnalysis: rotation must be 'none' or 'varimax'");
  if (options_.max_iter == 0 || !std::isfinite(options_.tol) || options_.tol <= 0)
    throw std::invalid_argument(
        "FactorAnalysis: positive iteration limit and finite positive tolerance required");
  std::set<std::string> seen;
  for (const auto& col : feature_columns_) {
    if (!seen.insert(col).second)
      throw std::invalid_argument("FactorAnalysis: duplicate feature");
    if (!data.has_column(col))
      throw std::invalid_argument("FactorAnalysis: column '" + col + "' not found");
    if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("FactorAnalysis: column '" + col + "' must be numeric");
  }

  kept_row_indices_.clear();
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    bool ok = true;
    for (const auto& col : feature_columns_) {
      if (data.is_null(col, i)) {
        ok = false;
        break;
      }
    }
    if (ok)
      kept_row_indices_.push_back(i);
  }

  const std::size_t n = kept_row_indices_.size();
  const std::size_t p = feature_columns_.size();
  const std::size_t k = options_.n_factors;
  if (n < 2)
    throw std::invalid_argument("FactorAnalysis: fewer than 2 complete observations");
  if (k > p)
    throw std::invalid_argument("FactorAnalysis: n_factors must not exceed the number of features");

  // Complete rows are validated before centring; null rows are not examined further.
  linalg::DenseMatrix<double> Xc(n, p, 0.0);
  mean_.assign(p, 0);
  scale_.assign(p, 1);
  for (std::size_t j = 0; j < p; ++j) {
    long double sum = 0;
    for (auto i : kept_row_indices_)
      sum += finite_fa(data.double_at(feature_columns_[j], i));
    mean_[j]       = finite_fa(sum / n);
    long double ss = 0;
    for (std::size_t i = 0; i < n; ++i) {
      Xc(i, j) = finite_fa(
          static_cast<long double>(data.double_at(feature_columns_[j], kept_row_indices_[i])) -
          mean_[j]);
      ss += static_cast<long double>(Xc(i, j)) * Xc(i, j);
    }
    if (!(ss > 0))
      throw std::invalid_argument(
          "FactorAnalysis: each selected feature needs positive sample variance");
    if (options_.standardize) {
      scale_[j] = finite_fa(std::sqrt(ss / (n - 1)));
      if (!(scale_[j] > 0))
        throw std::invalid_argument("FactorAnalysis: unrepresentable scale");
      for (std::size_t i = 0; i < n; ++i)
        Xc(i, j) = finite_fa(static_cast<long double>(Xc(i, j)) / scale_[j]);
    }
  }
  covariance_ = linalg::DenseMatrix<double>(p, p, 0.0);
  auto& C     = covariance_;
  for (std::size_t a = 0; a < p; ++a)
    for (std::size_t b = a; b < p; ++b) {
      long double sum = 0;
      for (std::size_t i = 0; i < n; ++i)
        sum += static_cast<long double>(Xc(i, a)) * Xc(i, b);
      C(a, b) = C(b, a) = finite_fa(sum / (n - 1));
    }
  total_variance_ = 0;
  for (std::size_t j = 0; j < p; ++j) {
    if (!(C(j, j) > 0))
      throw std::invalid_argument("FactorAnalysis: unrepresentable positive variance");
    total_variance_ = finite_fa(static_cast<long double>(total_variance_) + C(j, j));
  }

  // Initial communality estimate: squared multiple correlation, h_j^2 = 1 - 1/(C^{-1})_{jj}
  // (relative to that variable's own variance), the standard PAF starting point.
  const auto          Cinv = symmetric_inverse(C);
  std::vector<double> comm(p, 0.0);
  for (std::size_t j = 0; j < p; ++j) {
    const double smc = finite_fa(1.0L - 1.0L / (static_cast<long double>(Cinv(j, j)) * C(j, j)));
    comm[j]          = C(j, j) * std::clamp(smc, 0.0, 1.0);
  }

  // Principal axis factoring: repeatedly eigendecompose C with its diagonal replaced by the
  // current communalities, rebuild the loadings from the top-k eigenpairs, and refresh the
  // communalities as the loadings' row sums of squares, until they stop moving.
  linalg::DenseMatrix<double>      L(p, k, 0.0);
  std::vector<double>              history_iteration, history_change, history_rms, history_heywood;
  std::vector<std::vector<double>> history_comm(p);
  converged_  = false;
  iterations_ = 0;
  for (std::size_t iter = 0; iter < options_.max_iter; ++iter) {
    linalg::DenseMatrix<double> reduced = C;
    for (std::size_t j = 0; j < p; ++j)
      reduced(j, j) = comm[j];

    const auto eig = scaled_eigen(reduced);
    for (std::size_t c = 0; c < k; ++c) {
      const double lambda = std::max(eig.eigenvalues[c], 0.0);
      const double s      = std::sqrt(lambda);
      for (std::size_t j = 0; j < p; ++j)
        L(j, c) = eig.eigenvectors(j, c) * s;
    }

    double max_change = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
      double h2 = 0.0;
      for (std::size_t c = 0; c < k; ++c)
        h2 += L(j, c) * L(j, c);
      h2         = std::clamp(h2, 0.0, C(j, j)); // guard Heywood cases
      max_change = std::max(max_change, std::abs(h2 - comm[j]));
      comm[j]    = h2;
    }
    history_iteration.push_back(iter + 1);
    history_change.push_back(max_change);
    history_heywood.push_back(heywood_count(C, L));
    history_rms.push_back(covariance_residual_rms(C, L * L.transpose()));
    for (std::size_t j = 0; j < p; ++j)
      history_comm[j].push_back(comm[j]);
    iterations_ = iter + 1;
    if (max_change < options_.tol) {
      converged_ = true;
      break;
    }
  }

  iteration_history_.add_column("iteration", history_iteration);
  iteration_history_.add_column("max_change", history_change);
  iteration_history_.add_column("offdiagonal_rms", history_rms);
  iteration_history_.add_column("heywood_variables", history_heywood);
  for (std::size_t j = 0; j < p; ++j)
    iteration_history_.add_column("communality_" + std::to_string(j + 1), history_comm[j]);

  // Optional varimax rotation toward simple structure (leaves communalities and fit unchanged).
  if (options_.rotation == "varimax")
    varimax(L, options_.max_iter, options_.tol);

  // Order the (rotated) factors by descending variance explained, for a stable, PCA-like report.
  std::vector<double> ss(k, 0.0);
  for (std::size_t c = 0; c < k; ++c)
    for (std::size_t j = 0; j < p; ++j)
      ss[c] += L(j, c) * L(j, c);
  std::vector<std::size_t> order(k);
  std::iota(order.begin(), order.end(), 0);
  std::sort(
      order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ss[a] > ss[b]; });

  loadings_ = linalg::DenseMatrix<double>(p, k, 0.0);
  variance_explained_.assign(k, 0.0);
  for (std::size_t c = 0; c < k; ++c) {
    // Fix each factor's sign so its largest-magnitude loading is positive (a convention).
    std::size_t argmax = 0;
    double      best   = 0.0;
    for (std::size_t j = 0; j < p; ++j)
      if (std::abs(L(j, order[c])) > best) {
        best   = std::abs(L(j, order[c]));
        argmax = j;
      }
    const double sign = (L(argmax, order[c]) < 0.0) ? -1.0 : 1.0;
    for (std::size_t j = 0; j < p; ++j)
      loadings_(j, c) = sign * L(j, order[c]);
    variance_explained_[c] = ss[order[c]];
  }

  communalities_.assign(p, 0.0);
  uniquenesses_.assign(p, 0.0);
  for (std::size_t j = 0; j < p; ++j) {
    double h2 = 0.0;
    for (std::size_t c = 0; c < k; ++c)
      h2 = finite_fa(
          static_cast<long double>(h2) +
          static_cast<long double>(loadings_(j, c)) * loadings_(j, c));
    communalities_[j] = h2;
    uniquenesses_[j]  = std::max(C(j, j) - h2, 0.0);
  }

  // Regression (Thomson) factor scores: F = Z C^{-1} Lambda.
  linalg::DenseMatrix<double> B = Cinv * loadings_; // p x k
  scores_                       = Xc * B;           // n x k
  observations_                 = n;

  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t c = 0; c < k; ++c)
      (void)finite_fa(scores_(i, c));
}

std::vector<double> FactorAnalysis::factor_loadings(const std::size_t factor_index) const {
  if (factor_index >= loadings_.cols())
    throw std::out_of_range("FactorAnalysis::factor_loadings index out of range");
  return loadings_.col(factor_index);
}

std::vector<double> FactorAnalysis::factor_scores(const std::size_t factor_index) const {
  if (factor_index >= scores_.cols())
    throw std::out_of_range("FactorAnalysis::factor_scores index out of range");
  return scores_.col(factor_index);
}

std::vector<double> FactorAnalysis::proportion_variance() const {
  const double        total = total_variance_;
  std::vector<double> out(variance_explained_.size(), 0.0);
  if (total <= 0.0)
    return out;
  for (std::size_t c = 0; c < variance_explained_.size(); ++c)
    out[c] = variance_explained_[c] / total;
  return out;
}

std::string FactorAnalysis::summary() const {
  std::ostringstream oss;
  print_summary(oss);
  return oss.str();
}

void FactorAnalysis::print_summary() const {
  print_summary(std::cout);
}

void FactorAnalysis::print_summary(std::ostream& os) const {
  os << "Factor Analysis (principal axis factoring";
  if (options_.rotation != "none")
    os << ", " << options_.rotation << " rotation";
  os << ")\n";
  os << "Number of obs: " << observations_ << ", factors: " << num_factors() << "\n";
  os << (options_.standardize ? "Fit on correlation matrix\n" : "Fit on covariance matrix\n");
  os << "Communalities " << (converged_ ? "converged" : "did NOT converge") << " in " << iterations_
     << " iterations\n";

  os << "Material Heywood variables: " << heywood_count(covariance_, loadings_)
     << " (inspect raw uniqueness)\n";
  os << std::fixed << std::setprecision(4);
  const std::size_t k = num_factors();
  os << "\nLoadings:\n" << std::left << std::setw(16) << "";
  for (std::size_t c = 0; c < k; ++c)
    os << std::right << std::setw(10) << ("Factor" + std::to_string(c + 1));
  os << std::right << std::setw(12) << "Communality" << std::setw(11) << "Uniqueness\n";
  for (std::size_t j = 0; j < feature_columns_.size(); ++j) {
    os << std::left << std::setw(16) << feature_columns_[j];
    for (std::size_t c = 0; c < k; ++c)
      os << std::right << std::setw(10) << loadings_(j, c);
    os << std::right << std::setw(12) << communalities_[j] << std::setw(11) << uniquenesses_[j]
       << "\n";
  }

  const auto prop = proportion_variance();
  os << "\n" << std::left << std::setw(16) << "SS loadings";
  for (std::size_t c = 0; c < k; ++c)
    os << std::right << std::setw(10) << variance_explained_[c];
  os << "\n" << std::left << std::setw(16) << "Prop. variance";
  for (std::size_t c = 0; c < k; ++c)
    os << std::right << std::setw(10) << prop[c];
  os << "\n";
}

graphics::RPlot FactorAnalysis::plot_scores(
    const std::vector<std::string>& group_labels,
    const std::size_t               factor_x,
    const std::size_t               factor_y) const {
  if (factor_x >= num_factors() || factor_y >= num_factors())
    throw std::out_of_range("FactorAnalysis::plot_scores factor index out of range");
  if (group_labels.size() != observations_)
    throw std::invalid_argument(
        "FactorAnalysis::plot_scores: group_labels must have one entry per fitted observation");

  auto       plot   = graphics::RPlot::create();
  const auto groups = unique_sorted(group_labels);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    std::vector<double> xs, ys;
    for (std::size_t i = 0; i < observations_; ++i) {
      if (group_labels[i] != groups[g])
        continue;
      xs.push_back(scores_(i, factor_x));
      ys.push_back(scores_(i, factor_y));
    }
    plot.points(
        xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
  }
  plot.title("Factor Analysis Scores")
      .x_label("Factor " + std::to_string(factor_x + 1))
      .y_label("Factor " + std::to_string(factor_y + 1));
  return plot;
}

graphics::RPlot FactorAnalysis::plot_loadings(
    const std::size_t factor_x,
    const std::size_t factor_y) const {
  if (factor_x >= num_factors() || factor_y >= num_factors())
    throw std::out_of_range("FactorAnalysis::plot_loadings factor index out of range");
  if (factor_x == factor_y)
    throw std::invalid_argument("FactorAnalysis::plot_loadings needs distinct factors");

  auto plot = graphics::RPlot::create();
  // Only standardized loadings have a unit-variance circle reference.
  if (options_.standardize) {
    std::vector<double> cx, cy;
    for (std::size_t t = 0; t <= 100; ++t) {
      const double a = 2.0 * std::numbers::pi_v<double> * static_cast<double>(t) / 100.0;
      cx.push_back(std::cos(a));
      cy.push_back(std::sin(a));
    }
    plot.lines(cx, cy, "", {203, 213, 225});
  }
  plot.abline_h(0.0, {203, 213, 225});
  plot.abline_v(0.0, {203, 213, 225});

  std::vector<double> lx, ly;
  for (std::size_t j = 0; j < feature_columns_.size(); ++j) {
    lx.push_back(loadings_(j, factor_x));
    ly.push_back(loadings_(j, factor_y));
  }
  plot.points(lx, ly, "", kSeriesColors[0], 6.0);
  // Keep nearby feature labels vertically separated, with leaders if moved.
  std::vector<std::size_t> label_order(feature_columns_.size());
  std::iota(label_order.begin(), label_order.end(), 0);
  // Order labels by their displayed vertical coordinate.
  std::sort(label_order.begin(), label_order.end(), [&](std::size_t a, std::size_t b) {
    return ly[a] < ly[b];
  });
  double last   = -std::numeric_limits<double>::infinity();
  double yscale = 0;
  for (auto y : ly)
    yscale = std::max(yscale, std::abs(y));
  const double gap = .11 * std::max(yscale, options_.standardize ? 1. : 1e-12);
  for (auto j : label_order) {
    const double y = std::max(ly[j] + gap * .4, last + gap);
    last           = y;
    const double x = lx[j] + gap * .5;
    plot.text(x, y, feature_columns_[j], {17, 24, 39}, 12.0);
    if (std::abs(y - ly[j]) > gap * .8)
      plot.line({lx[j], x}, {ly[j], y}, "", {148, 163, 184});
  }

  plot.title("Factor Loadings")
      .x_label("Factor " + std::to_string(factor_x + 1))
      .y_label("Factor " + std::to_string(factor_y + 1));
  return plot;
}

double covariance_residual_rms(
    const linalg::DenseMatrix<double>& observed,
    const linalg::DenseMatrix<double>& reproduced) {
  const auto p = observed.rows();
  if (observed.cols() != p || reproduced.rows() != p || reproduced.cols() != p)
    throw std::invalid_argument("covariance_residual_rms: equal square matrices required");
  long double sum = 0;
  for (std::size_t i = 0; i < p; ++i)
    for (std::size_t j = 0; j < p; ++j) {
      (void)finite_fa(observed(i, j));
      (void)finite_fa(reproduced(i, j));
      const long double e = static_cast<long double>(observed(i, j)) - reproduced(i, j);
      if (i != j)
        sum += e * e;
    }
  return p < 2 ? 0 : finite_fa(std::sqrt(sum / (static_cast<long double>(p) * (p - 1))));
}
dstruct::DataFrame factor_model_sample(
    const linalg::DenseMatrix<double>& loadings,
    const std::vector<double>&         uniqueness,
    std::size_t                        observations,
    std::uint64_t                      seed) {
  const auto p = loadings.rows(), k = loadings.cols();
  if (p == 0 || k == 0 || uniqueness.size() != p || observations < 2)
    throw std::invalid_argument(
        "factor_model_sample: positive dimensions, matching uniqueness and n>=2 required");
  std::vector<double> noise(p), f(k);
  for (std::size_t j = 0; j < p; ++j) {
    if (finite_fa(uniqueness[j]) < 0)
      throw std::invalid_argument("factor_model_sample: nonnegative uniqueness required");
    noise[j] = std::sqrt(uniqueness[j]);
    for (std::size_t c = 0; c < k; ++c)
      (void)finite_fa(loadings(j, c));
  }
  std::mt19937_64                  rng(seed);
  std::normal_distribution<double> normal(0, 1);
  std::vector<std::vector<double>> columns(p, std::vector<double>(observations));
  for (std::size_t i = 0; i < observations; ++i) {
    for (auto& value : f)
      value = normal(rng);
    for (std::size_t j = 0; j < p; ++j) {
      long double x = static_cast<long double>(noise[j]) * normal(rng);
      for (std::size_t c = 0; c < k; ++c)
        x += static_cast<long double>(loadings(j, c)) * f[c];
      columns[j][i] = finite_fa(x);
    }
  }
  dstruct::DataFrame out;
  for (std::size_t j = 0; j < p; ++j)
    out.add_column("v" + std::to_string(j + 1), columns[j]);
  return out;
}
linalg::DenseMatrix<double> FactorAnalysis::common_covariance() const {
  const auto                  p = loadings_.rows();
  linalg::DenseMatrix<double> out(p, p, 0.0);
  for (std::size_t i = 0; i < p; ++i)
    for (std::size_t j = i; j < p; ++j) {
      long double sum = 0;
      for (std::size_t c = 0; c < num_factors(); ++c)
        sum += static_cast<long double>(loadings_(i, c)) * loadings_(j, c);
      out(i, j) = out(j, i) = finite_fa(sum);
    }
  return out;
}
linalg::DenseMatrix<double> FactorAnalysis::reproduced_covariance() const {
  auto out = common_covariance();
  for (std::size_t j = 0; j < out.rows(); ++j)
    out(j, j) = finite_fa(static_cast<long double>(out(j, j)) + uniquenesses_[j]);
  return out;
}
dstruct::DataFrame FactorAnalysis::loading_table() const {
  dstruct::DataFrame out;
  out.add_column("feature", feature_columns_);
  for (std::size_t c = 0; c < num_factors(); ++c)
    out.add_column("Factor" + std::to_string(c + 1), factor_loadings(c));
  return out;
}
dstruct::DataFrame FactorAnalysis::variance_table() const {
  std::vector<double> observed, raw, flag, fraction;
  for (std::size_t j = 0; j < feature_columns_.size(); ++j) {
    const auto v = covariance_(j, j), h = communalities_[j];
    observed.push_back(v);
    raw.push_back(v - h);
    flag.push_back(h - v > 64 * std::numeric_limits<double>::epsilon() * std::max(h, v) ? 1 : 0);
    fraction.push_back(finite_fa(static_cast<long double>(h) / v));
  }
  dstruct::DataFrame out;
  out.add_column("feature", feature_columns_);
  out.add_column("observed_variance", observed);
  out.add_column("communality", communalities_);
  out.add_column("raw_uniqueness", raw);
  out.add_column("uniqueness", uniquenesses_);
  out.add_column("heywood", flag);
  out.add_column("common_fraction", fraction);
  return out;
}
dstruct::DataFrame FactorAnalysis::covariance_pairs() const {
  const auto               common = common_covariance(), fitted = reproduced_covariance();
  std::vector<std::string> left, right;
  std::vector<double>      observed, c, f, residual;
  for (std::size_t i = 0; i < common.rows(); ++i)
    for (std::size_t j = i; j < common.cols(); ++j) {
      left.push_back(feature_columns_[i]);
      right.push_back(feature_columns_[j]);
      observed.push_back(covariance_(i, j));
      c.push_back(common(i, j));
      f.push_back(fitted(i, j));
      residual.push_back(finite_fa(static_cast<long double>(covariance_(i, j)) - fitted(i, j)));
    }
  dstruct::DataFrame out;
  out.add_column("feature_i", left);
  out.add_column("feature_j", right);
  out.add_column("observed", observed);
  out.add_column("common", c);
  out.add_column("reproduced", f);
  out.add_column("residual", residual);
  return out;
}
dstruct::DataFrame FactorAnalysis::diagnostics() const {
  const auto        p = feature_columns_.size(), k = num_factors();
  const long double df     = (static_cast<long double>(p - k) * (p - k) - p - k) / 2;
  long double       common = 0;
  double            excess = 0;
  for (std::size_t j = 0; j < p; ++j) {
    common += communalities_[j];
    excess = std::max(excess, communalities_[j] - covariance_(j, j));
  }
  // Compute exactly the row-normalized objective used by Kaiser varimax.
  long double criterion = 0;
  for (std::size_t c = 0; c < k; ++c) {
    long double sum = 0, square = 0;
    for (std::size_t j = 0; j < p; ++j) {
      const long double q = communalities_[j] > 0 ? static_cast<long double>(loadings_(j, c)) *
                                                        loadings_(j, c) / communalities_[j]
                                                  : 0;
      sum += q;
      square += q * q;
    }
    criterion += square - sum * sum / p;
  }
  dstruct::DataFrame out;
  out.add_column("observations", std::vector<double>{double(observations_)});
  out.add_column("features", std::vector<double>{double(p)});
  out.add_column("factors", std::vector<double>{double(k)});
  out.add_column("iterations", std::vector<double>{double(iterations_)});
  out.add_column("converged", std::vector<double>{converged_ ? 1.0 : 0.0});
  out.add_column("degrees_freedom", std::vector<double>{finite_fa(df)});
  out.add_column(
      "heywood_variables", std::vector<double>{double(heywood_count(covariance_, loadings_))});
  out.add_column(
      "offdiagonal_rms",
      std::vector<double>{covariance_residual_rms(covariance_, common_covariance())});
  out.add_column("max_diagonal_excess", std::vector<double>{excess});
  out.add_column("total_variance", std::vector<double>{total_variance_});
  out.add_column("common_variance", std::vector<double>{finite_fa(common)});
  out.add_column("kaiser_varimax", std::vector<double>{finite_fa(criterion)});
  return out;
}

} // namespace datamunge::stats
