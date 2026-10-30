/// @file glmm.cpp
/// @brief PQL relinearization, low-rank working solves and mixed-model diagnostics.
#include <algorithm>
#include <cmath>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/optim/differential_evolution.hpp>
#include <datamunge/optim/nelder_mead.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/mixed_model_formula.hpp>
#include <datamunge/stats/glmm.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::stats {

/// @brief Internal link arithmetic and block Gaussian working-model machinery.
namespace {

using linalg::cholesky;
using linalg::CholeskyDecomposition;
using linalg::DenseMatrix;

/// @brief Twice pi, for the Gaussian working normalization.
constexpr double kTwoPi = 6.283185307179586476925286766559;
/// @brief Lower working-mean guard, also the binomial distance from one.
constexpr double kMuEps = 1e-8;

/// @brief Imports packed-factor construction from the shared mixed-model grammar.
/// @param theta Packed lower triangle.
/// @param q Random-effect dimension.
/// @return Lower-triangular factor.
using detail::build_lambda;
/// @brief Imports the parser for a single parenthesized random-effects term.
/// @param text Complete mixed-model formula.
/// @return Located random-term span, or no span.
using detail::find_random_effect_span;
/// @brief Imports whitespace trimming for formula substrings.
/// @param s Formula substring.
/// @return Trimmed owned string.
using detail::mixed_model_trim;
/// @brief Imports the intercept/plain-slope random-design grammar.
/// @param rhs Content before the grouping pipe.
/// @return Parsed intercept and numeric slope names.
using detail::parse_random_effect_rhs;
/// @brief Parsed random-term representation from the shared grammar.
using detail::RandomEffectSpec;

/// @brief Stable conditional inverse link.
/// @param family Response family.
/// @param eta Linear predictor.
/// @return Response mean, possibly overflowing for the exponential link.
double inverse_link(const GLMMFamily family, const double eta) {
  switch (family) {
    case GLMMFamily::Binomial: {
      const double e = std::exp(-std::fabs(eta));
      return eta >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
    }
    case GLMMFamily::Poisson:
      return std::exp(eta);
  }
  return 0.0;
}

/// @brief Derivative d eta/d mu in the valid working domain.
/// @param family Response family.
/// @param mu Guarded conditional mean.
/// @return Link derivative.
double link_derivative(const GLMMFamily family, const double mu) {
  switch (family) {
    case GLMMFamily::Binomial:
      return 1.0 / (mu * (1.0 - mu));
    case GLMMFamily::Poisson:
      return 1.0 / mu;
  }
  return 0.0;
}

/// @brief Conditional variance function with dispersion fixed at one.
/// @param family Response family.
/// @param mu Guarded conditional mean.
/// @return Conditional variance.
double variance_function(const GLMMFamily family, const double mu) {
  switch (family) {
    case GLMMFamily::Binomial:
      return mu * (1.0 - mu);
    case GLMMFamily::Poisson:
      return mu;
  }
  return 0.0;
}

/// @brief Guards a finite working mean against zero precision.
/// @param family Response family.
/// @param mu Unclamped mean.
/// @return Guarded finite mean; nonfinite means throw.
double clamp_mu(const GLMMFamily family, const double mu) {
  if (!std::isfinite(mu))
    throw std::runtime_error("GLMM: working response mean is nonfinite");
  if (family == GLMMFamily::Binomial)
    return std::min(1.0 - kMuEps, std::max(kMuEps, mu));
  return std::max(kMuEps, mu);
}

/// @brief Rejects nonfinite responses and values outside the conditional sample space.
/// @param family Response family.
/// @param y Retained numeric responses.
void validate_response(const GLMMFamily family, const std::vector<double>& y) {
  for (double v : y) {
    if (!std::isfinite(v))
      throw std::invalid_argument("GLMM: response must be finite");
    if (family == GLMMFamily::Binomial && v != 0.0 && v != 1.0)
      throw std::invalid_argument("GLMM: binomial response must be exactly zero or one");
    if (family == GLMMFamily::Poisson && (v < 0.0 || v != std::floor(v)))
      throw std::invalid_argument("GLMM: Poisson response must be a nonnegative integer");
  }
}

/// @brief Exact printable group key, avoiding six-digit numeric-key collisions.
/// @param data Frame with the grouping column.
/// @param column Grouping column name.
/// @param row Non-null grouping row.
/// @return Owned string key; nonfinite numeric groups are rejected.
std::string group_key(const dstruct::DataFrame& data, const std::string& column, std::size_t row) {
  if (data.column_type(column) == dstruct::DataFrame::ColumnType::String)
    return data.string_at(column, row);
  const double value = data.double_at(column, row);
  if (!std::isfinite(value))
    throw std::invalid_argument("GLMM: numeric groups must be finite");
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10)
      << (value == 0.0 ? 0.0 : value);
  return out.str();
}

/// @brief A Woodbury representation of M^-1 for one working group.
struct WeightedGroupBlock {
  CholeskyDecomposition<double> chol_s;  ///< Cholesky of I+(Z*Lambda)'*W*(Z*Lambda).
  DenseMatrix<double>           zl;      ///< Group matrix Z*Lambda, n_g by q.
  std::vector<double>           weights; ///< Positive working precision diagonal.

  /// @brief Applies the group covariance inverse to a vector without forming M.
  /// @param v Length n_g right-hand side.
  /// @return Owned inverse-weighted vector.
  std::vector<double> solve(const std::vector<double>& v) const {
    std::vector<double> out(v.size()), rhs(zl.cols(), 0.0);
    for (std::size_t a = 0; a < v.size(); ++a) {
      out[a] = weights[a] * v[a];
      for (std::size_t k = 0; k < zl.cols(); ++k)
        rhs[k] += zl(a, k) * out[a];
    }
    const auto t = chol_s.solve(rhs);
    for (std::size_t a = 0; a < v.size(); ++a)
      for (std::size_t k = 0; k < zl.cols(); ++k)
        out[a] -= weights[a] * zl(a, k) * t[k];
    return out;
  }
  /// @brief Applies the inverse to multiple right-hand sides.
  /// @param X n_g by p right-hand sides.
  /// @return Owned n_g by p solution matrix.
  DenseMatrix<double> solve(const DenseMatrix<double>& X) const {
    DenseMatrix<double> out(X.rows(), X.cols(), 0.0);
    for (std::size_t j = 0; j < X.cols(); ++j) {
      std::vector<double> v(X.rows());
      for (std::size_t a = 0; a < X.rows(); ++a)
        v[a] = X(a, j);
      const auto t = solve(v);
      for (std::size_t a = 0; a < X.rows(); ++a)
        out(a, j) = t[a];
    }
    return out;
  }
};

/// @brief Profiled fixed effects, information and objective for frozen pseudo-data.
struct WeightedProfileFit {
  std::vector<WeightedGroupBlock> blocks;   ///< Per-group inverse-covariance operators.
  std::vector<double>             beta;     ///< Profiled fixed coefficients.
  DenseMatrix<double>             xtwx;     ///< Summed fixed working information, p by p.
  double                          rss{0.0}; ///< Profiled residual quadratic.
  double criterion{0.0}; ///< Gaussian working minus-twice log likelihood, dispersion one.
};

// Weighted analogue of LMM's per-group profiled fit: M_g = Z_g*Lambda*Lambda'*Z_g' +
// diag(1/w) instead of + I, and there is no free dispersion to profile out (fixed at 1 for
// Binomial/Poisson), so the criterion is a direct quadratic + log-determinant sum rather
// than LMM's log(RSS/n) form.
/// @brief Profiles fixed effects in a block-Gaussian working model using Woodbury solves.
/// @param theta Packed lower-triangular covariance factor.
/// @param X Fixed design, n by p.
/// @param z Frozen working response, length n.
/// @param w Frozen positive working precisions, length n.
/// @param Z Within-group random design, n by q.
/// @param group_rows Partition of the n rows.
/// @param q Random-effect dimension.
/// @return Owned inverse operators, beta, information and ML working criterion.
WeightedProfileFit compute_weighted_profile(
    const std::vector<double>&                   theta,
    const DenseMatrix<double>&                   X,
    const std::vector<double>&                   z,
    const std::vector<double>&                   w,
    const DenseMatrix<double>&                   Z,
    const std::vector<std::vector<std::size_t>>& group_rows,
    const std::size_t                            q) {
  const std::size_t         p      = X.cols();
  const DenseMatrix<double> lambda = build_lambda(theta, q);

  WeightedProfileFit result;
  result.blocks.resize(group_rows.size());
  result.xtwx = DenseMatrix<double>(p, p, 0.0);
  std::vector<double> xtwz(p, 0.0);
  double              log_det_sum = 0.0;

  for (std::size_t g = 0; g < group_rows.size(); ++g) {
    const auto&       rows = group_rows[g];
    const std::size_t ng   = rows.size();

    DenseMatrix<double> zl(ng, q, 0.0); // Z_g * Lambda
    for (std::size_t a = 0; a < ng; ++a)
      for (std::size_t k = 0; k < q; ++k) {
        double s = 0.0;
        for (std::size_t c = k; c < q; ++c)
          s += Z(rows[a], c) * lambda(c, k); // (Z Lambda)(a, k), Lambda lower triangular
        zl(a, k) = s;
      }

    DenseMatrix<double> S = DenseMatrix<double>::identity(q);
    std::vector<double> weights(ng);
    double              log_det = 0.0;
    for (std::size_t a = 0; a < ng; ++a) {
      weights[a] = w[rows[a]];
      if (!(weights[a] > 0.0) || !std::isfinite(weights[a]))
        throw std::runtime_error("GLMM: invalid working precision");
      log_det -= std::log(weights[a]);
      for (std::size_t j = 0; j < q; ++j)
        for (std::size_t k = 0; k < q; ++k)
          S(j, k) += weights[a] * zl(a, j) * zl(a, k);
    }
    auto chol_s = cholesky(S);
    if (!chol_s.ok)
      throw std::runtime_error("GLMM: working Woodbury matrix is not positive definite");
    for (std::size_t k = 0; k < q; ++k)
      log_det += 2.0 * std::log(chol_s.L(k, k));
    result.blocks[g] = WeightedGroupBlock{std::move(chol_s), std::move(zl), std::move(weights)};
    log_det_sum += log_det;

    DenseMatrix<double> Xg(ng, p, 0.0);
    std::vector<double> zg(ng);
    for (std::size_t a = 0; a < ng; ++a) {
      for (std::size_t j = 0; j < p; ++j)
        Xg(a, j) = X(rows[a], j);
      zg[a] = z[rows[a]];
    }
    const DenseMatrix<double> minv_xg = result.blocks[g].solve(Xg);
    const std::vector<double> minv_zg = result.blocks[g].solve(zg);

    for (std::size_t j1 = 0; j1 < p; ++j1) {
      for (std::size_t j2 = 0; j2 < p; ++j2) {
        double s = 0.0;
        for (std::size_t a = 0; a < ng; ++a)
          s += Xg(a, j1) * minv_xg(a, j2);
        result.xtwx(j1, j2) += s;
      }
      double sy = 0.0;
      for (std::size_t a = 0; a < ng; ++a)
        sy += Xg(a, j1) * minv_zg[a];
      xtwz[j1] += sy;
    }
  }

  auto chol_xtwx = cholesky(result.xtwx);
  if (!chol_xtwx.ok)
    throw std::runtime_error("GLMM: fixed-effects design is rank-deficient");
  result.beta = chol_xtwx.solve(xtwz);

  double rss = 0.0;
  for (std::size_t g = 0; g < group_rows.size(); ++g) {
    const auto&         rows = group_rows[g];
    const std::size_t   ng   = rows.size();
    std::vector<double> rg(ng);
    for (std::size_t a = 0; a < ng; ++a) {
      double fitted = 0.0;
      for (std::size_t j = 0; j < p; ++j)
        fitted += X(rows[a], j) * result.beta[j];
      rg[a] = z[rows[a]] - fitted;
    }
    const auto minv_rg = result.blocks[g].solve(rg);
    for (std::size_t a = 0; a < ng; ++a)
      rss += rg[a] * minv_rg[a];
  }
  result.rss = rss;

  const double n   = static_cast<double>(z.size());
  result.criterion = n * std::log(kTwoPi) + log_det_sum + rss;
  if (!std::isfinite(result.criterion))
    throw std::runtime_error("GLMM: nonfinite working criterion");
  return result;
}

/// @brief Bounded Gaussian working objective used by DE and local polishing.
class GLMMObjective : public optim::ArbitraryFunction {
 public:
  /// @brief Borrows frozen designs, pseudo-data and group partitions for one search.
  /// @param X Fixed design.
  /// @param z Working response.
  /// @param w Working precision.
  /// @param Z Random design.
  /// @param group_rows Group partition of retained rows.
  /// @param q Random-effect dimension.
  /// @param bound Covariance-factor box magnitude.
  GLMMObjective(
      const DenseMatrix<double>&                   X,
      const std::vector<double>&                   z,
      const std::vector<double>&                   w,
      const DenseMatrix<double>&                   Z,
      const std::vector<std::vector<std::size_t>>& group_rows,
      const std::size_t                            q,
      double                                       bound)
      : bound_(bound), X_(X), z_(z), w_(w), Z_(Z), group_rows_(group_rows), q_(q) {}

  /// @brief Scores a covariance factor, penalizing out-of-box or unsolvable candidates.
  /// @param theta Packed covariance factor.
  /// @return Working criterion, or a large finite failure penalty.
  double evaluate(const std::vector<double>& theta) override {
    std::size_t index = 0;
    for (std::size_t i = 0; i < q_; ++i)
      for (std::size_t j = 0; j <= i; ++j) {
        const double value = theta[index++];
        if (!std::isfinite(value) || value < (i == j ? 0.0 : -bound_) || value > bound_)
          return std::numeric_limits<double>::max() / 4.0;
      }
    try {
      return compute_weighted_profile(theta, X_, z_, w_, Z_, group_rows_, q_).criterion;
    } catch (const std::runtime_error&) {
      return std::numeric_limits<double>::max() / 4.0;
    }
  }

 private:
  double                                       bound_;      ///< Covariance-factor box bound.
  const DenseMatrix<double>&                   X_;          ///< Borrowed fixed design.
  const std::vector<double>&                   z_;          ///< Borrowed working response.
  const std::vector<double>&                   w_;          ///< Borrowed working precision.
  const DenseMatrix<double>&                   Z_;          ///< Borrowed random design.
  const std::vector<std::vector<std::size_t>>& group_rows_; ///< Borrowed row partition.
  std::size_t                                  q_;          ///< Random-effect dimension.
};

} // namespace

GLMM::GLMM(const dstruct::DataFrame& data, const std::string& formula, GLMMOptions options)
    : formula_text_(formula), fixed_formula_("y ~ 1"), options_(options) {
  fit(data);
}

std::string GLMM::family() const {
  switch (options_.family) {
    case GLMMFamily::Binomial:
      return "binomial";
    case GLMMFamily::Poisson:
      return "poisson";
  }
  return "unknown";
}

void GLMM::fit(const dstruct::DataFrame& data) {
  if (options_.family != GLMMFamily::Binomial && options_.family != GLMMFamily::Poisson)
    throw std::invalid_argument("GLMM: unsupported family");
  if (options_.max_iterations == 0 || !(options_.tol > 0.0) || !std::isfinite(options_.tol) ||
      !(options_.theta_bound > 0.0) || !std::isfinite(options_.theta_bound))
    throw std::invalid_argument("GLMM: invalid outer iteration, tolerance or covariance bound");
  if (options_.fixed_random_intercept_sd) {
    if (*options_.fixed_random_intercept_sd < 0 ||
        !std::isfinite(*options_.fixed_random_intercept_sd))
      throw std::invalid_argument("GLMM: known random-intercept SD must be finite and nonnegative");
  } else if (options_.de_population_size < 4 || options_.de_max_generations == 0)
    throw std::invalid_argument(
        "GLMM: DE requires at least four candidates and positive generations");
  const auto span = find_random_effect_span(formula_text_);
  if (!span.has_value())
    throw std::invalid_argument(
        "GLMM: formula must contain exactly one random-effects term '(... | group)'");
  const std::size_t pipe_pos = span->content.find('|');
  if (pipe_pos == std::string::npos)
    throw std::invalid_argument("GLMM: random-effects term must contain '|'");
  const std::string random_rhs = mixed_model_trim(span->content.substr(0, pipe_pos));
  group_variable_              = mixed_model_trim(span->content.substr(pipe_pos + 1));
  if (group_variable_.empty())
    throw std::invalid_argument("GLMM: missing grouping variable after '|'");

  const RandomEffectSpec spec = parse_random_effect_rhs(random_rhs);
  random_intercept_           = spec.intercept;
  random_slope_vars_          = spec.slope_vars;
  if (!random_intercept_ && random_slope_vars_.empty())
    throw std::invalid_argument("GLMM: random-effects term has no terms (no intercept, no slopes)");

  std::string fixed_text = formula_text_;
  fixed_text.erase(span->start, span->end - span->start + 1);
  fixed_text = mixed_model_trim(fixed_text);
  if (!fixed_text.empty() && fixed_text.back() == '+') {
    fixed_text.pop_back();
    fixed_text = mixed_model_trim(fixed_text);
  }
  if (!fixed_text.empty() && fixed_text.back() == '~')
    fixed_text += " 1";

  fixed_formula_         = Formula(fixed_text);
  design_                = fixed_formula_.resolve(data);
  const auto design_data = design_.build_matrix(data, /*require_response=*/true);

  if (!data.has_column(group_variable_))
    throw std::invalid_argument("GLMM: grouping column '" + group_variable_ + "' not found");
  for (const auto& v : random_slope_vars_) {
    if (!data.has_column(v))
      throw std::invalid_argument("GLMM: random-slope column '" + v + "' not found");
    if (data.column_type(v) != dstruct::DataFrame::ColumnType::Numeric)
      throw std::invalid_argument("GLMM: random-slope column '" + v + "' must be numeric");
  }

  std::vector<std::size_t> final_rows;
  std::vector<std::size_t> final_positions;
  for (std::size_t i = 0; i < design_data.used_row_indices.size(); ++i) {
    const std::size_t row = design_data.used_row_indices[i];
    if (data.is_null(group_variable_, row))
      continue;
    bool ok = true;
    for (const auto& v : random_slope_vars_) {
      if (data.is_null(v, row)) {
        ok = false;
        break;
      }
    }
    if (!ok)
      continue;
    final_rows.push_back(row);
    final_positions.push_back(i);
  }

  const std::size_t n = final_rows.size();
  const std::size_t p = design_data.X.cols();
  if (p == 0 || n <= p)
    throw std::invalid_argument("GLMM: not enough complete observations to fit the fixed effects");

  fitted_rows_   = final_rows;
  design_matrix_ = DenseMatrix<double>(n, p, 0.0);
  response_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < p; ++j) {
      const double v = design_data.X(final_positions[i], j);
      if (!std::isfinite(v))
        throw std::invalid_argument("GLMM: fixed design must be finite");
      design_matrix_(i, j) = v;
    }
    response_[i] = design_data.y[final_positions[i]];
  }
  validate_response(options_.family, response_);

  if (options_.fixed_random_intercept_sd && (!random_intercept_ || !random_slope_vars_.empty()))
    throw std::invalid_argument(
        "GLMM: a known intercept SD requires random-intercept-only structure");
  const std::size_t q = (random_intercept_ ? 1 : 0) + random_slope_vars_.size();
  random_effect_names_.clear();
  if (random_intercept_)
    random_effect_names_.emplace_back("(Intercept)");
  for (const auto& v : random_slope_vars_)
    random_effect_names_.push_back(v);

  group_is_numeric_ = data.column_type(group_variable_) == dstruct::DataFrame::ColumnType::Numeric;
  random_effect_design_ = DenseMatrix<double>(n, q, 0.0);
  group_index_.assign(n, 0);
  group_labels_.clear();
  std::unordered_map<std::string, std::size_t> group_lookup;

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t row = final_rows[i];
    const std::string key = group_key(data, group_variable_, row);
    const auto        it  = group_lookup.find(key);
    std::size_t       gidx;
    if (it == group_lookup.end()) {
      gidx = group_labels_.size();
      group_labels_.push_back(key);
      group_lookup.emplace(key, gidx);
    } else {
      gidx = it->second;
    }
    group_index_[i] = gidx;

    std::size_t col = 0;
    if (random_intercept_)
      random_effect_design_(i, col++) = 1.0;
    for (const auto& v : random_slope_vars_) {
      const double value = data.double_at(v, row);
      if (!std::isfinite(value))
        throw std::invalid_argument("GLMM: random slopes must be finite");
      random_effect_design_(i, col++) = value;
    }
  }

  if (group_labels_.size() < 2)
    throw std::invalid_argument(
        "GLMM: grouping factor '" + group_variable_ + "' must have at least 2 distinct levels");

  group_rows_.assign(group_labels_.size(), {});
  for (std::size_t i = 0; i < n; ++i)
    group_rows_[group_index_[i]].push_back(i);

  observations_ = n;

  // ---- Penalized quasi-likelihood: outer IRLS re-linearization, inner theta search ----
  std::vector<double> beta(p, 0.0);
  {
    double mean_y = 0.0;
    for (const double v : response_)
      mean_y += v;
    mean_y            = clamp_mu(options_.family, mean_y / static_cast<double>(n));
    const double eta0 = (options_.family == GLMMFamily::Binomial)
                            ? std::log(mean_y / (1.0 - mean_y))
                            : std::log(mean_y);
    if (design_.has_intercept && p > 0)
      beta[0] = eta0;
  }
  std::vector<std::vector<double>> b(group_labels_.size(), std::vector<double>(q, 0.0));

  std::vector<double> theta(q * (q + 1) / 2);
  {
    std::size_t idx = 0;
    for (std::size_t i = 0; i < q; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        theta[idx++] = (i == j) ? std::min(1.0, options_.theta_bound) : 0.0;
  }
  std::vector<double> lower(theta.size()), upper(theta.size());
  {
    std::size_t idx = 0;
    for (std::size_t i = 0; i < q; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        if (i == j) {
          lower[idx] = 0.0;
          upper[idx] = options_.theta_bound;
        } else {
          lower[idx] = -options_.theta_bound;
          upper[idx] = options_.theta_bound;
        }
        ++idx;
      }
    }
  }

  if (options_.fixed_random_intercept_sd)
    theta[0] = *options_.fixed_random_intercept_sd;
  beta_path_    = {beta};
  effects_path_ = {b};
  theta_path_   = {theta};
  criterion_path_.clear();
  change_path_.clear();
  converged_                    = false;
  std::vector<double> prev_beta = beta;
  std::vector<double> z(n), w(n);

  for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
    for (std::size_t i = 0; i < n; ++i) {
      double e = 0.0;
      for (std::size_t j = 0; j < p; ++j)
        e += design_matrix_(i, j) * beta[j];
      const auto& bg = b[group_index_[i]];
      for (std::size_t k = 0; k < q; ++k)
        e += random_effect_design_(i, k) * bg[k];
      if (!std::isfinite(e))
        throw std::runtime_error("GLMM: nonfinite working linear predictor");
      const double mu     = clamp_mu(options_.family, inverse_link(options_.family, e));
      const double gprime = link_derivative(options_.family, mu);
      z[i]                = e + (response_[i] - mu) * gprime;
      w[i]                = 1.0 / (gprime * gprime * variance_function(options_.family, mu));
    }

    const auto    old_b     = b;
    const auto    old_theta = theta;
    GLMMObjective objective(
        design_matrix_, z, w, random_effect_design_, group_rows_, q, options_.theta_bound);
    optim::DEOptions de_options;
    de_options.population_size = options_.de_population_size;
    de_options.max_generations = options_.de_max_generations;
    de_options.seed            = options_.seed;
    optim::DifferentialEvolution de(de_options);
    if (!options_.fixed_random_intercept_sd) {
      de.optimize(objective, theta, lower, upper);
      std::vector<double>      polished = theta;
      optim::NelderMeadOptions nm;
      nm.max_iterations = 1000;
      nm.tolerance      = 1e-10;
      optim::NelderMead(nm).optimize(objective, polished);
      if (objective.evaluate(polished) < objective.evaluate(theta))
        theta = polished;
      // Include the exact zero-variance boundary, which random sampling cannot hit.
      if (q == 1 && objective.evaluate(std::vector<double>{0.0}) <= objective.evaluate(theta))
        theta[0] = 0.0;
    }
    lambda_ = build_lambda(theta, q);

    const auto profile = compute_weighted_profile(
        theta, design_matrix_, z, w, random_effect_design_, group_rows_, q);
    beta = profile.beta;

    for (std::size_t g = 0; g < group_rows_.size(); ++g) {
      const auto&         rows = group_rows_[g];
      const std::size_t   ng   = rows.size();
      std::vector<double> rg(ng);
      for (std::size_t a = 0; a < ng; ++a) {
        double f = 0.0;
        for (std::size_t j = 0; j < p; ++j)
          f += design_matrix_(rows[a], j) * beta[j];
        rg[a] = z[rows[a]] - f;
      }
      const auto minv_rg = profile.blocks[g].solve(rg);

      std::vector<double> zt_minv_r(q, 0.0);
      for (std::size_t k = 0; k < q; ++k) {
        double s = 0.0;
        for (std::size_t a = 0; a < ng; ++a)
          s += random_effect_design_(rows[a], k) * minv_rg[a];
        zt_minv_r[k] = s;
      }
      std::vector<double> bg(q, 0.0);
      for (std::size_t i2 = 0; i2 < q; ++i2) {
        double s = 0.0;
        for (std::size_t k = 0; k < q; ++k) {
          double lltk = 0.0;
          for (std::size_t c = 0; c < q; ++c)
            lltk += lambda_(i2, c) * lambda_(k, c);
          s += lltk * zt_minv_r[k];
        }
        bg[i2] = s;
      }
      b[g] = bg;
    }

    double max_change = 0.0;
    /// @brief Largest coordinate change scaled by one plus its preceding magnitude.
    auto change = [&](double old, double now) {
      if (!std::isfinite(now))
        throw std::runtime_error("GLMM: nonfinite PQL state");
      max_change = std::max(max_change, std::fabs(now - old) / (1.0 + std::fabs(old)));
    };
    for (std::size_t j = 0; j < p; ++j)
      change(prev_beta[j], beta[j]);
    for (std::size_t g = 0; g < b.size(); ++g)
      for (std::size_t k = 0; k < q; ++k)
        change(old_b[g][k], b[g][k]);
    for (std::size_t k = 0; k < theta.size(); ++k)
      change(old_theta[k], theta[k]);
    prev_beta        = beta;
    iterations_used_ = iter + 1;
    beta_path_.push_back(beta);
    effects_path_.push_back(b);
    theta_path_.push_back(theta);
    change_path_.push_back(max_change);
    criterion_path_.push_back(profile.criterion);
    if (max_change < options_.tol) {
      converged_ = true;
      break;
    }
  }
  covariance_at_boundary_ = false;
  if (options_.fixed_random_intercept_sd)
    covariance_at_boundary_ = theta[0] == 0.0;
  else {
    std::size_t index = 0;
    for (std::size_t i = 0; i < q; ++i)
      for (std::size_t j = 0; j <= i; ++j) {
        const double value = theta[index++];
        if ((i == j && value <= 1e-6 * std::max(1.0, options_.theta_bound)) ||
            std::fabs(value) >= options_.theta_bound * (1 - 1e-6))
          covariance_at_boundary_ = true;
      }
  }

  coefficients_   = beta;
  random_effects_ = b;

  linear_predictors_.resize(n);
  fitted_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    double e = 0.0;
    for (std::size_t j = 0; j < p; ++j)
      e += design_matrix_(i, j) * coefficients_[j];
    const auto& bg = b[group_index_[i]];
    for (std::size_t k = 0; k < q; ++k)
      e += random_effect_design_(i, k) * bg[k];
    linear_predictors_[i] = e;
    const double mu       = clamp_mu(options_.family, inverse_link(options_.family, e));
    fitted_[i]            = mu;
    const double gprime   = link_derivative(options_.family, mu);
    z[i]                  = e + (response_[i] - mu) * gprime;
    w[i]                  = 1.0 / (gprime * gprime * variance_function(options_.family, mu));
  }

  const auto final_profile =
      compute_weighted_profile(theta, design_matrix_, z, w, random_effect_design_, group_rows_, q);
  deviance_ = final_profile.criterion;

  fixed_effect_covariance_ = cholesky(final_profile.xtwx).solve(DenseMatrix<double>::identity(p));
  standard_errors_.resize(p);
  z_values_.resize(p);
  p_values_.resize(p);
  for (std::size_t j = 0; j < p; ++j) {
    standard_errors_[j] = std::sqrt(std::max(0.0, fixed_effect_covariance_(j, j)));
    z_values_[j]        = coefficients_[j] / standard_errors_[j];
    p_values_[j]        = 2.0 * (1.0 - random::normal_cdf(std::abs(z_values_[j])));
  }

  random_effect_covariance_ = DenseMatrix<double>(q, q, 0.0);
  for (std::size_t i = 0; i < q; ++i)
    for (std::size_t j = 0; j < q; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < q; ++k)
        s += lambda_(i, k) * lambda_(j, k);
      random_effect_covariance_(i, j) = s;
    }
}

std::vector<double> GLMM::random_effect_std_devs() const {
  std::vector<double> sd(random_effect_covariance_.rows());
  for (std::size_t i = 0; i < sd.size(); ++i)
    sd[i] = std::sqrt(std::max(0.0, random_effect_covariance_(i, i)));
  return sd;
}

double GLMM::random_effect_correlation(const std::size_t i, const std::size_t j) const {
  if (i >= random_effect_names_.size() || j >= random_effect_names_.size())
    throw std::out_of_range("GLMM: random-effect correlation coordinate out of range");
  const double sii = random_effect_covariance_(i, i);
  const double sjj = random_effect_covariance_(j, j);
  if (sii <= 0.0 || sjj <= 0.0)
    return 0.0;
  return random_effect_covariance_(i, j) / std::sqrt(sii * sjj);
}

double GLMM::aic() const {
  const std::size_t q = random_effect_names_.size();
  const std::size_t k =
      coefficients_.size() + (options_.fixed_random_intercept_sd ? 0 : q * (q + 1) / 2);
  return deviance_ + 2.0 * static_cast<double>(k);
}

double GLMM::bic() const {
  const std::size_t q = random_effect_names_.size();
  const std::size_t k =
      coefficients_.size() + (options_.fixed_random_intercept_sd ? 0 : q * (q + 1) / 2);
  return deviance_ + std::log(static_cast<double>(observations_)) * static_cast<double>(k);
}

std::string GLMM::summary() const {
  std::ostringstream oss;
  print_summary(oss);
  return oss.str();
}

void GLMM::print_summary() const {
  print_summary(std::cout);
}

void GLMM::print_summary(std::ostream& os) const {
  const std::size_t q = random_effect_names_.size();
  os << "Generalized linear mixed model fit by PQL (" << family() << ", "
     << (options_.family == GLMMFamily::Binomial ? "logit" : "log") << " link)\n";
  os << "Formula: " << formula_text_ << "\n\n";

  os << "Working-Gaussian quantities (not marginal GLMM likelihood criteria):\n";
  os << "  working AIC  working BIC  working criterion\n";
  os << std::fixed << std::setprecision(1);
  os << ' ' << std::setw(8) << aic() << ' ' << std::setw(8) << bic() << ' ' << std::setw(8)
     << deviance_ << "\n\n";

  const auto sd = random_effect_std_devs();
  os << "Random effects:\n";
  os << ' ' << std::left << std::setw(9) << "Groups" << std::setw(13) << "Name" << std::right
     << std::setw(10) << "Variance" << std::setw(10) << "Std.Dev." << std::setw(7) << "Corr"
     << "\n";
  os << std::setprecision(4);
  for (std::size_t i = 0; i < q; ++i) {
    os << ' ' << std::left << std::setw(9) << (i == 0 ? group_variable_ : std::string())
       << std::setw(13) << random_effect_names_[i] << std::right << std::setw(10)
       << random_effect_covariance_(i, i) << std::setw(10) << sd[i];
    for (std::size_t j = 0; j < i; ++j)
      os << std::setw(7) << random_effect_correlation(i, j);
    os << "\n";
  }
  os << "Number of obs: " << observations_ << ", groups: " << group_variable_ << ", "
     << group_labels_.size() << "\n";
  os << "PQL outer iterations used: " << iterations_used_
     << "; converged: " << (converged_ ? "yes" : "no") << "\n";
  os << "Covariance at boundary: " << (covariance_at_boundary_ ? "yes" : "no") << "\n\n";

  os << "Fixed effects:\n";
  os << std::left << std::setw(16) << "" << std::right << std::setw(12) << "Estimate"
     << std::setw(12) << "Std.Error" << std::setw(10) << "z value" << std::setw(12) << "Pr(>|z|)"
     << "\n";
  for (std::size_t j = 0; j < coefficients_.size(); ++j) {
    os << std::left << std::setw(16) << design_.coefficient_names[j] << std::right << std::setw(12)
       << coefficients_[j] << std::setw(12) << standard_errors_[j] << std::setw(10) << z_values_[j]
       << std::setw(12) << p_values_[j] << "\n";
  }
}

std::vector<double> GLMM::predict_fixed(const dstruct::DataFrame& newdata) const {
  design_.validate_categorical_levels(newdata, "GLMM::predict_fixed");
  const auto          raw = design_.build_matrix(newdata, false);
  std::vector<double> out(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < raw.X.rows(); ++i) {
    double eta = 0.0;
    for (std::size_t j = 0; j < rank(); ++j) {
      if (!std::isfinite(raw.X(i, j)))
        throw std::invalid_argument("GLMM: query predictors must be finite");
      eta += raw.X(i, j) * coefficients_[j];
    }
    if (!std::isfinite(eta))
      throw std::invalid_argument("GLMM: query linear predictor overflow");
    const double mu = inverse_link(options_.family, eta);
    if (!std::isfinite(mu))
      throw std::invalid_argument("GLMM: query mean overflow");
    out[raw.used_row_indices[i]] = mu;
  }
  return out;
}

dstruct::DataFrame GLMM::prediction_frame(const dstruct::DataFrame& newdata) const {
  const auto                                   fixed = predict_fixed(newdata);
  auto                                         fit   = fixed;
  std::vector<std::string>                     status(newdata.nrows(), "incomplete");
  std::unordered_map<std::string, std::size_t> lookup;
  for (std::size_t g = 0; g < num_groups(); ++g)
    lookup.emplace(group_labels_[g], g);
  // Use the design predictor, not a forward link of the possibly saturated mean.
  const auto raw = design_.build_matrix(newdata, false);
  for (std::size_t i = 0; i < raw.X.rows(); ++i) {
    const auto row = raw.used_row_indices[i];
    status[row]    = "missing";
    if (!newdata.has_column(group_variable_) || newdata.is_null(group_variable_, row))
      continue;
    const bool numeric =
        newdata.column_type(group_variable_) == dstruct::DataFrame::ColumnType::Numeric;
    if (numeric != group_is_numeric_)
      throw std::invalid_argument("GLMM: query grouping column type differs from training");
    const auto it = lookup.find(group_key(newdata, group_variable_, row));
    if (it == lookup.end()) {
      status[row] = "unseen";
      continue;
    }
    status[row] = "known";
    double eta  = 0.0;
    for (std::size_t j = 0; j < rank(); ++j)
      eta += raw.X(i, j) * coefficients_[j];
    const auto& b = random_effects_[it->second];
    std::size_t k = 0;
    if (random_intercept_)
      eta += b[k++];
    bool complete = true;
    for (const auto& v : random_slope_vars_) {
      if (!newdata.has_column(v))
        throw std::invalid_argument("GLMM: query missing random slope " + v);
      if (newdata.is_null(v, row)) {
        complete = false;
        break;
      }
      const double x = newdata.double_at(v, row);
      if (!std::isfinite(x))
        throw std::invalid_argument("GLMM: query random slopes must be finite");
      eta += x * b[k++];
    }
    if (!complete) {
      fit[row]    = std::numeric_limits<double>::quiet_NaN();
      status[row] = "incomplete";
      continue;
    }
    if (!std::isfinite(eta))
      throw std::invalid_argument("GLMM: query linear predictor overflow");
    fit[row] = inverse_link(options_.family, eta);
    if (!std::isfinite(fit[row]))
      throw std::invalid_argument("GLMM: query mean overflow");
  }
  dstruct::DataFrame out;
  out.add_column("fit", fit);
  out.add_column("fixed_fit", fixed);
  out.add_column("group_status", status);
  return out;
}

std::vector<double> GLMM::predict(const dstruct::DataFrame& newdata) const {
  return prediction_frame(newdata).numeric_column("fit").values();
}

dstruct::DataFrame GLMM::coefficient_frame() const {
  dstruct::DataFrame out;
  out.add_column("term", coefficient_names());
  out.add_column("estimate", coefficients_);
  out.add_column("std_error", standard_errors_);
  out.add_column("z_value", z_values_);
  out.add_column("p_value", p_values_);
  return out;
}

dstruct::DataFrame GLMM::random_effects_frame() const {
  dstruct::DataFrame out;
  out.add_column("group", group_labels_);
  std::vector<double> n(num_groups());
  for (std::size_t g = 0; g < num_groups(); ++g)
    n[g] = static_cast<double>(group_rows_[g].size());
  out.add_column("n", n);
  for (std::size_t k = 0; k < random_effect_names_.size(); ++k) {
    std::vector<double> column(num_groups());
    for (std::size_t g = 0; g < num_groups(); ++g)
      column[g] = random_effects_[g][k];
    out.add_column(random_effect_names_[k], column);
  }
  return out;
}

dstruct::DataFrame GLMM::working_data(std::size_t iteration) const {
  if (iteration >= iterations_used_)
    throw std::out_of_range("GLMM::working_data: iteration out of range");
  std::vector<double> row(observations_), eta(observations_), mu(observations_),
      weight(observations_), z(observations_);
  std::vector<std::string> group(observations_);
  for (std::size_t i = 0; i < observations_; ++i) {
    row[i]   = static_cast<double>(fitted_rows_[i]);
    group[i] = group_labels_[group_index_[i]];
    eta[i]   = 0.0;
    for (std::size_t j = 0; j < rank(); ++j)
      eta[i] += design_matrix_(i, j) * beta_path_[iteration][j];
    for (std::size_t k = 0; k < random_effect_names_.size(); ++k)
      eta[i] += random_effect_design_(i, k) * effects_path_[iteration][group_index_[i]][k];
    mu[i]           = clamp_mu(options_.family, inverse_link(options_.family, eta[i]));
    const double gp = link_derivative(options_.family, mu[i]);
    weight[i]       = 1.0 / (gp * gp * variance_function(options_.family, mu[i]));
    z[i]            = eta[i] + (response_[i] - mu[i]) * gp;
  }
  dstruct::DataFrame out;
  out.add_column("row", row);
  out.add_column("group", group);
  out.add_column("response", response_);
  out.add_column("eta", eta);
  out.add_column("mu", mu);
  out.add_column("weight", weight);
  out.add_column("z", z);
  return out;
}

dstruct::DataFrame GLMM::iteration_history() const {
  dstruct::DataFrame  out;
  std::vector<double> index(iterations_used_);
  for (std::size_t i = 0; i < iterations_used_; ++i)
    index[i] = static_cast<double>(i + 1);
  out.add_column("iteration", index);
  out.add_column("max_relative_change", change_path_);
  out.add_column("working_criterion", criterion_path_);
  for (std::size_t j = 0; j < rank(); ++j) {
    std::vector<double> column(iterations_used_);
    for (std::size_t i = 0; i < iterations_used_; ++i)
      column[i] = beta_path_[i + 1][j];
    out.add_column("beta:" + coefficient_names()[j], column);
  }
  for (std::size_t k = 0; k < random_effect_names_.size(); ++k) {
    std::vector<double> column(iterations_used_);
    for (std::size_t i = 0; i < iterations_used_; ++i) {
      const auto l = build_lambda(theta_path_[i + 1], random_effect_names_.size());
      double     v = 0.0;
      for (std::size_t j = 0; j <= k; ++j)
        v += l(k, j) * l(k, j);
      column[i] = std::sqrt(v);
    }
    out.add_column("sd:" + random_effect_names_[k], column);
  }
  return out;
}

double GLMM::working_criterion_at(const std::vector<double>& theta, std::size_t iteration) const {
  const auto q = random_effect_names_.size();
  if (theta.size() != q * (q + 1) / 2)
    throw std::invalid_argument("GLMM: wrong covariance-factor dimension");
  for (double x : theta)
    if (!std::isfinite(x))
      throw std::invalid_argument("GLMM: factor entries must be finite");
  const auto d = working_data(iteration);
  return compute_weighted_profile(
             theta,
             design_matrix_,
             d.numeric_column("z").values(),
             d.numeric_column("weight").values(),
             random_effect_design_,
             group_rows_,
             q)
      .criterion;
}

dstruct::DataFrame GLMM::binomial_score(const dstruct::DataFrame& data, double threshold) const {
  if (options_.family != GLMMFamily::Binomial || !(threshold > 0 && threshold < 1) ||
      !std::isfinite(threshold))
    throw std::invalid_argument(
        "GLMM::binomial_score: binary model and threshold in (0,1) required");
  const auto raw         = design_.build_matrix(data, true);
  const auto predictions = predict(data.take_rows(raw.used_row_indices));
  double     n = 0, correct = 0, brier = 0, loss = 0;
  for (std::size_t i = 0; i < raw.y.size(); ++i) {
    const double p = predictions[i], y = raw.y[i];
    if (std::isnan(p))
      continue;
    validate_response(GLMMFamily::Binomial, std::vector<double>{y});
    const double probability = std::clamp(
        p, std::numeric_limits<double>::epsilon(), 1.0 - std::numeric_limits<double>::epsilon());
    n += 1;
    correct += static_cast<double>((p >= threshold) == (y == 1));
    brier += (p - y) * (p - y);
    loss -= y * std::log(probability) + (1 - y) * std::log1p(-probability);
  }
  if (n == 0)
    throw std::runtime_error("GLMM::binomial_score: no complete scoring rows");
  dstruct::DataFrame out;
  out.add_column("n", std::vector<double>{n});
  out.add_column("accuracy", std::vector<double>{correct / n});
  out.add_column("brier", std::vector<double>{brier / n});
  out.add_column("log_loss", std::vector<double>{loss / n});
  return out;
}

dstruct::DataFrame GLMM::training_details() const {
  std::vector<double>      row(observations_), fixed(observations_), residual(observations_);
  std::vector<std::string> group(observations_);
  for (std::size_t i = 0; i < observations_; ++i) {
    row[i]     = static_cast<double>(fitted_rows_[i]);
    group[i]   = group_labels_[group_index_[i]];
    double eta = 0.0;
    for (std::size_t j = 0; j < rank(); ++j)
      eta += design_matrix_(i, j) * coefficients_[j];
    fixed[i] = inverse_link(options_.family, eta);
    residual[i] =
        (response_[i] - fitted_[i]) / std::sqrt(variance_function(options_.family, fitted_[i]));
  }
  dstruct::DataFrame out;
  out.add_column("row", row);
  out.add_column("group", group);
  out.add_column("response", response_);
  out.add_column("fitted", fitted_);
  out.add_column("fixed_fit", fixed);
  out.add_column("eta", linear_predictors_);
  out.add_column("pearson_residual", residual);
  return out;
}

dstruct::DataFrame GLMM::model_summary() const {
  dstruct::DataFrame out;
  out.add_column("family", std::vector<std::string>{family()});
  out.add_column("n", std::vector<double>{static_cast<double>(observations_)});
  out.add_column("groups", std::vector<double>{static_cast<double>(num_groups())});
  out.add_column("p", std::vector<double>{static_cast<double>(rank())});
  out.add_column("q", std::vector<double>{static_cast<double>(random_effect_names_.size())});
  out.add_column("iterations", std::vector<double>{static_cast<double>(iterations_used_)});
  out.add_column("converged", std::vector<std::string>{converged_ ? "yes" : "no"});
  out.add_column(
      "covariance_boundary", std::vector<std::string>{covariance_at_boundary_ ? "yes" : "no"});
  out.add_column("working_criterion", std::vector<double>{deviance_});
  return out;
}

dstruct::DataFrame simulate_glmm_random_intercepts(
    GLMMFamily    family,
    std::size_t   groups,
    std::size_t   observations_per_group,
    double        intercept,
    double        slope,
    double        random_intercept_sd,
    std::uint64_t seed) {
  if ((family != GLMMFamily::Binomial && family != GLMMFamily::Poisson) || groups < 2 ||
      observations_per_group < 2 || !std::isfinite(intercept) || !std::isfinite(slope) ||
      !std::isfinite(random_intercept_sd) || random_intercept_sd < 0 ||
      groups > std::numeric_limits<std::size_t>::max() / observations_per_group)
    throw std::invalid_argument("simulate_glmm_random_intercepts: invalid parameters");
  std::mt19937_64                        rng(seed);
  std::normal_distribution<double>       normal;
  std::uniform_real_distribution<double> uniform;
  std::vector<double>                    effects(groups);
  for (auto& b : effects)
    b = random_intercept_sd * normal(rng);
  std::vector<double>      x, y, truth, mean;
  std::vector<std::string> labels;
  const auto               n = groups * observations_per_group;
  x.reserve(n);
  y.reserve(n);
  truth.reserve(n);
  mean.reserve(n);
  labels.reserve(n);
  for (std::size_t g = 0; g < groups; ++g)
    for (std::size_t j = 0; j < observations_per_group; ++j) {
      const double value =
          -1.0 + 2.0 * static_cast<double>(j) / static_cast<double>(observations_per_group - 1);
      const double eta = intercept + slope * value + effects[g];
      const double mu  = inverse_link(family, eta);
      if (!std::isfinite(eta) || !std::isfinite(mu) ||
          (family == GLMMFamily::Poisson && (mu > 1e6 || mu <= 0)))
        throw std::invalid_argument(
            "simulate_glmm_random_intercepts: generating mean outside supported range");
      x.push_back(value);
      truth.push_back(effects[g]);
      mean.push_back(mu);
      labels.push_back("g" + std::to_string(g + 1));
      if (family == GLMMFamily::Binomial)
        y.push_back(uniform(rng) < mu ? 1.0 : 0.0);
      else {
        std::poisson_distribution<int> poisson(mu);
        y.push_back(static_cast<double>(poisson(rng)));
      }
    }
  dstruct::DataFrame out;
  out.add_column("group", labels);
  out.add_column("x", x);
  out.add_column("y", y);
  out.add_column("true_effect", truth);
  out.add_column("true_mean", mean);
  return out;
}

} // namespace datamunge::stats
