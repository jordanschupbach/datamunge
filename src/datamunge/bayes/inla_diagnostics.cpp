/// @file inla_diagnostics.cpp
/// @brief INLA diagnostic tables, retained marginal mixtures and direct scalar checks.
#include <cmath>
#include <datamunge/bayes/distributions.hpp>
#include <datamunge/bayes/inla_diagnostics.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::bayes {
/// @brief Private result validation and scalar-density adapter.
namespace {
/// @brief Check shapes, normalization and Gaussian components of a retained mixture.
/// @param fit Complete result from INLA::fit().
/// @throws std::invalid_argument Invalid/mismatched/nonfinite arrays or component variances.
void validate_mixture(const INLAResult& fit) {
  const auto k = fit.theta_grid_weights.size(), p = fit.latent_mean.size();
  if (!k || !p || fit.theta_dim == 0 || fit.theta_grid.size() != k * fit.theta_dim ||
      fit.conditional_means.size() != k || fit.conditional_covariances.size() != k)
    throw std::invalid_argument("INLA diagnostics: incompatible mixture shapes");
  double total = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const double w = fit.theta_grid_weights[i];
    if (!std::isfinite(w) || w < 0 || fit.conditional_means[i].size() != p ||
        fit.conditional_covariances[i].rows() != p || fit.conditional_covariances[i].cols() != p)
      throw std::invalid_argument("INLA diagnostics: invalid mixture component");
    total += w;
    for (std::size_t j = 0; j < p; ++j)
      if (!std::isfinite(fit.conditional_means[i][j]) ||
          !std::isfinite(fit.conditional_covariances[i](j, j)) ||
          fit.conditional_covariances[i](j, j) <= 0)
        throw std::invalid_argument("INLA diagnostics: invalid Gaussian moments");
  }
  if (!std::isfinite(total) || std::abs(total - 1) > 1e-9)
    throw std::invalid_argument("INLA diagnostics: weights not normalized");
}
/// @brief Borrowed scalar-intercept model, integrated independently of conditional mode-finding.
class ScalarDensity final : public optim::ArbitraryFunction {
 public:
  /// @brief Borrow the model inputs until grid_posterior() returns.
  /// @param y Observed outcomes.
  /// @param likelihood Observation log-density callback.
  /// @param q Scalar Gaussian precision.
  /// @param theta Fixed callback hyperparameters.
  ScalarDensity(
      const std::vector<double>& y,
      const INLALikelihoodFn&    likelihood,
      double                     q,
      const std::vector<double>& theta)
      : y_(y), likelihood_(likelihood), q_(q), theta_(theta) {}
  /// @brief Evaluate the actual normalized-prior log posterior at scalar x.
  /// @param x One latent intercept.
  /// @return Log likelihood plus normal log prior, without Laplace approximation.
  double evaluate(const std::vector<double>& x) override {
    double lp = normal_lpdf(x[0], 0, 1 / std::sqrt(q_));
    for (double y : y_)
      lp += likelihood_(y, x[0], theta_).log_density;
    return lp;
  }

 private:
  const std::vector<double>& y_;          ///< Borrowed observations.
  const INLALikelihoodFn&    likelihood_; ///< Borrowed observation callback.
  double                     q_;          ///< Prior precision.
  const std::vector<double>& theta_;      ///< Borrowed fixed hyperparameters.
};
} // namespace
dstruct::DataFrame inla_conditional_table(const INLAConditional& fit) {
  const auto p = fit.mode.size();
  if (!p || fit.covariance.rows() != p || fit.covariance.cols() != p)
    throw std::invalid_argument("INLA diagnostics: invalid conditional shape");
  std::vector<double> index, sd;
  for (std::size_t j = 0; j < p; ++j) {
    if (!std::isfinite(fit.mode[j]) || !std::isfinite(fit.covariance(j, j)) ||
        fit.covariance(j, j) <= 0)
      throw std::invalid_argument("INLA diagnostics: invalid conditional variance/mode");
    index.push_back(j);
    sd.push_back(std::sqrt(fit.covariance(j, j)));
  }
  dstruct::DataFrame out;
  out.add_column("component", index);
  out.add_column("mode", fit.mode);
  out.add_column("sd", sd);
  out.add_column("iterations", std::vector<double>(p, fit.iterations));
  out.add_column("gradient_inf", std::vector<double>(p, fit.gradient_inf));
  out.add_column("conditional_log_evidence", std::vector<double>(p, fit.log_marginal_likelihood));
  return out;
}
dstruct::DataFrame inla_latent_summary(const INLAResult& fit) {
  const auto p = fit.latent_mean.size();
  if (fit.latent_sd.size() != p || fit.latent_within_variance.size() != p ||
      fit.latent_between_variance.size() != p)
    throw std::invalid_argument("INLA diagnostics: moment shape mismatch");
  std::vector<double> index, total;
  for (std::size_t j = 0; j < p; ++j) {
    if (!std::isfinite(fit.latent_mean[j]) || !std::isfinite(fit.latent_sd[j]) ||
        fit.latent_sd[j] < 0 || !std::isfinite(fit.latent_within_variance[j]) ||
        fit.latent_within_variance[j] < 0 || !std::isfinite(fit.latent_between_variance[j]) ||
        fit.latent_between_variance[j] < 0)
      throw std::invalid_argument("INLA diagnostics: invalid mixture moments");
    index.push_back(j);
    total.push_back(fit.latent_sd[j] * fit.latent_sd[j]);
  }
  dstruct::DataFrame out;
  out.add_column("component", index);
  out.add_column("mean", fit.latent_mean);
  out.add_column("sd", fit.latent_sd);
  out.add_column("within_variance", fit.latent_within_variance);
  out.add_column("between_variance", fit.latent_between_variance);
  out.add_column("total_variance", total);
  return out;
}
dstruct::DataFrame inla_grid_table(const INLAResult& fit) {
  validate_mixture(fit);
  const auto k = fit.theta_grid_weights.size(), p = fit.latent_mean.size();
  if (fit.theta_grid_cell_volumes.size() != k || fit.theta_log_posterior.size() != k ||
      fit.conditional_iterations.size() != k || fit.conditional_gradient_inf.size() != k)
    throw std::invalid_argument("INLA diagnostics: grid diagnostic shape mismatch");
  dstruct::DataFrame out;
  for (std::size_t j = 0; j < fit.theta_dim; ++j) {
    std::vector<double> values;
    for (std::size_t i = 0; i < k; ++i)
      values.push_back(fit.theta_grid[i * fit.theta_dim + j]);
    out.add_column("theta_" + std::to_string(j), values);
  }
  out.add_column("weight", fit.theta_grid_weights);
  out.add_column("cell_volume", fit.theta_grid_cell_volumes);
  out.add_column("log_posterior", fit.theta_log_posterior);
  out.add_column(
      "iterations",
      std::vector<double>(fit.conditional_iterations.begin(), fit.conditional_iterations.end()));
  out.add_column("gradient_inf", fit.conditional_gradient_inf);
  for (std::size_t j = 0; j < p; ++j) {
    std::vector<double> values;
    for (std::size_t i = 0; i < k; ++i)
      values.push_back(fit.conditional_means[i][j]);
    out.add_column("mode_" + std::to_string(j), values);
  }
  return out;
}
dstruct::DataFrame inla_latent_density(
    const INLAResult&          fit,
    std::size_t                component,
    const std::vector<double>& values) {
  validate_mixture(fit);
  if (component >= fit.latent_mean.size())
    throw std::invalid_argument("INLA diagnostics: invalid component");
  std::vector<double> density;
  for (double v : values) {
    if (!std::isfinite(v))
      throw std::invalid_argument("INLA diagnostics: nonfinite evaluation point");
    double d = 0;
    for (std::size_t k = 0; k < fit.theta_grid_weights.size(); ++k)
      d += fit.theta_grid_weights[k] *
           std::exp(normal_lpdf(
               v,
               fit.conditional_means[k][component],
               std::sqrt(fit.conditional_covariances[k](component, component))));
    density.push_back(d);
  }
  dstruct::DataFrame out;
  out.add_column("value", values);
  out.add_column("density", density);
  return out;
}
double inla_latent_quantile(const INLAResult& fit, std::size_t component, double probability) {
  validate_mixture(fit);
  if (component >= fit.latent_mean.size() || !std::isfinite(probability) || probability <= 0 ||
      probability >= 1)
    throw std::invalid_argument("INLA quantile: invalid coordinate or probability");
  double lower = std::numeric_limits<double>::infinity(), upper = -lower;
  for (std::size_t i = 0; i < fit.theta_grid_weights.size(); ++i) {
    const double m  = fit.conditional_means[i][component],
                 sd = std::sqrt(fit.conditional_covariances[i](component, component));
    lower           = std::min(lower, m - 10 * sd);
    upper           = std::max(upper, m + 10 * sd);
  }
  // Mixture CDF uses erfc to retain stable Gaussian tail probabilities.
  auto cdf = [&](double x) {
    double p = 0;
    for (std::size_t i = 0; i < fit.theta_grid_weights.size(); ++i)
      p += fit.theta_grid_weights[i] * .5 *
           std::erfc(
               (fit.conditional_means[i][component] - x) /
               std::sqrt(2 * fit.conditional_covariances[i](component, component)));
    return p;
  };
  while (cdf(lower) > probability)
    lower -= upper - lower;
  while (cdf(upper) < probability)
    upper += upper - lower;
  for (int iteration = 0; iteration < 120; ++iteration) {
    const double middle = lower / 2 + upper / 2;
    if (middle == lower || middle == upper)
      break;
    if (cdf(middle) < probability)
      lower = middle;
    else
      upper = middle;
  }
  return lower / 2 + upper / 2;
}
GridPosterior inla_scalar_reference(
    const std::vector<double>& y,
    const INLALikelihoodFn&    likelihood,
    double                     precision,
    const std::vector<double>& theta,
    double                     lower,
    double                     upper,
    std::size_t                points) {
  if (y.empty() || !likelihood || !std::isfinite(precision) || precision <= 0)
    throw std::invalid_argument("INLA reference: nonempty model and positive precision required");
  for (double v : y)
    if (!std::isfinite(v))
      throw std::invalid_argument("INLA reference: nonfinite response");
  for (double v : theta)
    if (!std::isfinite(v))
      throw std::invalid_argument("INLA reference: nonfinite theta");
  ScalarDensity model(y, likelihood, precision, theta);
  return grid_posterior(model, {lower}, {upper}, points);
}
} // namespace datamunge::bayes
