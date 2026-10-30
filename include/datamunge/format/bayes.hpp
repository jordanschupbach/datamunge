#pragma once

/// @file bayes.hpp
/// @brief std::format / std::print support for datamunge::bayes.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Sampler results print a compact draw summary (number of draws, dimension, per-parameter
/// posterior mean and standard deviation) rather than the raw chains. Samplers print their
/// configured options under the sampler's own name, e.g. "HMC(num_samples=1000, ...)".
///
/// @par Example
/// @code
/// dm::bayes::HMCResult r;
/// r.samples = {{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0}};
/// r.accept_rate = 0.75;
/// r.final_step_size = 0.5;
/// std::print("{}\n", r);
/// // HMCResult(draws=3, dim=2, mean=[2, 20], sd=[1, 10], accept_rate=0.75, final_step_size=0.5)
/// @endcode

#include <datamunge/bayes/bayes.hpp>
#include <datamunge/format/optim.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <cmath>
#include <cstddef>
#include <format>
#include <string>
#include <vector>

namespace datamunge::format_detail {

/// @brief "draws=n, dim=d, mean=[...], sd=[...]" for a set of MCMC draws (sd omitted below 2 draws).
/// @param samples One vector per draw, all the same length.
/// @return The compact per-parameter summary.
inline std::string bayes_draws_text(const std::vector<std::vector<double>>& samples) {
  const std::size_t n = samples.size();
  if (n == 0) return "draws=0";
  const std::size_t d = samples.front().size();
  std::vector<double> mean(d, 0.0), sd(d, 0.0);
  for (const auto& s : samples)
    for (std::size_t j = 0; j < d && j < s.size(); ++j) mean[j] += s[j];
  for (double& m : mean) m /= static_cast<double>(n);
  std::string out = std::format("draws={}, dim={}, mean={}", n, d, preview(mean));
  if (n < 2) return out;
  for (const auto& s : samples)
    for (std::size_t j = 0; j < d && j < s.size(); ++j) sd[j] += (s[j] - mean[j]) * (s[j] - mean[j]);
  for (double& v : sd) v = std::sqrt(v / static_cast<double>(n - 1));
  return out + ", sd=" + preview(sd);
}

}  // namespace datamunge::format_detail

/// @brief Prints a bayes::INLAIntegrationStrategy as its enumerator name ("Grid" or "EmpiricalBayes").
template <>
struct std::formatter<datamunge::bayes::INLAIntegrationStrategy> : datamunge::format_detail::text_formatter<datamunge::bayes::INLAIntegrationStrategy> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or "INLAIntegrationStrategy(<int>)" for an unknown value.
  static std::string text(datamunge::bayes::INLAIntegrationStrategy x) {
    switch (x) {
      case datamunge::bayes::INLAIntegrationStrategy::Grid: return "Grid";
      case datamunge::bayes::INLAIntegrationStrategy::EmpiricalBayes: return "EmpiricalBayes";
    }
    return std::format("INLAIntegrationStrategy({})", static_cast<int>(x));
  }
};

/// @brief Prints bayes::HMCOptions as one line of its fields.
template <>
struct std::formatter<datamunge::bayes::HMCOptions> : datamunge::format_detail::text_formatter<datamunge::bayes::HMCOptions> {
  /// @brief One line of its fields.
  /// @param x The value to print.
  /// @return "HMCOptions(num_samples=..., num_warmup=..., ...)".
  static std::string text(const datamunge::bayes::HMCOptions& x) {
    return std::format("HMCOptions(num_samples={}, num_warmup={}, num_leapfrog_steps={}, initial_step_size={}, "
                       "target_accept_rate={}, seed={})",
                       x.num_samples, x.num_warmup, x.num_leapfrog_steps, x.initial_step_size, x.target_accept_rate, x.seed);
  }
};

/// @brief Prints a bayes::HMCResult as a draw summary plus its adaptation diagnostics.
template <>
struct std::formatter<datamunge::bayes::HMCResult> : datamunge::format_detail::text_formatter<datamunge::bayes::HMCResult> {
  /// @brief A draw summary plus its adaptation diagnostics.
  /// @param x The value to print.
  /// @return "HMCResult(draws=n, dim=d, mean=[...], sd=[...], accept_rate=a, final_step_size=s)".
  static std::string text(const datamunge::bayes::HMCResult& x) {
    return std::format("HMCResult({}, accept_rate={}, final_step_size={})",
                       datamunge::format_detail::bayes_draws_text(x.samples), x.accept_rate, x.final_step_size);
  }
};

/// @brief Prints bayes::NUTSOptions as one line of its fields.
template <>
struct std::formatter<datamunge::bayes::NUTSOptions> : datamunge::format_detail::text_formatter<datamunge::bayes::NUTSOptions> {
  /// @brief One line of its fields.
  /// @param x The value to print.
  /// @return "NUTSOptions(num_samples=..., num_warmup=..., ...)".
  static std::string text(const datamunge::bayes::NUTSOptions& x) {
    return std::format("NUTSOptions(num_samples={}, num_warmup={}, max_tree_depth={}, initial_step_size={}, "
                       "target_accept_rate={}, max_delta_error={}, seed={})",
                       x.num_samples, x.num_warmup, x.max_tree_depth, x.initial_step_size, x.target_accept_rate,
                       x.max_delta_error, x.seed);
  }
};

/// @brief Prints a bayes::NUTSResult as a draw summary plus its diagnostics.
template <>
struct std::formatter<datamunge::bayes::NUTSResult> : datamunge::format_detail::text_formatter<datamunge::bayes::NUTSResult> {
  /// @brief A draw summary plus its diagnostics.
  /// @param x The value to print.
  /// @return "NUTSResult(draws=n, dim=d, mean=[...], sd=[...], accept_rate=a, final_step_size=s, divergences=k)".
  static std::string text(const datamunge::bayes::NUTSResult& x) {
    return std::format("NUTSResult({}, accept_rate={}, final_step_size={}, divergences={})",
                       datamunge::format_detail::bayes_draws_text(x.samples), x.accept_rate, x.final_step_size,
                       x.num_divergences);
  }
};

/// @brief Prints bayes::RWMOptions as one line of its fields.
template <>
struct std::formatter<datamunge::bayes::RWMOptions> : datamunge::format_detail::text_formatter<datamunge::bayes::RWMOptions> {
  /// @brief One line of its fields.
  /// @param x The value to print.
  /// @return "RWMOptions(num_samples=..., num_warmup=..., ...)".
  static std::string text(const datamunge::bayes::RWMOptions& x) {
    return std::format("RWMOptions(num_samples={}, num_warmup={}, initial_step_size={}, target_accept_rate={}, seed={})",
                       x.num_samples, x.num_warmup, x.initial_step_size, x.target_accept_rate, x.seed);
  }
};

/// @brief Prints a bayes::RWMResult as a draw summary plus its adaptation diagnostics.
template <>
struct std::formatter<datamunge::bayes::RWMResult> : datamunge::format_detail::text_formatter<datamunge::bayes::RWMResult> {
  /// @brief A draw summary plus its adaptation diagnostics.
  /// @param x The value to print.
  /// @return "RWMResult(draws=n, dim=d, mean=[...], sd=[...], accept_rate=a, final_step_size=s)".
  static std::string text(const datamunge::bayes::RWMResult& x) {
    return std::format("RWMResult({}, accept_rate={}, final_step_size={})",
                       datamunge::format_detail::bayes_draws_text(x.samples), x.accept_rate, x.final_step_size);
  }
};

/// @brief Prints bayes::GibbsOptions as one line of its fields.
template <>
struct std::formatter<datamunge::bayes::GibbsOptions> : datamunge::format_detail::text_formatter<datamunge::bayes::GibbsOptions> {
  /// @brief One line of its fields.
  /// @param x The value to print.
  /// @return "GibbsOptions(num_samples=..., num_warmup=..., ...)".
  static std::string text(const datamunge::bayes::GibbsOptions& x) {
    return std::format("GibbsOptions(num_samples={}, num_warmup={}, initial_step_sizes={}, target_accept_rate={}, seed={})",
                       x.num_samples, x.num_warmup, datamunge::format_detail::preview(x.initial_step_sizes),
                       x.target_accept_rate, x.seed);
  }
};

/// @brief Prints a bayes::GibbsResult as a draw summary plus per-coordinate diagnostics.
template <>
struct std::formatter<datamunge::bayes::GibbsResult> : datamunge::format_detail::text_formatter<datamunge::bayes::GibbsResult> {
  /// @brief A draw summary plus per-coordinate diagnostics.
  /// @param x The value to print.
  /// @return "GibbsResult(draws=n, dim=d, mean=[...], sd=[...], accept_rates=[...], final_step_sizes=[...])".
  static std::string text(const datamunge::bayes::GibbsResult& x) {
    return std::format("GibbsResult({}, accept_rates={}, final_step_sizes={})",
                       datamunge::format_detail::bayes_draws_text(x.samples),
                       datamunge::format_detail::preview(x.accept_rates),
                       datamunge::format_detail::preview(x.final_step_sizes));
  }
};

/// @brief Prints bayes::ImportanceSamplingOptions as one line of its fields.
template <>
struct std::formatter<datamunge::bayes::ImportanceSamplingOptions> : datamunge::format_detail::text_formatter<datamunge::bayes::ImportanceSamplingOptions> {
  /// @brief One line of its fields.
  /// @param x The value to print.
  /// @return "ImportanceSamplingOptions(num_samples=n, seed=s)".
  static std::string text(const datamunge::bayes::ImportanceSamplingOptions& x) {
    return std::format("ImportanceSamplingOptions(num_samples={}, seed={})", x.num_samples, x.seed);
  }
};

/// @brief Prints a bayes::ImportanceSamplingResult as its weighted mean, ESS and log-evidence.
template <>
struct std::formatter<datamunge::bayes::ImportanceSamplingResult> : datamunge::format_detail::text_formatter<datamunge::bayes::ImportanceSamplingResult> {
  /// @brief Its weighted mean, ESS and log-evidence.
  /// @param x The value to print.
  /// @return "ImportanceSamplingResult(draws=n, dim=d, weighted_mean=[...], effective_sample_size=e, log_evidence=l)".
  static std::string text(const datamunge::bayes::ImportanceSamplingResult& x) {
    const std::size_t n = x.samples.size();
    const std::size_t d = n == 0 ? 0 : x.samples.front().size();
    std::vector<double> mean(d, 0.0);
    if (x.normalized_weights.size() == n)
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d && j < x.samples[i].size(); ++j) mean[j] += x.normalized_weights[i] * x.samples[i][j];
    return std::format("ImportanceSamplingResult(draws={}, dim={}, weighted_mean={}, effective_sample_size={}, log_evidence={})",
                       n, d, datamunge::format_detail::preview(mean), x.effective_sample_size, x.log_evidence);
  }
};

/// @brief Prints a bayes::INLALikelihoodPoint as "INLALikelihoodPoint(log_density=..., score=..., curvature=...)".
template <>
struct std::formatter<datamunge::bayes::INLALikelihoodPoint> : datamunge::format_detail::text_formatter<datamunge::bayes::INLALikelihoodPoint> {
  /// @brief "INLALikelihoodPoint(log_density=..., score=..., curvature=...)".
  /// @param x The value to print.
  /// @return "INLALikelihoodPoint(log_density=l, score=s, curvature=c)".
  static std::string text(const datamunge::bayes::INLALikelihoodPoint& x) {
    return std::format("INLALikelihoodPoint(log_density={}, score={}, curvature={})", x.log_density, x.score, x.curvature);
  }
};

/// @brief Prints bayes::INLAOptions as one line of its fields.
template <>
struct std::formatter<datamunge::bayes::INLAOptions> : datamunge::format_detail::text_formatter<datamunge::bayes::INLAOptions> {
  /// @brief One line of its fields.
  /// @param x The value to print.
  /// @return "INLAOptions(strategy=..., newton_max_iterations=..., ...)".
  static std::string text(const datamunge::bayes::INLAOptions& x) {
    return std::format("INLAOptions(strategy={}, newton_max_iterations={}, newton_tolerance={}, grid_points_per_dim={}, "
                       "grid_span={}, mode_population_size={}, mode_max_generations={}, seed={})",
                       x.strategy, x.newton_max_iterations, x.newton_tolerance, x.grid_points_per_dim, x.grid_span,
                       x.mode_population_size, x.mode_max_generations, x.seed);
  }
};

/// @brief Prints a bayes::INLAResult as a short multi-line posterior summary.
template <>
struct std::formatter<datamunge::bayes::INLAResult> : datamunge::format_detail::text_formatter<datamunge::bayes::INLAResult> {
  /// @brief A short multi-line posterior summary.
  /// @param x The value to print.
  /// @return A multi-line "INLAResult" summary of the latent-field and hyperparameter posteriors.
  static std::string text(const datamunge::bayes::INLAResult& x) {
    return std::format(
        "INLAResult(theta_dim={}, log_marginal_likelihood={}, grid_points={})\n"
        "  latent_mean: {}\n"
        "  latent_sd:   {}\n"
        "  theta_mean:  {}\n"
        "  theta_sd:    {}",
        x.theta_dim, x.log_marginal_likelihood, x.theta_grid_weights.size(),
        datamunge::format_detail::preview(x.latent_mean), datamunge::format_detail::preview(x.latent_sd),
        datamunge::format_detail::preview(x.theta_mean), datamunge::format_detail::preview(x.theta_sd));
  }
};

/// @brief Prints bayes::MAPOptions as one line of its fields.
template <>
struct std::formatter<datamunge::bayes::MAPOptions> : datamunge::format_detail::text_formatter<datamunge::bayes::MAPOptions> {
  /// @brief One line of its fields.
  /// @param x The value to print.
  /// @return "MAPOptions(max_iterations=n, tolerance=t, history_size=h)".
  static std::string text(const datamunge::bayes::MAPOptions& x) {
    return std::format("MAPOptions(max_iterations={}, tolerance={}, history_size={})", x.max_iterations, x.tolerance,
                       x.history_size);
  }
};

/// @brief Prints a bayes::HMC as "HMC(...)" with its configured options.
template <>
struct std::formatter<datamunge::bayes::HMC> : datamunge::format_detail::text_formatter<datamunge::bayes::HMC> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "HMC(field=..., ...)".
  static std::string text(const datamunge::bayes::HMC& x) {
    return datamunge::format_detail::optim_with_options("HMC", x.options());
  }
};

/// @brief Prints a bayes::NUTS as "NUTS(...)" with its configured options.
template <>
struct std::formatter<datamunge::bayes::NUTS> : datamunge::format_detail::text_formatter<datamunge::bayes::NUTS> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "NUTS(field=..., ...)".
  static std::string text(const datamunge::bayes::NUTS& x) {
    return datamunge::format_detail::optim_with_options("NUTS", x.options());
  }
};

/// @brief Prints a bayes::RandomWalkMetropolis as "RandomWalkMetropolis(...)" with its configured options.
template <>
struct std::formatter<datamunge::bayes::RandomWalkMetropolis> : datamunge::format_detail::text_formatter<datamunge::bayes::RandomWalkMetropolis> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "RandomWalkMetropolis(field=..., ...)".
  static std::string text(const datamunge::bayes::RandomWalkMetropolis& x) {
    return datamunge::format_detail::optim_with_options("RandomWalkMetropolis", x.options());
  }
};

/// @brief Prints a bayes::GibbsSampler as "GibbsSampler(...)" with its configured options.
template <>
struct std::formatter<datamunge::bayes::GibbsSampler> : datamunge::format_detail::text_formatter<datamunge::bayes::GibbsSampler> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "GibbsSampler(field=..., ...)".
  static std::string text(const datamunge::bayes::GibbsSampler& x) {
    return datamunge::format_detail::optim_with_options("GibbsSampler", x.options());
  }
};

/// @brief Prints a bayes::ImportanceSampling as "ImportanceSampling(...)" with its configured options.
template <>
struct std::formatter<datamunge::bayes::ImportanceSampling> : datamunge::format_detail::text_formatter<datamunge::bayes::ImportanceSampling> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "ImportanceSampling(field=..., ...)".
  static std::string text(const datamunge::bayes::ImportanceSampling& x) {
    return datamunge::format_detail::optim_with_options("ImportanceSampling", x.options());
  }
};

/// @brief Prints a bayes::INLA as "INLA(...)" with its configured options.
template <>
struct std::formatter<datamunge::bayes::INLA> : datamunge::format_detail::text_formatter<datamunge::bayes::INLA> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "INLA(field=..., ...)".
  static std::string text(const datamunge::bayes::INLA& x) {
    return datamunge::format_detail::optim_with_options("INLA", x.options());
  }
};

/// @brief Prints a bayes::MAP as "MAP(...)" with its configured options.
template <>
struct std::formatter<datamunge::bayes::MAP> : datamunge::format_detail::text_formatter<datamunge::bayes::MAP> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "MAP(field=..., ...)".
  static std::string text(const datamunge::bayes::MAP& x) {
    return datamunge::format_detail::optim_with_options("MAP", x.options());
  }
};
