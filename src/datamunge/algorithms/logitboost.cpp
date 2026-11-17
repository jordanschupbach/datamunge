/// @file logitboost.cpp
/// @brief Shared checked Newton producer for ordinary fits and actual round capture.
#include <datamunge/algorithms/logitboost.hpp>
namespace datamunge::algorithms {
/// @brief Private shared producing kernel.
namespace {
/// @brief Fit once and optionally own each actual committed Newton round.
/// @param features Complete finite rectangular matrix.
/// @param labels Complete binary labels.
/// @param num_rounds Requested actual updates.
/// @param max_response Positive finite working-response cap.
/// @param capture Optional actual round output.
/// @param max_entries Optional complete capture quota.
/// @return Actual fitted model.
LogitBoostModel fit(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             num_rounds,
    double                                  max_response,
    std::vector<LogitBoostRound>*           capture,
    std::size_t                             max_entries) {
  if (features.size() != labels.size())
    throw std::invalid_argument("logitboost: size mismatch");
  if (!std::isfinite(max_response) || max_response <= 0)
    throw std::invalid_argument("logitboost: response cap must be positive finite");
  const auto dimension = features.empty() ? 0 : features.front().size();
  for (std::size_t i = 0; i < features.size(); ++i) {
    if (features[i].size() != dimension || !dimension || (labels[i] != 0 && labels[i] != 1))
      throw std::invalid_argument("logitboost: invalid matrix or labels");
    for (double value : features[i])
      if (!std::isfinite(value))
        throw std::invalid_argument("logitboost: nonfinite feature");
  }
  if (features.size() > (std::numeric_limits<std::size_t>::max()-1)/3)
    throw std::length_error("logitboost: capture geometry overflow");
  if (capture && !features.empty() && max_entries && num_rounds > max_entries / (1 + 3 * features.size()))
    throw std::length_error("logitboost: trace quota");
  LogitBoostModel   model;
  const std::size_t n = features.size();
  if (n == 0)
    return model;

  std::vector<double> F(n, 0.0), p(n, 0.5);
  for (std::size_t t = 0; t < num_rounds; ++t) {
    std::vector<double> z(n), w(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double pw = std::max(p[i] * (1.0 - p[i]), 1e-12);
      double       zi = (static_cast<double>(labels[i]) - p[i]) / pw;
      zi              = std::max(-max_response, std::min(max_response, zi)); // clip
      z[i]            = zi;
      w[i]            = pw;
    }
    RegressionStump stump = detail::fit_regression_stump(features, z, w);
    model.stumps.push_back(stump);

    double loglik = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      F[i] += 0.5 * stump.predict(features[i]);
      if (!std::isfinite(F[i]))
        throw std::overflow_error("logitboost: nonfinite score");
      p[i]            = 1.0 / (1.0 + std::exp(-2.0 * F[i]));
      const double pi = std::max(std::min(p[i], 1.0 - 1e-12), 1e-12);
      loglik += labels[i] ? std::log(pi) : std::log(1.0 - pi);
    }
    model.log_likelihood.push_back(loglik);
    if (capture)
      capture->push_back({z, w, p, stump, loglik});
  }
  return model;
}
} // namespace
LogitBoostModel logitboost_train(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             num_rounds,
    double                                  max_response) {
  return fit(features, labels, num_rounds, max_response, nullptr, 0);
}
LogitBoostTrace logitboost_trace(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             num_rounds,
    double                                  max_response,
    std::size_t                             max_entries) {
  LogitBoostTrace t;
  t.model = fit(features, labels, num_rounds, max_response, &t.rounds, max_entries);
  return t;
}
} // namespace datamunge::algorithms
