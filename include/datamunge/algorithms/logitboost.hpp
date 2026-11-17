#pragma once

/// \file logitboost.hpp
/// \brief LogitBoost: additive logistic regression by boosting regression stumps
///        (Friedman, Hastie & Tibshirani 2000).
///
/// AdaBoost can be understood as fitting an additive model under an *exponential*
/// loss. *LogitBoost* replaces that with the *binomial log-likelihood* -- the loss of
/// logistic regression -- and fits the additive model by Newton stepping (iteratively
/// reweighted least squares). This makes it a stagewise way to build a logistic
/// regression whose linear predictor is a sum of weak regressors, while its logistic loss grows
/// linearly in a negative margin. This alone does not establish robustness or
/// probability calibration for a fitted ensemble.
///
/// The producing Newton step floors p*(1-p) at1e-12, clips its working response,
/// fits a weighted least-squares stump, and adds half its output to F. The
/// probability link is1/(1+exp(-2F)). Reported training likelihood clips p before logs.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// A regression stump: threshold one feature, output a constant on each side.
/// @par Example
/// @code
/// dm::algorithms::RegressionStump s{0,0,-2,2};
/// std::print("{}\n",s.predict({-1})); // -2
/// @endcode
struct RegressionStump {
  std::size_t feature     = 0; ///< Selected zero-based feature index.
  double      threshold   = 0.0; ///< Finite split threshold, with equality sent left.
  double      left_value  = 0.0; ///< Output for x[feature] <= threshold.
  double      right_value = 0.0; ///< Output for x[feature] >  threshold.

  /// @brief Evaluate the selected side after validating the complete query and stump.
  /// @param x Complete finite row containing the selected feature.
  /// @return Left value for x[feature]<=threshold, otherwise right value.
  /// @throws std::invalid_argument For invalid feature index, nonfinite row or stump fields.
  /// @code
  /// dm::algorithms::RegressionStump s{0,0,-2,2};
  /// std::print("{}\n",s.predict({1})); // 2
  /// @endcode
  double predict(const std::vector<double>& x) const {
    if (feature >= x.size() || !std::isfinite(threshold) || !std::isfinite(left_value) ||
        !std::isfinite(right_value))
      throw std::invalid_argument("logitboost: invalid stump or query dimension");
    for (double v : x)
      if (!std::isfinite(v))
        throw std::invalid_argument("logitboost: nonfinite query");
    return (x[feature] <= threshold) ? left_value : right_value;
  }
};

/// A trained LogitBoost model: an additive ensemble of half-weighted regression stumps.
/// @par Example
/// @code
/// auto m=dm::algorithms::logitboost_train({{-1},{1}},{0,1},1);
/// std::print("{}\n",m.stumps.size()); // 1
/// @endcode
struct LogitBoostModel {
  std::vector<RegressionStump> stumps; ///< Each contributes 0.5 * stump to F.
  std::vector<double>
      log_likelihood; ///< Clipped-probability binomial training diagnostic after each round.
};

namespace detail {

/// Weighted-least-squares regression stump fit to responses \p z with weights \p w.
inline RegressionStump fit_regression_stump(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>&              z,
    const std::vector<double>&              w) {
  const std::size_t n = X.size(), dim = X.front().size();
  RegressionStump   best;
  double            best_sse = std::numeric_limits<double>::infinity();

  for (std::size_t f = 0; f < dim; ++f) {
    std::vector<double> vals;
    vals.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
      vals.push_back(X[i][f]);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    std::vector<double> thresholds;
    double              below = vals.front() - 1.0;
    if (!(below < vals.front()))
      below = std::nextafter(vals.front(), -std::numeric_limits<double>::infinity());
    if (std::isfinite(below))
      thresholds.push_back(below);
    else
      thresholds.push_back(vals.back());
    for (std::size_t k = 0; k + 1 < vals.size(); ++k) {
      double mid = .5 * (vals[k] + vals[k + 1]);
      if (!std::isfinite(mid))
        mid = std::midpoint(vals[k], vals[k + 1]);
      if (mid >= vals[k + 1])
        mid = vals[k];
      thresholds.push_back(mid);
    }

    for (double thr : thresholds) {
      double wl = 0, wr = 0, wzl = 0, wzr = 0;
      for (std::size_t i = 0; i < n; ++i) {
        if (X[i][f] <= thr) {
          wl += w[i];
          wzl += w[i] * z[i];
        } else {
          wr += w[i];
          wzr += w[i] * z[i];
        }
      }
      const double ml  = wl > 0 ? wzl / wl : 0.0;
      const double mr  = wr > 0 ? wzr / wr : 0.0;
      double       sse = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double pred = (X[i][f] <= thr) ? ml : mr;
        sse += w[i] * (z[i] - pred) * (z[i] - pred);
      }
      if (!std::isfinite(sse) || !std::isfinite(ml) || !std::isfinite(mr))
        throw std::overflow_error("logitboost: nonfinite weighted fit");
      if (sse < best_sse) {
        best_sse = sse;
        best     = {f, thr, ml, mr};
      }
    }
  }
  return best;
}

} // namespace detail

/// \brief Train a LogitBoost binary classifier.
///
/// \param features   Design matrix.
/// \param labels     Labels in \f$\{0,1\}\f$.
/// \param num_rounds Boosting rounds.
/// \param max_response Positive finite clip on the Newton working response.
/// @return Actual ensemble and clipped-probability training diagnostics.
/// @throws std::invalid_argument For invalid complete finite rectangular inputs or nonbinary labels.
/// @throws std::overflow_error For nonfinite weighted fit arithmetic or accumulated scores.
/// @note Each feature scans every candidate against every observation: O(T*d*n*n),
/// in addition to sorting. Finite ordinary decision order is retained; extreme thresholds
/// use finite midpoint/endpoint fallbacks. Ties keep the earliest candidate.
/// @par Example
/// @code
/// auto m=dm::algorithms::logitboost_train({{-1},{1}},{0,1},1);
/// std::print("{}\n",m.log_likelihood.size()); // 1
/// @endcode
LogitBoostModel logitboost_train(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             num_rounds   = 50,
    double                                  max_response = 4.0);

/// @brief Actual Newton inputs and committed probabilities from one producing round.
/// @par Example
/// @code
/// auto t=dm::algorithms::logitboost_trace({{-1},{1}},{0,1},1);
/// std::print("{}\n",t.rounds[0].responses[0]); // -2
/// @endcode
struct LogitBoostRound {
  std::vector<double> responses;      ///< Actual clipped working responses.
  std::vector<double> weights;        ///< Actual floored Newton weights.
  std::vector<double> probabilities;  ///< Probabilities after the committed update.
  RegressionStump     stump;          ///< Actual selected weighted least squares stump.
  double              log_likelihood; ///< Clipped probability training diagnostic.
};
/// @brief Owned actual producing rounds and returned ensemble.
/// @par Example
/// @code
/// auto t=dm::algorithms::logitboost_trace({{-1},{1}},{0,1},1);
/// std::print("{}\n",t.rounds.size()); // 1
/// @endcode
struct LogitBoostTrace {
  LogitBoostModel              model;  ///< Actual fitted model.
  std::vector<LogitBoostRound> rounds; ///< Actual complete rounds, without replay.
};
/// @brief Fit and capture actual rounds; quota counts one round and three n-vectors per round.
/// @param features Complete finite rectangular matrix with positive dimension unless empty.
/// @param labels Binary labels, one per row.
/// @param num_rounds Number of actual Newton updates.
/// @param max_response Positive finite clipping bound.
/// @param max_entries Logical capture quota; zero disables the optional quota.
/// @return Owned fitted ensemble and actual Newton states.
/// @throws std::invalid_argument For invalid complete inputs.
/// @throws std::length_error If the complete known capture exceeds its quota.
/// @par Example
/// @code
/// auto t=dm::algorithms::logitboost_trace({{-1},{1}},{0,1},1,.5);
/// std::print("{}\n",t.rounds[0].responses[1]); // 0.5
/// @endcode
LogitBoostTrace logitboost_trace(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             num_rounds   = 50,
    double                                  max_response = 4.0,
    std::size_t                             max_entries  = 1000000);

/// @brief Probability that the label is1 under the additive model; no calibration guarantee.
/// @param model Ensemble with finite valid stump fields.
/// @param x Complete finite query containing every selected feature.
/// @return Logistic link of the sum of half-stump outputs; empty ensemble returns0.5.
/// @throws std::invalid_argument For invalid stump/query data.
/// @throws std::overflow_error If the accumulated score is not finite.
/// @par Example
/// @code
/// auto m=dm::algorithms::logitboost_train({{-1},{1}},{0,1},0);
/// std::print("{}\n",dm::algorithms::logitboost_probability(m,{0})); // 0.5
/// @endcode
inline double logitboost_probability(const LogitBoostModel& model, const std::vector<double>& x) {
  double F = 0.0;
  for (const auto& s : model.stumps) {
    F += 0.5 * s.predict(x);
    if (!std::isfinite(F))
      throw std::overflow_error("logitboost: nonfinite score");
  }
  return 1.0 / (1.0 + std::exp(-2.0 * F));
}

/// @param model Ensemble with finite valid stump fields.
/// @param x Complete finite query containing every selected feature.
/// @return Binary class, assigning probability0.5 to class1.
/// @throws std::invalid_argument For invalid stump/query data.
/// @throws std::overflow_error If the accumulated score is not finite.
/// Predict class \f$0\f$ or \f$1\f$ with a trained LogitBoost model.
/// @par Example
/// @code
/// auto m=dm::algorithms::logitboost_train({{-1},{1}},{0,1},1);
/// std::print("{}\n",dm::algorithms::logitboost_predict(m,{1})); // 1
/// @endcode
inline int logitboost_predict(const LogitBoostModel& model, const std::vector<double>& x) {
  return logitboost_probability(model, x) >= 0.5 ? 1 : 0;
}

} // namespace datamunge::algorithms
