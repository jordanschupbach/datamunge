#pragma once

/// \file logitboost.hpp
/// \brief LogitBoost: additive logistic regression by boosting regression stumps
///        (Friedman, Hastie & Tibshirani 2000).
///
/// AdaBoost can be understood as fitting an additive model under an *exponential*
/// loss. *LogitBoost* replaces that with the *binomial log-likelihood* -- the loss of
/// logistic regression -- and fits the additive model by Newton stepping (iteratively
/// reweighted least squares). This makes it a stagewise way to build a logistic
/// regression whose linear predictor is a sum of weak regressors, and it is more
/// robust to noisy labels and outliers than AdaBoost because the logistic loss grows
/// only linearly (not exponentially) in the margin.
///
/// For labels \f$y\in\{0,1\}\f$ and current probability estimates
/// \f$p_i = 1/(1+e^{-2F(x_i)})\f$, each round forms the Newton *working response* and
/// *weights*
/// \f[
///   z_i = \frac{y_i - p_i}{p_i(1-p_i)},\qquad w_i = p_i(1-p_i),
/// \f]
/// fits a regression stump \f$f\f$ to \f$z\f$ by *weighted* least squares, and updates
/// \f$F \leftarrow F + \tfrac12 f\f$. The final classifier is \f$\operatorname{sign}(F)\f$.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// A regression stump: threshold one feature, output a constant on each side.
struct RegressionStump {
    std::size_t feature   = 0;
    double      threshold = 0.0;
    double      left_value = 0.0;   ///< Output for x[feature] <= threshold.
    double      right_value = 0.0;  ///< Output for x[feature] >  threshold.

    double predict(const std::vector<double>& x) const {
        return (x[feature] <= threshold) ? left_value : right_value;
    }
};

/// A trained LogitBoost model: an additive ensemble of half-weighted regression stumps.
struct LogitBoostModel {
    std::vector<RegressionStump> stumps;        ///< Each contributes 0.5 * stump to F.
    std::vector<double>          log_likelihood; ///< Binomial log-likelihood after each round.
};

namespace detail {

/// Weighted-least-squares regression stump fit to responses \p z with weights \p w.
inline RegressionStump fit_regression_stump(const std::vector<std::vector<double>>& X,
                                            const std::vector<double>& z, const std::vector<double>& w) {
    const std::size_t n = X.size(), dim = X.front().size();
    RegressionStump   best;
    double            best_sse = std::numeric_limits<double>::infinity();

    for (std::size_t f = 0; f < dim; ++f) {
        std::vector<double> vals;
        vals.reserve(n);
        for (std::size_t i = 0; i < n; ++i) vals.push_back(X[i][f]);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        std::vector<double> thresholds;
        thresholds.push_back(vals.front() - 1.0);
        for (std::size_t k = 0; k + 1 < vals.size(); ++k) thresholds.push_back(0.5 * (vals[k] + vals[k + 1]));

        for (double thr : thresholds) {
            double wl = 0, wr = 0, wzl = 0, wzr = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (X[i][f] <= thr) { wl += w[i]; wzl += w[i] * z[i]; }
                else { wr += w[i]; wzr += w[i] * z[i]; }
            }
            const double ml = wl > 0 ? wzl / wl : 0.0;
            const double mr = wr > 0 ? wzr / wr : 0.0;
            double       sse = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double pred = (X[i][f] <= thr) ? ml : mr;
                sse += w[i] * (z[i] - pred) * (z[i] - pred);
            }
            if (sse < best_sse) {
                best_sse = sse;
                best     = {f, thr, ml, mr};
            }
        }
    }
    return best;
}

}  // namespace detail

/// \brief Train a LogitBoost binary classifier.
///
/// \param features   Design matrix.
/// \param labels     Labels in \f$\{0,1\}\f$.
/// \param num_rounds Boosting rounds.
/// \param max_response Clip on the Newton working response (guards \f$p(1-p)\to 0\f$).
inline LogitBoostModel logitboost_train(const std::vector<std::vector<double>>& features,
                                        const std::vector<int>& labels, std::size_t num_rounds = 50,
                                        double max_response = 4.0) {
    if (features.size() != labels.size()) throw std::invalid_argument("logitboost: size mismatch");
    LogitBoostModel   model;
    const std::size_t n = features.size();
    if (n == 0) return model;

    std::vector<double> F(n, 0.0), p(n, 0.5);
    for (std::size_t t = 0; t < num_rounds; ++t) {
        std::vector<double> z(n), w(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double pw = std::max(p[i] * (1.0 - p[i]), 1e-12);
            double       zi = (static_cast<double>(labels[i]) - p[i]) / pw;
            zi              = std::max(-max_response, std::min(max_response, zi));  // clip
            z[i]            = zi;
            w[i]            = pw;
        }
        RegressionStump stump = detail::fit_regression_stump(features, z, w);
        model.stumps.push_back(stump);

        double loglik = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            F[i] += 0.5 * stump.predict(features[i]);
            p[i] = 1.0 / (1.0 + std::exp(-2.0 * F[i]));
            const double pi = std::max(std::min(p[i], 1.0 - 1e-12), 1e-12);
            loglik += labels[i] ? std::log(pi) : std::log(1.0 - pi);
        }
        model.log_likelihood.push_back(loglik);
    }
    return model;
}

/// Probability that the label is 1, under a trained LogitBoost model.
inline double logitboost_probability(const LogitBoostModel& model, const std::vector<double>& x) {
    double F = 0.0;
    for (const auto& s : model.stumps) F += 0.5 * s.predict(x);
    return 1.0 / (1.0 + std::exp(-2.0 * F));
}

/// Predict class \f$0\f$ or \f$1\f$ with a trained LogitBoost model.
inline int logitboost_predict(const LogitBoostModel& model, const std::vector<double>& x) {
    return logitboost_probability(model, x) >= 0.5 ? 1 : 0;
}

}  // namespace datamunge::algorithms
