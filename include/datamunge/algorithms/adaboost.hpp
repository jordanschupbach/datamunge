#pragma once

/// \file adaboost.hpp
/// \brief AdaBoost.M1 -- adaptive boosting of decision stumps into a strong
///        binary classifier (Freund & Schapire 1997).
///
/// *Boosting* builds a strong classifier as a weighted vote of many *weak*
/// classifiers, each only required to do slightly better than chance. AdaBoost
/// trains the weak learners sequentially on a *reweighted* view of the data: after
/// each round the examples the current ensemble gets wrong are up-weighted, so the
/// next weak learner is forced to concentrate on the hard cases. The weak learners
/// here are *decision stumps* -- one-level decision trees that threshold a single
/// feature.
///
/// With weak-learner error \f$\varepsilon_t<\tfrac12\f$ at round \f$t\f$, its vote is
/// \f$\alpha_t=\tfrac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}>0\f$, and example
/// weights update as \f$D_i \leftarrow D_i\,e^{-\alpha_t y_i h_t(x_i)}\f$ (then
/// renormalized): correctly classified points shrink, misclassified points grow.
/// The final rule is the sign of the weighted vote,
/// \f$H(x)=\operatorname{sign}\!\big(\sum_t \alpha_t h_t(x)\big)\f$. A central result
/// is that the training error falls at least as fast as
/// \f$\prod_t 2\sqrt{\varepsilon_t(1-\varepsilon_t)}\f$ -- exponentially while each
/// learner beats chance -- and that boosting keeps improving *test* margins even
/// after training error hits zero.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// A decision stump: threshold one feature, predict \f$\pm 1\f$, with an ensemble weight.
struct DecisionStump {
    std::size_t feature   = 0;    ///< Feature index tested.
    double      threshold = 0.0;  ///< Split point.
    int         polarity  = 1;    ///< Predict \c polarity if \f$x_{\text{feature}}\le\text{threshold}\f$, else \f$-\text{polarity}\f$.
    double      alpha     = 0.0;  ///< Vote weight \f$\alpha_t\f$ in the ensemble.

    /// Raw \f$\pm 1\f$ prediction of this stump (ignoring \c alpha).
    int classify(const std::vector<double>& x) const {
        return (x[feature] <= threshold) ? polarity : -polarity;
    }
};

/// A trained AdaBoost ensemble.
struct AdaBoostModel {
    std::vector<DecisionStump> stumps;          ///< Weak learners in training order, each carrying \c alpha.
    std::vector<double>        training_error;  ///< Ensemble 0/1 training error after each round.
    std::vector<double>        weak_error;      ///< Weighted error \f$\varepsilon_t\f$ of each weak learner.
};

/// @brief Implementation helpers for adaboost_train().
namespace detail {

/// \brief Best weighted-error decision stump for the current example weights.
///
/// For each feature the examples are sorted once and the thresholds swept in increasing order
/// while accumulating the weight of each class on the "<=" side, so a feature costs
/// \f$O(n\log n)\f$ rather than \f$O(n^2)\f$. Candidate thresholds are one below the minimum,
/// the midpoints between consecutive distinct values, and the maximum; ties keep the first
/// stump found (by feature, then threshold, then polarity +1 before -1).
inline DecisionStump train_stump(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                                 const std::vector<double>& w, double& best_error) {
    const std::size_t n   = X.size();
    const std::size_t dim = X.front().size();
    DecisionStump     best;
    best_error = std::numeric_limits<double>::infinity();

    double pos_total = 0.0, neg_total = 0.0;
    for (std::size_t i = 0; i < n; ++i) (y[i] > 0 ? pos_total : neg_total) += w[i];

    std::vector<std::size_t> order(n);
    for (std::size_t j = 0; j < dim; ++j) {
        for (std::size_t i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return X[a][j] < X[b][j]; });
        double     left_pos = 0.0, left_neg = 0.0;
        const auto consider = [&](double thr) {
            // Polarity +1 predicts +1 on the "<=" side; polarity -1 the reverse.
            const double err_plus  = left_neg + (pos_total - left_pos);
            const double err_minus = left_pos + (neg_total - left_neg);
            if (err_plus < best_error) {
                best_error = err_plus;
                best       = {j, thr, 1, 0.0};
            }
            if (err_minus < best_error) {
                best_error = err_minus;
                best       = {j, thr, -1, 0.0};
            }
        };
        consider(X[order.front()][j] - 1.0);  // everything on the ">" side
        std::size_t k = 0;
        while (k < n) {
            const double v = X[order[k]][j];
            for (; k < n && X[order[k]][j] == v; ++k) (y[order[k]] > 0 ? left_pos : left_neg) += w[order[k]];
            consider(k < n ? 0.5 * (v + X[order[k]][j]) : v);
        }
    }
    return best;
}

}  // namespace detail

/// \brief Train an AdaBoost.M1 ensemble of decision stumps.
///
/// Labels must be \f$\pm 1\f$. Example weights start uniform. Each round fits the
/// minimum weighted-error stump, computes its vote \f$\alpha_t\f$, reweights the
/// examples, and renormalizes. Training stops early if a weak learner achieves zero
/// weighted error (a perfect stump; it receives a large capped vote) or fails to
/// beat chance (\f$\varepsilon_t\ge\tfrac12\f$).
///
/// \param features   Row-major design matrix.
/// \param labels     Class labels, each \f$+1\f$ or \f$-1\f$.
/// \param num_rounds Maximum number of boosting rounds.
inline AdaBoostModel adaboost_train(const std::vector<std::vector<double>>& features,
                                    const std::vector<int>& labels, std::size_t num_rounds = 50) {
    if (features.size() != labels.size()) throw std::invalid_argument("adaboost: size mismatch");
    AdaBoostModel model;
    const std::size_t n = features.size();
    if (n == 0) return model;

    std::vector<double> w(n, 1.0 / static_cast<double>(n));
    std::vector<double> score(n, 0.0);

    for (std::size_t t = 0; t < num_rounds; ++t) {
        double        eps   = 0.0;
        DecisionStump stump = detail::train_stump(features, labels, w, eps);
        if (eps >= 0.5) break;  // weak learner no better than chance

        constexpr double kEpsFloor = 1e-12;
        const double     safe_eps  = eps < kEpsFloor ? kEpsFloor : eps;
        stump.alpha                = 0.5 * std::log((1.0 - safe_eps) / safe_eps);
        model.stumps.push_back(stump);
        model.weak_error.push_back(eps);

        // Reweight: D_i *= exp(-alpha y_i h(x_i)); then normalize.
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const int h = stump.classify(features[i]);
            w[i] *= std::exp(-stump.alpha * static_cast<double>(labels[i]) * static_cast<double>(h));
            norm += w[i];
        }
        if (norm > 0.0)
            for (double& wi : w) wi /= norm;

        // Ensemble training error so far, from the running scores F_i = sum_t alpha_t h_t(x_i).
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < n; ++i) {
            score[i] += stump.alpha * static_cast<double>(stump.classify(features[i]));
            if ((score[i] >= 0.0 ? 1 : -1) != labels[i]) ++wrong;
        }
        model.training_error.push_back(static_cast<double>(wrong) / static_cast<double>(n));
        if (eps < kEpsFloor) break;  // a perfect stump already separates the (reweighted) data
    }
    return model;
}

/// Raw weighted vote \f$\sum_t \alpha_t h_t(x)\f$ (its sign is the class; its magnitude the margin).
inline double adaboost_margin(const AdaBoostModel& model, const std::vector<double>& x) {
    double s = 0.0;
    for (const auto& st : model.stumps) s += st.alpha * static_cast<double>(st.classify(x));
    return s;
}

/// Predict \f$+1\f$ or \f$-1\f$ with the AdaBoost ensemble.
inline int adaboost_predict(const AdaBoostModel& model, const std::vector<double>& x) {
    return adaboost_margin(model, x) >= 0.0 ? 1 : -1;
}

}  // namespace datamunge::algorithms
