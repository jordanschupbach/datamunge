/// @file aipw.cpp
/// @brief Cross-fitted augmented inverse probability weighting.
#include <datamunge/causal/aipw.hpp>
#include <datamunge/causal/matching.hpp>
#include <datamunge/linalg/least_squares.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/glm.hpp>
#include "detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Double-valued covariate matrix.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Copies the selected rows of a matrix.
Matrix take_rows(const Matrix& x, const std::vector<std::size_t>& rows) {
    Matrix out(rows.size(), x.cols());
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (std::size_t j = 0; j < x.cols(); ++j)
            out(r, j) = x(rows[r], j);
    return out;
}

/// @brief Adds covariate columns named covariate1..p to a data frame.
void add_covariates(dstruct::DataFrame& frame, const Matrix& x) {
    for (std::size_t j = 0; j < x.cols(); ++j) {
        std::vector<double> column(x.rows());
        for (std::size_t i = 0; i < x.rows(); ++i)
            column[i] = x(i, j);
        frame.add_column("covariate" + std::to_string(j + 1), column);
    }
}

/// @brief Calls a learner and checks the number and finiteness of its predictions.
template<class Learner, class Train>
std::vector<double> predict_with(const Learner& learner, const Matrix& train_x, const Train& train_y,
                                 const Matrix& test_x) {
    auto p = learner(train_x, train_y, test_x);
    if (p.size() != test_x.rows() || !std::ranges::all_of(p, [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("cross_fitted_aipw: a learner returned the wrong number of predictions or non-finite values");
    return p;
}

/// @brief Random fold assignment, balanced within each treatment group.
std::vector<std::size_t> stratified_folds(const std::vector<int>& a, std::size_t folds, std::uint64_t seed) {
    random::SplitMix64 rng(seed);
    std::vector<std::size_t> ids(a.size());
    std::size_t offset = 0;
    for (int arm = 0; arm < 2; ++arm) {
        std::vector<std::size_t> rows;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] == arm)
                rows.push_back(i);
        for (std::size_t k = rows.size(); k > 1; --k) // Fisher-Yates
            std::swap(rows[k - 1], rows[static_cast<std::size_t>(rng.uniform01() * static_cast<double>(k)) % k]);
        for (std::size_t r = 0; r < rows.size(); ++r)
            ids[rows[r]] = (offset + r) % folds; // continue the cycle so fold sizes stay balanced overall
        offset += rows.size();
    }
    return ids;
}
} // namespace

std::vector<double> logistic_propensity_learner(const linalg::DenseMatrix<double>& train_x,
                                                const std::vector<int>& train_treatment,
                                                const linalg::DenseMatrix<double>& test_x) {
    if (train_x.cols() != test_x.cols())
        throw std::invalid_argument("logistic_propensity_learner: train and test covariates differ in width");
    dstruct::DataFrame train, test;
    train.add_column("treated", std::vector<double>(train_treatment.begin(), train_treatment.end()));
    add_covariates(train, train_x);
    add_covariates(test, test_x);
    std::string formula = "treated ~ 1";
    for (std::size_t j = 0; j < train_x.cols(); ++j)
        formula += " + covariate" + std::to_string(j + 1);
    const stats::GLM model(train, formula, {.family = stats::GLMFamily::Binomial});
    if (!model.converged() || model.boundary())
        throw std::runtime_error("logistic_propensity_learner: logistic fit did not converge or separates the groups");
    return model.predict(test);
}

std::vector<double> linear_outcome_learner(const linalg::DenseMatrix<double>& train_x,
                                           const std::vector<double>& train_outcome,
                                           const linalg::DenseMatrix<double>& test_x) {
    const std::size_t n = train_x.rows(), p = train_x.cols() + 1;
    if (train_outcome.size() != n || test_x.cols() + 1 != p)
        throw std::invalid_argument("linear_outcome_learner: dimension mismatch");
    Matrix design(n, p, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 1; j < p; ++j)
            design(i, j) = train_x(i, j - 1);
    const auto fit = linalg::least_squares(design, train_outcome, linalg::LeastSquaresMethod::SVD);
    if (fit.rank != p)
        throw std::runtime_error("linear_outcome_learner: rank-deficient training design");
    std::vector<double> out(test_x.rows(), fit.coefficients[0]);
    for (std::size_t i = 0; i < test_x.rows(); ++i)
        for (std::size_t j = 1; j < p; ++j)
            out[i] += fit.coefficients[j] * test_x(i, j - 1);
    return out;
}

namespace detail {
Nuisances fit_nuisances(const std::vector<double>& outcome, const std::vector<int>& treatment,
                        const linalg::DenseMatrix<double>& covariates, std::size_t folds,
                        const std::vector<std::size_t>& fold_ids, std::uint64_t seed,
                        const PropensityLearner& propensity_learner, const OutcomeLearner& outcome_learner) {
    const std::size_t n = treatment.size();
    if (outcome.size() != n || covariates.rows() != n)
        throw std::invalid_argument("causal estimator: outcome, treatment and covariates must have one entry per unit");
    finite_matrix(covariates);
    std::array<std::size_t, 2> count{};
    for (std::size_t i = 0; i < n; ++i) {
        if ((treatment[i] != 0 && treatment[i] != 1) || !std::isfinite(outcome[i]))
            throw std::invalid_argument("causal estimator: treatment must be 0 or 1 and outcomes finite");
        ++count[static_cast<std::size_t>(treatment[i])];
    }
    if (count[0] == 0 || count[1] == 0)
        throw std::invalid_argument("causal estimator: both treatment groups need units");
    Nuisances out;
    if (!fold_ids.empty()) {
        if (fold_ids.size() != n)
            throw std::invalid_argument("causal estimator: fold_ids needs one entry per unit");
        out.fold_ids = fold_ids;
        folds = *std::ranges::max_element(out.fold_ids) + 1;
    } else {
        if (folds == 0 || folds > std::min(count[0], count[1]))
            throw std::invalid_argument("causal estimator: folds must be between 1 and the smaller group size");
        out.fold_ids = folds == 1 ? std::vector<std::size_t>(n, 0) : stratified_folds(treatment, folds, seed);
    }
    const PropensityLearner propensity = propensity_learner ? propensity_learner : logistic_propensity_learner;
    const OutcomeLearner regression = outcome_learner ? outcome_learner : linear_outcome_learner;
    out.propensity.assign(n, 0.0);
    out.control_prediction.assign(n, 0.0);
    out.treated_prediction.assign(n, 0.0);
    for (std::size_t k = 0; k < folds; ++k) {
        std::vector<std::size_t> test, train, train_arm[2];
        for (std::size_t i = 0; i < n; ++i) {
            const bool in_test = out.fold_ids[i] == k;
            if (in_test)
                test.push_back(i);
            if (!in_test || folds == 1) {
                train.push_back(i);
                train_arm[treatment[i]].push_back(i);
            }
        }
        if (test.empty())
            continue;
        if (train_arm[0].empty() || train_arm[1].empty())
            throw std::invalid_argument("causal estimator: a training split lacks one treatment group");
        const Matrix test_x = take_rows(covariates, test);
        std::vector<int> train_a;
        for (const std::size_t i : train)
            train_a.push_back(treatment[i]);
        const auto e = predict_with(propensity, take_rows(covariates, train), train_a, test_x);
        std::array<std::vector<double>, 2> mu;
        for (int arm = 0; arm < 2; ++arm) {
            std::vector<double> y;
            for (const std::size_t i : train_arm[arm])
                y.push_back(outcome[i]);
            mu[arm] = predict_with(regression, take_rows(covariates, train_arm[arm]), y, test_x);
        }
        for (std::size_t r = 0; r < test.size(); ++r) {
            out.propensity[test[r]] = e[r];
            out.control_prediction[test[r]] = mu[0][r];
            out.treated_prediction[test[r]] = mu[1][r];
        }
    }
    return out;
}

std::size_t clip_propensities(std::vector<double>& propensity, double bound) {
    if (!(bound >= 0 && bound < 0.5))
        throw std::invalid_argument("causal estimator: propensity_bound must lie in [0, 0.5)");
    std::size_t clipped = 0;
    for (double& e : propensity) {
        const double c = std::clamp(e, bound, 1 - bound);
        clipped += c != e;
        e = c;
        if (!(e > 0 && e < 1))
            throw std::runtime_error("causal estimator: propensities of exactly 0 or 1; set a positive propensity_bound");
    }
    return clipped;
}
} // namespace detail

AipwResult cross_fitted_aipw(const std::vector<double>& outcome, const std::vector<int>& treatment,
                             const linalg::DenseMatrix<double>& covariates, const AipwOptions& options) {
    if (!(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("cross_fitted_aipw: confidence must lie in (0, 1)");
    auto nuisance = detail::fit_nuisances(outcome, treatment, covariates, options.folds, options.fold_ids,
                                          options.seed, options.propensity, options.outcome);
    AipwResult out;
    out.clipped = detail::clip_propensities(nuisance.propensity, options.propensity_bound);
    out.propensity = std::move(nuisance.propensity);
    out.control_prediction = std::move(nuisance.control_prediction);
    out.treated_prediction = std::move(nuisance.treated_prediction);
    out.fold_ids = std::move(nuisance.fold_ids);
    const std::size_t n = treatment.size();
    std::vector<double> psi1(n), psi0(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = treatment[i], e = out.propensity[i];
        psi1[i] = out.treated_prediction[i] + a * (outcome[i] - out.treated_prediction[i]) / e;
        psi0[i] = out.control_prediction[i] + (1 - a) * (outcome[i] - out.control_prediction[i]) / (1 - e);
    }
    const double nn = static_cast<double>(n);
    out.treated_mean = std::accumulate(psi1.begin(), psi1.end(), 0.0) / nn;
    out.control_mean = std::accumulate(psi0.begin(), psi0.end(), 0.0) / nn;
    out.ate = out.treated_mean - out.control_mean;
    out.influence.resize(n);
    double ss = 0;
    for (std::size_t i = 0; i < n; ++i) {
        out.influence[i] = psi1[i] - psi0[i] - out.ate;
        ss += out.influence[i] * out.influence[i];
    }
    out.standard_error = std::sqrt(ss / (nn - 1) / nn);
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    out.lower = out.ate - z * out.standard_error;
    out.upper = out.ate + z * out.standard_error;
    out.p_value = 2 * (1 - random::normal_cdf(std::abs(out.ate) / out.standard_error));
    return out;
}

TmleResult targeted_mle(const std::vector<double>& outcome, const std::vector<int>& treatment,
                        const linalg::DenseMatrix<double>& covariates, const TmleOptions& options) {
    if (!(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("targeted_mle: confidence must lie in (0, 1)");
    if (!(options.outcome_bound > 0 && options.outcome_bound < 0.5))
        throw std::invalid_argument("targeted_mle: outcome_bound must lie in (0, 0.5)");
    auto nuisance = detail::fit_nuisances(outcome, treatment, covariates, options.folds, options.fold_ids,
                                          options.seed, options.propensity, options.outcome);
    TmleResult out;
    out.clipped = detail::clip_propensities(nuisance.propensity, options.propensity_bound);
    const std::size_t n = treatment.size();
    const double nn = static_cast<double>(n);
    // Map the outcome to [0, 1].
    double lo = *std::ranges::min_element(outcome), hi = *std::ranges::max_element(outcome);
    if (options.outcome_range) {
        lo = options.outcome_range->first;
        hi = options.outcome_range->second;
        if (!(hi > lo) || std::ranges::any_of(outcome, [&](double y) { return y < lo || y > hi; }))
            throw std::invalid_argument("targeted_mle: outcome_range must contain every outcome");
    }
    if (!(hi > lo))
        throw std::invalid_argument("targeted_mle: outcomes must not all be equal");
    const double width = hi - lo, b = options.outcome_bound;
    const auto scale = [&](double y) { return std::clamp((y - lo) / width, b, 1 - b); };
    const auto logit = [](double p) { return std::log(p / (1 - p)); };
    const auto expit = [](double x) { return 1 / (1 + std::exp(-x)); };
    std::vector<double> ystar(n), q0(n), q1(n), h0(n), h1(n);
    for (std::size_t i = 0; i < n; ++i) {
        ystar[i] = (outcome[i] - lo) / width;
        q0[i] = logit(scale(nuisance.control_prediction[i]));
        q1[i] = logit(scale(nuisance.treated_prediction[i]));
        h1[i] = 1 / nuisance.propensity[i];
        h0[i] = 1 / (1 - nuisance.propensity[i]);
    }
    // Initial (plug-in) estimate.
    double initial = 0;
    for (std::size_t i = 0; i < n; ++i)
        initial += (expit(q1[i]) - expit(q0[i])) / nn;
    out.initial_ate = initial * width;
    // Fluctuation: quasi-binomial regression of Y* on (H0, H1) with offset logit Q(A, X). Each unit
    // carries only its own arm's covariate, so the two coefficients solve separate 1-D score equations.
    out.converged = true;
    for (int arm = 0; arm < 2; ++arm) {
        const auto& q = arm ? q1 : q0;
        const auto& h = arm ? h1 : h0;
        const auto score = [&](double eps, double& info) {
            double s = 0;
            info = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (treatment[i] != arm)
                    continue;
                const double p = expit(q[i] + eps * h[i]);
                s += h[i] * (ystar[i] - p);
                info += h[i] * h[i] * p * (1 - p);
            }
            return s;
        };
        double eps = 0;
        bool solved = false;
        for (std::size_t it = 0; it < 100; ++it) {
            double info = 0;
            const double s = score(eps, info);
            if (std::abs(s) < 1e-12 * nn) {
                solved = true;
                break;
            }
            double step = s / info, current = std::abs(s);
            for (int k = 0; k < 60; ++k) { // the score is monotone in epsilon: halve until it shrinks
                double unused = 0;
                if (std::abs(score(eps + step, unused)) < current)
                    break;
                step /= 2;
            }
            eps += step;
            out.iterations += 1;
        }
        out.converged = out.converged && solved;
        out.epsilon[static_cast<std::size_t>(arm)] = eps;
    }
    std::vector<double> q1s(n), q0s(n);
    double m1 = 0, m0 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        q1s[i] = expit(q1[i] + out.epsilon[1] * h1[i]);
        q0s[i] = expit(q0[i] + out.epsilon[0] * h0[i]);
        m1 += q1s[i] / nn;
        m0 += q0s[i] / nn;
    }
    const double psi = m1 - m0;
    out.ate = psi * width;
    out.treated_mean = lo + m1 * width;
    out.control_mean = lo + m0 * width;
    out.influence.resize(n);
    double ss = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double residual = treatment[i] ? h1[i] * (ystar[i] - q1s[i]) : -h0[i] * (ystar[i] - q0s[i]);
        out.influence[i] = (residual + q1s[i] - q0s[i] - psi) * width;
        ss += out.influence[i] * out.influence[i];
    }
    out.standard_error = std::sqrt(ss / (nn - 1) / nn);
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    out.lower = out.ate - z * out.standard_error;
    out.upper = out.ate + z * out.standard_error;
    out.p_value = 2 * (1 - random::normal_cdf(std::abs(out.ate) / out.standard_error));
    out.propensity = std::move(nuisance.propensity);
    out.fold_ids = std::move(nuisance.fold_ids);
    return out;
}

} // namespace datamunge::causal
