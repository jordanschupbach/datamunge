/// @file gam.cpp
/// @brief Fixed-penalty additive fitting with centered univariate and tensor P-splines.

#include <datamunge/stats/gam.hpp>

#include "glm_family.hpp"

#include <datamunge/fda/pspline.hpp>
#include <datamunge/linalg/cholesky.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Private design and penalized weighted least-squares calculations.
namespace {

/// @brief Validates the predictor column count and common row length.
/// @param columns Predictor matrix in column order.
/// @param terms Expected number of predictor columns.
/// @return Common row count, which may be zero at prediction time.
/// @throws std::invalid_argument For invalid column counts or ragged columns.
std::size_t row_count(const std::vector<std::vector<double>>& columns, std::size_t terms) {
    if (columns.empty() || columns.size() != terms)
        throw std::invalid_argument("GAM: predictor column count does not match the model");
    const auto n = columns.front().size();
    for (const auto& column : columns)
        if (column.size() != n) throw std::invalid_argument("GAM: predictor columns must be rectangular");
    return n;
}

/// @brief Checks that an inverse link evaluation remains in the modeled family domain.
/// @param family Fitted family.
/// @param eta Link-scale predictor.
/// @param mu Inverse link result.
/// @return True for finite link and response values in the family domain.
bool valid_mean(GLMFamily family, double eta, double mu) {
    return std::isfinite(eta) && std::isfinite(mu) &&
           (family == GLMFamily::Gaussian || mu > 0) &&
           (family != GLMFamily::Gamma || eta > 0);
}

/// @brief Stores a candidate iterate's link and response values and objective components.
struct Iterate {
    std::vector<double> eta; ///< Candidate link-scale predictor.
    std::vector<double> mu; ///< Candidate response-scale mean.
    double deviance{0}; ///< Unpenalized deviance.
    long double objective{0}; ///< Deviance plus quadratic penalty, infinity outside the domain.
};

/// @brief Evaluates a candidate IRLS iterate and its penalized objective.
/// @param design Dense training design.
/// @param penalty Coefficient penalty matrix.
/// @param beta Candidate coefficient vector.
/// @param y Training response.
/// @param family Response model.
/// @return Candidate iterate with infinity objective when invalid.
Iterate evaluate(const linalg::DenseMatrix<double>& design, const linalg::DenseMatrix<double>& penalty,
                 const std::vector<double>& beta, const std::vector<double>& y, GLMFamily family) {
    Iterate state;
    state.eta.resize(y.size());
    state.mu.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        long double eta = 0;
        for (std::size_t j = 0; j < beta.size(); ++j) eta += static_cast<long double>(design(i, j)) * beta[j];
        state.eta[i] = static_cast<double>(eta);
        state.mu[i] = detail::glm_inverse_link(family, state.eta[i]);
        if (!valid_mean(family, state.eta[i], state.mu[i])) {
            state.objective = std::numeric_limits<long double>::infinity();
            return state;
        }
        state.deviance += detail::glm_unit_deviance(family, y[i], state.mu[i]);
    }
    state.objective = state.deviance;
    for (std::size_t j = 0; j < beta.size(); ++j)
        for (std::size_t k = 0; k < beta.size(); ++k)
            state.objective += static_cast<long double>(beta[j]) * penalty(j, k) * beta[k];
    return state;
}

/// @brief Solves an equilibrated penalized weighted working model.
/// @param design Training model matrix.
/// @param penalty Fixed coefficient penalty.
/// @param y Responses in design order.
/// @param state Previous fitted iterate.
/// @param family Valid family and link.
/// @return Unconstrained trial coefficient vector.
/// @throws std::runtime_error If the working information is nonfinite or rank deficient.
std::vector<double> working_fit(const linalg::DenseMatrix<double>& design,
                                const linalg::DenseMatrix<double>& penalty,
                                const std::vector<double>& y, const Iterate& state, GLMFamily family) {
    const auto p = design.cols();
    linalg::DenseMatrix<long double> gram(p, p);
    std::vector<long double> rhs(p);
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto g = detail::glm_link_derivative(family, state.mu[i]);
        const long double weight = 1.L / g / g / detail::glm_variance(family, state.mu[i]);
        const auto z = state.eta[i] + (static_cast<long double>(y[i]) - state.mu[i]) * g;
        if (!(weight > 0) || !std::isfinite(weight) || !std::isfinite(z))
            throw std::runtime_error("GAM: invalid IRLS working response or weight");
        for (std::size_t j = 0; j < p; ++j) {
            const auto weighted = weight * design(i, j);
            if (weighted == 0) continue;
            rhs[j] += weighted * z;
            for (std::size_t k = 0; k <= j; ++k) gram(j, k) += weighted * design(i, k);
        }
    }

    std::vector<long double> scales(p);
    for (std::size_t j = 0; j < p; ++j) {
        const auto diagonal = gram(j, j) + penalty(j, j);
        if (!(diagonal > 0) || !std::isfinite(diagonal))
            throw std::runtime_error("GAM: unidentifiable or nonfinite design");
        scales[j] = std::sqrt(diagonal);
    }
    for (std::size_t j = 0; j < p; ++j) {
        rhs[j] /= scales[j];
        for (std::size_t k = 0; k <= j; ++k) {
            const auto value = (gram(j, k) + penalty(j, k)) / scales[j] / scales[k];
            gram(j, k) = gram(k, j) = value;
        }
    }
    const auto factor = linalg::cholesky(gram);
    if (!factor.ok) throw std::runtime_error("GAM: penalized design is rank deficient");
    for (std::size_t j = 0; j < p; ++j)
        if (!(factor.L(j, j) > std::sqrt(std::numeric_limits<long double>::epsilon())))
            throw std::runtime_error("GAM: penalized design is numerically rank deficient");

    const auto scaled = factor.solve(rhs);
    std::vector<double> result(p);
    for (std::size_t j = 0; j < p; ++j) {
        result[j] = static_cast<double>(scaled[j] / scales[j]);
        if (!std::isfinite(result[j])) throw std::runtime_error("GAM: nonfinite coefficient");
    }
    return result;
}

/// @brief Checks a family and returns a finite starting intercept.
/// @param y Nonempty training responses.
/// @param family Selected GLM family.
/// @return Link of the sample mean.
/// @throws std::invalid_argument For unsupported families, invalid responses or boundary means.
double initial_intercept(const std::vector<double>& y, GLMFamily family) {
    switch (family) {
        case GLMFamily::Gaussian:
        case GLMFamily::Poisson:
        case GLMFamily::Binomial:
        case GLMFamily::Gamma: break;
        default: throw std::invalid_argument("GAM: unknown family");
    }
    long double total = 0;
    for (double value : y) {
        detail::glm_validate_response(family, value);
        total += value;
    }
    const auto mean = static_cast<double>(total / y.size());
    if ((family != GLMFamily::Gaussian && mean <= 0) ||
        (family == GLMFamily::Binomial && mean >= 1))
        throw std::invalid_argument("GAM: response mean lies on the family boundary");
    const auto eta = detail::glm_link(family, mean);
    if (!std::isfinite(eta)) throw std::invalid_argument("GAM: nonfinite initial intercept");
    return eta;
}
} // namespace

std::vector<double> AdditiveModel::term_row(const Term& term, double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("GAM: predictors must be finite");
    if (!term.spline) return {x - term.centers.front()};
    auto result = term.centers;
    for (auto& value : result) value = -value;
    const auto local = term.basis.evaluate_point(std::clamp(x, term.lower, term.upper));
    for (std::size_t i = 0; i < local.indices.size(); ++i)
        if (local.indices[i] < result.size()) result[local.indices[i]] += local.values[i];
    return result;
}

std::vector<double> AdditiveModel::tensor_row(const TensorTerm& term,
    const std::vector<const std::vector<double>*>& columns, std::size_t row) {
    std::vector<double> result(term.centers.size(), 1);
    for (std::size_t j = 0; j < term.margins.size(); ++j) {
        const auto values = term_row(term.margins[j], (*columns[j])[row]);
        for (std::size_t k = 0; k < result.size(); ++k)
            result[k] *= values[(k / term.strides[j]) % values.size()];
    }
    for (std::size_t k = 0; k < result.size(); ++k)
        result[k] -= term.centers[k];
    return result;
}

std::vector<const std::vector<double>*> AdditiveModel::tensor_columns(const TensorTerm& term,
    const std::vector<std::vector<double>>& predictors) {
    std::vector<const std::vector<double>*> result;
    result.reserve(term.predictors.size());
    for (const auto j : term.predictors)
        result.push_back(&predictors[j]);
    return result;
}

AdditiveModel::AdditiveModel(const std::vector<double>& y, const std::vector<std::vector<double>>& predictors,
                             const std::vector<std::string>& groups, double group_penalty, const GAMOptions& options)
    : family_(options.family) {
    if (y.empty() || row_count(predictors, predictors.size()) != y.size() || options.max_iter == 0 ||
        !(options.tol > 0) || !std::isfinite(options.tol) ||
        (!options.smooths.empty() && options.smooths.size() != predictors.size()))
        throw std::invalid_argument("GAM: invalid data shape, smooth configuration or solver controls");
    const double start = initial_intercept(y, family_);
    if (!groups.empty()) {
        group_levels_ = groups;
        std::ranges::sort(group_levels_);
        group_levels_.erase(std::unique(group_levels_.begin(), group_levels_.end()), group_levels_.end());
        if (group_levels_.size() < 2 || group_levels_.front().empty())
            throw std::invalid_argument("GAMM: need at least two distinct nonempty group labels");
    }

    std::size_t fixed = 1;
    const auto capacity = std::vector<double>{}.max_size();
    for (std::size_t j = 0; j < predictors.size(); ++j) {
        const auto spec = options.smooths.empty() ? GAMSmooth{} : options.smooths[j];
        if ((!std::isfinite(spec.lambda) || spec.lambda < 0) ||
            (spec.basis_size != 0 && spec.basis_size < 4))
            throw std::invalid_argument("GAM: require a finite nonnegative penalty and cubic basis size >= 4 or 0");
        const auto& column = predictors[j];
        if (!std::ranges::all_of(column, [](double value) { return std::isfinite(value); }))
            throw std::invalid_argument("GAM: nonfinite predictor");
        const auto [lo, hi] = std::ranges::minmax_element(column);
        if (*lo == *hi || !std::isfinite(*hi - *lo))
            throw std::invalid_argument("GAM: predictor range must be finite and positive");
        const bool spline = spec.basis_size != 0;
        const auto width = spline ? spec.basis_size - 1 : 1;
        if (width > std::numeric_limits<std::size_t>::max() - fixed)
            throw std::invalid_argument("GAM: design width overflows");
        const auto next = fixed + width;
        if (next > capacity / next || y.size() > capacity / next)
            throw std::invalid_argument("GAM: design or penalty exceeds storage capacity");
        terms_.push_back({spline ? fda::BSplineBasis::open_uniform(3, spec.basis_size, *lo, *hi)
                                 : fda::BSplineBasis{},
                          spline, std::vector<double>(width, 0), *lo, *hi, fixed});
        fixed = next;
    }

    for (const auto& spec : options.tensor_smooths) {
        if (spec.margins.size() < 2 || spec.margins.size() > predictors.size())
            throw std::invalid_argument("GAM: a tensor requires at least two distinct predictors");
        TensorTerm term{{}, {}, std::vector<std::size_t>(spec.margins.size(), 1), {}, fixed};
        std::size_t width = 1;
        for (const auto& margin : spec.margins) {
            if (margin.predictor >= terms_.size() || margin.basis_size < 4 ||
                !std::isfinite(margin.lambda) || margin.lambda < 0 ||
                std::ranges::find(term.predictors, margin.predictor) != term.predictors.end())
                throw std::invalid_argument("GAM: invalid tensor predictor, cubic basis size or penalty");
            const auto reduced = margin.basis_size - 1;
            if (reduced > (capacity - fixed) / width)
                throw std::invalid_argument("GAM: tensor design width exceeds storage capacity");
            width *= reduced;
            const auto next = fixed + width;
            if (next > capacity / next || y.size() > capacity / next)
                throw std::invalid_argument("GAM: tensor design or penalty exceeds storage capacity");
            const auto& axis = terms_[margin.predictor];
            term.predictors.push_back(margin.predictor);
            term.margins.push_back({fda::BSplineBasis::open_uniform(3, margin.basis_size, axis.lower, axis.upper),
                                    true, std::vector<double>(reduced), axis.lower, axis.upper, 0});
        }
        for (const auto& other : tensor_terms_)
            if (std::ranges::is_permutation(other.predictors, term.predictors))
                throw std::invalid_argument("GAM: duplicate tensor interaction");
        for (std::size_t j = spec.margins.size() - 1; j > 0; --j)
            term.strides[j - 1] = term.strides[j] * term.margins[j].centers.size();
        term.centers.resize(width);
        tensor_terms_.push_back(std::move(term));
        fixed += width;
    }

    if (group_levels_.size() > std::numeric_limits<std::size_t>::max() - fixed)
        throw std::invalid_argument("GAM: design width overflows");
    const auto p = fixed + group_levels_.size();
    if (p > capacity / p || y.size() > capacity / p)
        throw std::invalid_argument("GAM: design or penalty exceeds storage capacity");
    linalg::DenseMatrix<double> design(y.size(), p, 0);
    linalg::DenseMatrix<double> penalty(p, p, 0);
    for (std::size_t i = 0; i < y.size(); ++i) {
        design(i, 0) = 1;
        if (!groups.empty()) {
            const auto group = std::ranges::lower_bound(group_levels_, groups[i]) - group_levels_.begin();
            design(i, fixed + group) = 1;
        }
    }
    for (std::size_t j = fixed; j < p; ++j) penalty(j, j) = group_penalty;
    for (std::size_t j = 0; j < terms_.size(); ++j) {
        auto& term = terms_[j];
        const auto width = term.centers.size();
        std::vector<long double> sums(width);
        for (std::size_t i = 0; i < y.size(); ++i) {
            const auto row = term_row(term, predictors[j][i]);
            for (std::size_t k = 0; k < width; ++k) {
                design(i, term.offset + k) = row[k];
                sums[k] += row[k];
            }
        }
        for (std::size_t k = 0; k < width; ++k) {
            term.centers[k] = static_cast<double>(sums[k] / y.size());
            for (std::size_t i = 0; i < y.size(); ++i) design(i, term.offset + k) -= term.centers[k];
        }
        if (term.spline) {
            const auto lambda = options.smooths.empty() ? GAMSmooth{}.lambda : options.smooths[j].lambda;
            const auto full = fda::pspline_penalty_matrix(width + 1, 2);
            // The final coefficient is fixed at zero because a constant spline is absorbed by the intercept.
            for (std::size_t k = 0; k < width; ++k)
                for (std::size_t l = 0; l < width; ++l)
                    penalty(term.offset + k, term.offset + l) = lambda * full(k, l);
        }
    }
    for (std::size_t t = 0; t < tensor_terms_.size(); ++t) {
        auto& term = tensor_terms_[t];
        const auto columns = tensor_columns(term, predictors);
        const auto width = term.centers.size();
        std::vector<linalg::DenseMatrix<double>> differences, metrics;
        for (std::size_t j = 0; j < term.margins.size(); ++j) {
            auto& margin = term.margins[j];
            const auto reduced = margin.centers.size();
            std::vector<long double> sums(reduced);
            for (const auto x : *columns[j]) {
                const auto row = term_row(margin, x);
                for (std::size_t k = 0; k < reduced; ++k)
                    sums[k] += row[k];
            }
            for (std::size_t k = 0; k < reduced; ++k)
                margin.centers[k] = static_cast<double>(sums[k] / y.size());

            // With B the full basis and m the retained column means, the
            // centered basis is B*T for T = [I;0] - 1*m'. Second differences
            // kill constants, while the identity penalty factor becomes T'*T.
            differences.push_back(fda::pspline_penalty_matrix(reduced + 1, 2));
            auto& metric = metrics.emplace_back(reduced, reduced);
            for (std::size_t k = 0; k < reduced; ++k)
                for (std::size_t l = 0; l < reduced; ++l) {
                    const auto mk = margin.centers[k], ml = margin.centers[l];
                    metric(k, l) = (k == l ? 1. : 0.) - mk - ml + (reduced + 1) * mk * ml;
                }
        }

        std::vector<long double> sums(width);
        for (std::size_t i = 0; i < y.size(); ++i) {
            const auto row = tensor_row(term, columns, i);
            for (std::size_t j = 0; j < width; ++j) {
                design(i, term.offset + j) = row[j];
                sums[j] += row[j];
            }
        }
        for (std::size_t j = 0; j < width; ++j) {
            term.centers[j] = static_cast<double>(sums[j] / y.size());
            for (std::size_t i = 0; i < y.size(); ++i)
                design(i, term.offset + j) -= term.centers[j];
            for (std::size_t k = 0; k <= j; ++k) {
                long double entry = 0, metric_product = 1;
                // Sum the Kronecker product for every penalized margin.
                for (std::size_t axis = 0; axis < term.margins.size(); ++axis) {
                    const auto reduced = term.margins[axis].centers.size();
                    const auto a = (j / term.strides[axis]) % reduced;
                    const auto b = (k / term.strides[axis]) % reduced;
                    const auto metric = metrics[axis](a, b);
                    const long double lambda = options.tensor_smooths[t].margins[axis].lambda;
                    entry = entry * metric + metric_product * lambda * differences[axis](a, b);
                    metric_product *= metric;
                }
                penalty(term.offset + j, term.offset + k) = penalty(term.offset + k, term.offset + j) =
                    static_cast<double>(entry);
            }
        }
    }
    coefficients_.assign(p, 0);
    coefficients_.front() = start;
    auto state = evaluate(design, penalty, coefficients_, y, family_);
    if (!std::isfinite(state.objective)) throw std::runtime_error("GAM: invalid starting objective");
    for (std::size_t iter = 0; iter < options.max_iter; ++iter) {
        const auto proposed = working_fit(design, penalty, y, state, family_);
        double change = 0;
        for (std::size_t j = 0; j < p; ++j)
            change = std::max(change, std::abs(proposed[j] - coefficients_[j]) / (1 + std::abs(proposed[j])));
        auto candidate = proposed;
        auto trial = evaluate(design, penalty, candidate, y, family_);
        const auto objective_limit = state.objective + 1e-12L * (1 + std::abs(state.objective));
        for (double step = 0.5; !(trial.objective <= objective_limit); step *= 0.5) {
            if (step < 1e-12) throw std::runtime_error("GAM: could not find a decreasing IRLS update");
            for (std::size_t j = 0; j < p; ++j) candidate[j] = std::lerp(coefficients_[j], proposed[j], step);
            trial = evaluate(design, penalty, candidate, y, family_);
        }
        coefficients_ = std::move(candidate);
        state = std::move(trial);
        ++iterations_;
        if (change < options.tol) {
            converged_ = true;
            break;
        }
    }
    fitted_ = std::move(state.mu);
    deviance_ = state.deviance;
}

std::vector<double> AdditiveModel::smooth(std::size_t term, const std::vector<double>& x) const {
    if (term >= terms_.size()) throw std::invalid_argument("GAM: invalid term index");
    const auto& stored = terms_[term];
    std::vector<double> result;
    result.reserve(x.size());
    for (double value : x) {
        const auto row = term_row(stored, value);
        long double contribution = 0;
        for (std::size_t j = 0; j < row.size(); ++j)
            contribution += static_cast<long double>(row[j]) * coefficients_[stored.offset + j];
        const auto output = static_cast<double>(contribution);
        if (!std::isfinite(output)) throw std::runtime_error("GAM: term contribution overflows");
        result.push_back(output);
    }
    return result;
}

std::vector<double> AdditiveModel::tensor_contribution(const TensorTerm& term,
    const std::vector<const std::vector<double>*>& columns) const {
    std::vector<double> result(columns.front()->size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto row = tensor_row(term, columns, i);
        long double contribution = 0;
        for (std::size_t j = 0; j < row.size(); ++j)
            contribution += static_cast<long double>(row[j]) * coefficients_[term.offset + j];
        result[i] = static_cast<double>(contribution);
        if (!std::isfinite(result[i]))
            throw std::runtime_error("GAM: tensor contribution overflows");
    }
    return result;
}

std::vector<double> AdditiveModel::tensor_smooth(std::size_t tensor,
    const std::vector<std::vector<double>>& coordinates) const {
    if (tensor >= tensor_terms_.size())
        throw std::invalid_argument("GAM: invalid tensor index");
    const auto& term = tensor_terms_[tensor];
    row_count(coordinates, term.margins.size());
    std::vector<const std::vector<double>*> columns;
    columns.reserve(coordinates.size());
    for (const auto& column : coordinates)
        columns.push_back(&column);
    return tensor_contribution(term, columns);
}

std::vector<double> AdditiveModel::predict(const std::vector<std::vector<double>>& predictors,
                                          const std::vector<std::string>& groups) const {
    const auto n = row_count(predictors, terms_.size());
    if (!groups.empty() && (groups.size() != n || group_levels_.empty()))
        throw std::invalid_argument("GAM: invalid prediction group labels");
    std::vector<long double> eta(n, intercept());
    for (std::size_t j = 0; j < terms_.size(); ++j) {
        const auto component = smooth(j, predictors[j]);
        for (std::size_t i = 0; i < n; ++i) eta[i] += component[i];
    }
    for (const auto& term : tensor_terms_) {
        const auto component = tensor_contribution(term, tensor_columns(term, predictors));
        for (std::size_t i = 0; i < n; ++i) eta[i] += component[i];
    }
    const auto fixed = coefficients_.size() - group_levels_.size();
    std::vector<double> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!groups.empty()) {
            if (groups[i].empty()) throw std::invalid_argument("GAMM: empty prediction label");
            const auto group = std::ranges::lower_bound(group_levels_, groups[i]);
            if (group != group_levels_.end() && *group == groups[i])
                eta[i] += coefficients_[fixed + (group - group_levels_.begin())];
        }
        const auto linear = static_cast<double>(eta[i]);
        result[i] = detail::glm_inverse_link(family_, linear);
        if (!valid_mean(family_, linear, result[i])) throw std::runtime_error("GAM: prediction outside family domain");
    }
    return result;
}

graphics::RPlot AdditiveModel::plot_smooth(std::size_t term, std::size_t points) const {
    if (term >= terms_.size() || points < 2) throw std::invalid_argument("GAM: invalid term or plot grid");
    const auto& stored = terms_[term];
    std::vector<double> x(points);
    for (std::size_t i = 0; i < points; ++i)
        x[i] = std::lerp(stored.lower, stored.upper, static_cast<double>(i) / (points - 1));
    return graphics::RPlot::create().lines(x, smooth(term, x), "Fitted component", {36, 105, 172})
        .abline_h(0).x_label("Predictor " + std::to_string(term + 1)).y_label("Centered link-scale contribution");
}

std::vector<double> AdditiveModel::random_intercepts() const {
    return {coefficients_.end() - group_levels_.size(), coefficients_.end()};
}

AdditiveModel fit_gam(const std::vector<double>& y, const std::vector<std::vector<double>>& predictors,
                      const GAMOptions& options) {
    return AdditiveModel(y, predictors, {}, 0, options);
}

AdditiveModel fit_gamm(const std::vector<double>& y, const std::vector<std::vector<double>>& predictors,
                       const std::vector<std::string>& groups, const GAMMOptions& options) {
    const double sd = options.random_intercept_sd;
    const double dispersion = options.dispersion;
    const auto family = options.gam.family;
    if (groups.empty() || groups.size() != y.size() || !(sd > 0) || !std::isfinite(sd) ||
        !(dispersion > 0) || !std::isfinite(dispersion) ||
        ((family == GLMFamily::Binomial || family == GLMFamily::Poisson) && dispersion != 1))
        throw std::invalid_argument("GAMM: invalid group labels, random-intercept SD or dispersion");
    const auto penalty = static_cast<double>(static_cast<long double>(dispersion) / sd / sd);
    if (!(penalty > 0) || !std::isfinite(penalty))
        throw std::invalid_argument("GAMM: random-intercept penalty outside finite range");
    return AdditiveModel(y, predictors, groups, penalty, options.gam);
}

} // namespace datamunge::stats
