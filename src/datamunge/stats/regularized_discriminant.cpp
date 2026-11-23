/// @file regularized_discriminant.cpp
/// @brief Stable Gaussian classification with pooled and spherical covariance regularization.

#include <datamunge/stats/regularized_discriminant.hpp>

#include <datamunge/linalg/cholesky.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Working scatter and validation for discriminant fitting.
namespace {

/// @brief Extended-precision covariance, scatter, or class centering offsets.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Wide-precision class parameters and scratch space.
using Vector = std::vector<long double>;
/// @brief Original row indices grouped by label.
using Groups = std::map<std::string,std::vector<std::size_t>>;

/// @brief Tests whether every entry is finite.
bool finite(const std::vector<double>& values) {
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}

/// @brief Checks the storage required by a dense matrix.
void check_storage(std::size_t rows, std::size_t cols) {
    if (cols > 0 && rows > Vector{}.max_size() / cols)
        throw std::length_error("discriminant analysis: dense workspace exceeds storage capacity");
}

/// @brief Converts a fitted parameter to finite double precision.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("discriminant analysis: fitted parameter exceeds the finite reporting range");
    return static_cast<double>(value);
}

/// @brief Validates training data and builds class membership in deterministic label order.
Groups groups(const linalg::DenseMatrix<double>& predictors, const std::vector<std::string>& labels,
    const RegularizedDiscriminantOptions& options) {
    const auto n = predictors.rows(), p = predictors.cols();
    /// @brief Checks one regularization fraction.
    const auto valid_fraction = [](double value) { return value >= 0 && value <= 1; };
    if (n < 2 || p == 0 || labels.size() != n || !finite(predictors.data()) ||
        !valid_fraction(options.pooling) || !valid_fraction(options.shrinkage) ||
        (options.normalization != DiscriminantCovariance::Sample &&
         options.normalization != DiscriminantCovariance::MaximumLikelihood))
        throw std::invalid_argument("discriminant analysis: invalid data or regularization");
    Groups result;
    for (std::size_t i = 0; i < n; ++i) {
        if (labels[i].empty())
            throw std::invalid_argument("discriminant analysis: class labels must be nonempty");
        result[labels[i]].push_back(i);
    }
    if (result.size() < 2)
        throw std::invalid_argument("discriminant analysis: at least two observed classes are required");
    if ((!options.priors.empty() && options.priors.size() != result.size()) ||
        !std::ranges::all_of(options.priors, [](double value) { return std::isfinite(value) && value > 0; }))
        throw std::invalid_argument("discriminant analysis: supply one positive prior weight per class");
    return result;
}

} // namespace

DiscriminantPrediction RegularizedDiscriminantFit::predict(const linalg::DenseMatrix<double>& predictors) const {
    const auto n = predictors.rows(), p = predictors.cols(), count = classes_.size();
    if (count == 0 || factors_.size() != count || p != means_.cols() || !finite(predictors.data()))
        throw std::invalid_argument("discriminant analysis: incompatible prediction rows");
    check_storage(n,count);
    DiscriminantPrediction result{{},linalg::DenseMatrix<double>(n,count)};
    result.class_label.reserve(n);
    Vector standardized(p), scores(count);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < count; ++k) {
            const auto& factor = factors_[k];
            long double squared_distance = 0;
            for (std::size_t j = 0; j < p; ++j) {
                const auto difference = (static_cast<long double>(predictors(i,j))-means_(k,j))-mean_offsets_(k,j);
                auto value = difference/factor.deviations[j];
                for (std::size_t a = 0; a < j; ++a)
                    value -= factor.lower(j,a)*standardized[a];
                standardized[j] = value/factor.lower(j,j);
                squared_distance += standardized[j]*standardized[j];
            }
            scores[k] = factor.log_weight - .5L*squared_distance;
        }
        const auto winner = std::ranges::max_element(scores);
        result.class_label.push_back(classes_[winner-scores.begin()]);
        const auto maximum = *winner;
        long double total = 0;
        for (auto& score : scores) {
            score = std::exp(score-maximum);
            total += score;
        }
        for (std::size_t k = 0; k < count; ++k)
            result.posterior(i,k) = static_cast<double>(scores[k]/total);
    }
    return result;
}

RegularizedDiscriminantFit fit_regularized_discriminant(const linalg::DenseMatrix<double>& predictors,
    const std::vector<std::string>& labels, const RegularizedDiscriminantOptions& options) {
    const auto membership = groups(predictors,labels,options);
    const auto n = predictors.rows(), p = predictors.cols(), classes = membership.size();
    check_storage(p,p);
    check_storage(classes,p*p);
    const auto class_penalty = options.normalization == DiscriminantCovariance::Sample ? 1u : 0u;
    const long double pooling = options.pooling, shrinkage = options.shrinkage;

    RegularizedDiscriminantFit result;
    result.classes_.reserve(classes);
    result.priors_.resize(classes);
    result.means_ = linalg::DenseMatrix<double>(classes,p);
    result.mean_offsets_ = Matrix(classes,p);
    result.covariances_.reserve(classes);
    result.factors_.reserve(classes);
    std::vector<Matrix> within;
    within.reserve(classes);
    Matrix pooled(p,p);
    Vector prior_weights(classes), divisor(classes);
    std::size_t group_index = 0;
    for (const auto& [label,rows] : membership) {
        const auto k = group_index++;
        result.classes_.push_back(label);
        prior_weights[k] = options.priors.empty() ? static_cast<long double>(rows.size()) : options.priors[k];
        divisor[k] = std::lerp(static_cast<long double>(rows.size()-class_penalty),
            static_cast<long double>(n-classes*class_penalty),pooling);
        if (divisor[k] <= 0)
            throw std::domain_error("discriminant analysis: a class has no covariance degrees of freedom");
        Vector origins(p), offsets(p);
        for (std::size_t j = 0; j < p; ++j) {
            origins[j] = predictors(rows.front(),j);
            for (const auto i : rows)
                offsets[j] += predictors(i,j)-origins[j];
            offsets[j] /= rows.size();
            result.means_(k,j) = output(origins[j]+offsets[j]);
            result.mean_offsets_(k,j) = offsets[j] - (result.means_(k,j)-origins[j]);
        }
        auto& scatter = within.emplace_back(p,p);
        Vector differences(p);
        for (const auto i : rows) {
            for (std::size_t j = 0; j < p; ++j)
                differences[j] = (predictors(i,j)-origins[j])-offsets[j];
            for (std::size_t j = 0; j < p; ++j)
                for (std::size_t a = 0; a <= j; ++a)
                    scatter(j,a) += differences[j]*differences[a];
        }
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t a = 0; a <= j; ++a)
                pooled(j,a) += scatter(j,a);
    }

    const auto total_prior = std::accumulate(prior_weights.begin(),prior_weights.end(),0.0L);
    for (std::size_t k = 0; k < classes; ++k) {
        const auto prior = prior_weights[k]/total_prior;
        result.priors_[k] = static_cast<double>(prior);
        auto covariance = std::move(within[k]);
        long double average = 0;
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t a = 0; a <= j; ++a)
                covariance(j,a) = std::lerp(covariance(j,a),pooled(j,a),pooling)/divisor[k];
            average += covariance(j,j)/p;
        }
        Vector deviations(p);
        auto& reported = result.covariances_.emplace_back(p,p);
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t a = 0; a < j; ++a)
                covariance(j,a) *= 1-shrinkage;
            covariance(j,j) = std::lerp(covariance(j,j),average,shrinkage);
            if (!(covariance(j,j) > 0))
                throw std::domain_error("discriminant analysis: a class covariance has a constant direction");
            deviations[j] = std::sqrt(covariance(j,j));
            for (std::size_t a = 0; a <= j; ++a)
                reported(j,a) = reported(a,j) = output(covariance(j,a));
            if (reported(j,j) == 0)
                throw std::overflow_error("discriminant analysis: covariance variance underflows double precision");
        }
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t a = 0; a < j; ++a)
                covariance(j,a) = covariance(a,j) = (covariance(j,a)/deviations[j])/deviations[a];
            covariance(j,j) = 1;
        }
        auto factor = linalg::cholesky(covariance);
        if (!factor.ok)
            throw std::domain_error("discriminant analysis: class covariance is singular");
        const auto pivot_tolerance = 8*std::sqrt(std::numeric_limits<double>::epsilon());
        long double log_weight = std::log(prior);
        for (std::size_t j = 0; j < p; ++j) {
            if (!(factor.L(j,j) > pivot_tolerance))
                throw std::domain_error("discriminant analysis: class covariance is numerically singular");
            log_weight -= std::log(deviations[j])+std::log(factor.L(j,j));
        }
        result.factors_.push_back({std::move(factor.L),std::move(deviations),log_weight});
    }
    return result;
}

RegularizedDiscriminantFit fit_quadratic_discriminant(const linalg::DenseMatrix<double>& predictors,
    const std::vector<std::string>& labels, const std::vector<double>& priors) {
    return fit_regularized_discriminant(predictors,labels,{.priors=priors});
}

} // namespace datamunge::stats
