/// @file nested_cross_validation.cpp
/// @brief Nested model selection with isolated predictor and response subsets.

#include <datamunge/stats/nested_cross_validation.hpp>

#include <datamunge/nn/validation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Row gathering and finite loss calculations.
namespace {

/// @brief Copies selected predictor rows in the supplied index order.
linalg::DenseMatrix<double> gather(const linalg::DenseMatrix<double>& matrix,
    const std::vector<std::size_t>& indices) {
    linalg::DenseMatrix<double> subset(indices.size(),matrix.cols());
    for (std::size_t i = 0; i < indices.size(); ++i)
        for (std::size_t j = 0; j < matrix.cols(); ++j)
            subset(i,j) = matrix(indices[i],j);
    return subset;
}

/// @brief Copies the response values corresponding to the selected fitting or scoring rows.
std::vector<double> gather(const std::vector<double>& values, const std::vector<std::size_t>& indices) {
    std::vector<double> subset;
    subset.reserve(indices.size());
    for (const auto i : indices)
        subset.push_back(values[i]);
    return subset;
}

/// @brief Converts a loss to the finite reporting range.
double finite_loss(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::domain_error("nested cross-validation: loss is outside finite range");
    return static_cast<double>(value);
}

/// @brief Adds observation losses divided by the eventual number of scoring rows.
long double weighted_loss(const std::vector<double>& observed, const std::vector<double>& predicted,
    const NestedLoss& loss, std::size_t denominator) {
    if (predicted.size() != observed.size())
        throw std::invalid_argument("nested cross-validation: prediction count must match the queried rows");
    long double sum = 0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        if (!std::isfinite(predicted[i]))
            throw std::domain_error("nested cross-validation: nonfinite prediction");
        const long double difference = static_cast<long double>(observed[i])-predicted[i];
        sum += finite_loss(loss ? loss(observed[i],predicted[i]) : difference*difference) /
            static_cast<long double>(denominator);
    }
    return sum;
}

/// @brief Reuses one set of inner partitions for every candidate and aggregates row-level losses.
std::vector<double> tune(const linalg::DenseMatrix<double>& x, const std::vector<double>& y,
    const std::vector<NestedCandidate>& candidates, const std::vector<nn::ml::Fold>& folds,
    const NestedLoss& loss) {
    std::vector<long double> score(candidates.size());
    for (const auto& fold : folds) {
        const auto x_fit = gather(x,fold.training), x_validate = gather(x,fold.validation);
        const auto y_fit = gather(y,fold.training), y_validate = gather(y,fold.validation);
        for (std::size_t c = 0; c < candidates.size(); ++c) {
            const auto predictions = candidates[c].predict(x_fit,y_fit,x_validate);
            score[c] += weighted_loss(y_validate,predictions,loss,y.size());
        }
    }
    std::vector<double> result(score.size());
    std::ranges::transform(score,result.begin(),finite_loss);
    return result;
}

} // namespace

NestedCVResult nested_cross_validate(const linalg::DenseMatrix<double>& predictors,
    const std::vector<double>& response, const std::vector<NestedCandidate>& candidates,
    const NestedCVOptions& options, const NestedLoss& loss) {
    const auto n = predictors.rows();
    if (predictors.cols() == 0 || response.size() != n || options.inner_folds < 2 ||
        options.outer_folds < 2 || static_cast<std::size_t>(options.outer_folds) > n ||
        candidates.empty() || !std::ranges::all_of(candidates,[](const auto& c) { return bool(c.predict); }) ||
        !std::ranges::all_of(predictors.data(),[](double x) { return std::isfinite(x); }) ||
        !std::ranges::all_of(response,[](double y) { return std::isfinite(y); }))
        throw std::invalid_argument("nested cross-validation: invalid data, candidates or fold counts");
    const auto fold_count = static_cast<std::size_t>(options.outer_folds);
    const auto min_training = n - (n/fold_count + (n%fold_count != 0));
    if (static_cast<std::size_t>(options.inner_folds) > min_training)
        throw std::invalid_argument("nested cross-validation: inner folds exceed the smallest outer training size");

    auto folds = nn::ml::KFold(n,options.outer_folds,options.seed,options.shuffle);
    std::mt19937_64 inner_seed(options.seed);
    NestedCVResult result;
    result.predictions.resize(n);
    long double overall = 0;
    for (auto& fold : folds) {
        const auto train = gather(predictors,fold.training);
        const auto train_response = gather(response,fold.training);
        const auto inner = nn::ml::KFold(train.rows(),options.inner_folds,inner_seed(),options.shuffle);
        auto score = tune(train,train_response,candidates,inner,loss);
        const auto best = static_cast<std::size_t>(std::ranges::min_element(score)-score.begin());
        const auto& chosen = candidates[best];
        const auto predicted = chosen.predict(train,train_response,gather(predictors,fold.validation));
        const auto test_response = gather(response,fold.validation);
        const auto mean = weighted_loss(test_response,predicted,loss,predicted.size());
        for (std::size_t i = 0; i < predicted.size(); ++i)
            result.predictions[fold.validation[i]] = predicted[i];
        overall += mean*(static_cast<long double>(predicted.size())/n);
        result.folds.push_back({std::move(fold.validation),std::move(score),best,chosen.name,finite_loss(mean)});
    }
    result.mean_loss = finite_loss(overall);
    return result;
}

} // namespace datamunge::stats
