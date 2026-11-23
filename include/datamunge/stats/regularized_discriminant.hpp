#pragma once

/// @file regularized_discriminant.hpp
/// @brief Quadratic Gaussian classification with class covariance regularization.

#include <datamunge/linalg/dense_matrix.hpp>

#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Normalization used for within-class covariance estimates.
/// @par Example
/// @code
/// const auto normalization = dm::stats::DiscriminantCovariance::Sample;
/// std::println("{}", normalization == dm::stats::DiscriminantCovariance::Sample); // true
/// @endcode
enum class DiscriminantCovariance {
    Sample,            ///< Divide scatter by the available within-class degrees of freedom.
    MaximumLikelihood  ///< Divide scatter by its observation count.
};

/// @brief Pooling, isotropic shrinkage, priors and covariance normalization.
/// @par Example
/// @code
/// const dm::stats::RegularizedDiscriminantOptions options{.pooling=.4,.shrinkage=.2};
/// std::println("{:.1f}", options.pooling); // 0.4
/// @endcode
struct RegularizedDiscriminantOptions {
    double pooling{0};    ///< Mix class and pooled scatter and their divisors by this fraction in [0,1].
    double shrinkage{0};  ///< Mix the resulting covariance with its average variance times the identity.
    DiscriminantCovariance normalization{DiscriminantCovariance::Sample}; ///< Scatter divisor convention.
    std::vector<double> priors; ///< Positive class weights in sorted label order. Empty uses sample proportions.
};

/// @brief Class labels and posterior class probabilities for new rows.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6,1,{0,1,2,4,6,8});
/// const auto fit = dm::stats::fit_quadratic_discriminant(x,{"A","A","A","B","B","B"});
/// const auto result = fit.predict(dm::linalg::DenseMatrix<double>(1,1,{1}));
/// std::println("{}", result.class_label.front()); // A
/// @endcode
struct DiscriminantPrediction {
    std::vector<std::string> class_label;   ///< Maximum-posterior label per row, with ties resolved by label order.
    linalg::DenseMatrix<double> posterior; ///< Posterior probabilities in the fitted class order.
};

/// @brief Owned Gaussian discriminant model with a factored covariance per class.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6,1,{0,1,2,4,6,8});
/// const auto fit = dm::stats::fit_quadratic_discriminant(x,{"A","A","A","B","B","B"});
/// std::println("{}", fit.classes().size()); // 2
/// @endcode
class RegularizedDiscriminantFit {
public:
    /// @brief Returns the sorted fitted labels.
    /// @return Class labels in probability and parameter order.
    /// @see fit_regularized_discriminant()
    [[nodiscard]] const std::vector<std::string>& classes() const noexcept { return classes_; }

    /// @brief Returns fitted prior probabilities in label order.
    /// @return Normalized class probabilities.
    /// @see fit_regularized_discriminant()
    [[nodiscard]] const std::vector<double>& priors() const noexcept { return priors_; }

    /// @brief Returns fitted class means.
    /// @return Class-by-predictor matrix in label and input-column order.
    /// @see fit_regularized_discriminant()
    [[nodiscard]] const linalg::DenseMatrix<double>& class_means() const noexcept { return means_; }

    /// @brief Returns fitted covariances after pooling and shrinkage.
    /// @return One predictor-by-predictor covariance matrix per class, in label order.
    /// @see fit_regularized_discriminant()
    [[nodiscard]] const std::vector<linalg::DenseMatrix<double>>& class_covariances() const noexcept { return covariances_; }

    /// @brief Predicts class labels and posterior probabilities using the fitted Gaussian models.
    /// @param predictors Finite query rows with columns in training order.
    /// @return Labels and probabilities in query row order.
    /// @throws std::invalid_argument Wrong feature count or nonfinite predictors.
    /// @throws std::length_error The prediction matrix exceeds storage capacity.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(6,1,{0,1,2,4,6,8});
    /// const auto fit = dm::stats::fit_quadratic_discriminant(x,{"A","A","A","B","B","B"});
    /// std::println("{}", fit.predict(dm::linalg::DenseMatrix<double>(1,1,{6})).class_label.front()); // B
    /// @endcode
    [[nodiscard]] DiscriminantPrediction predict(const linalg::DenseMatrix<double>& predictors) const;

private:
    /// @brief Storage initialized by the fitted-model factory.
    RegularizedDiscriminantFit() = default;

    /// @brief Working factor and log normalizer for one class.
    struct ClassFactor {
        linalg::DenseMatrix<long double> lower; ///< Cholesky factor of the class correlation matrix.
        std::vector<long double> deviations;    ///< Marginal standard deviations.
        long double log_weight;                 ///< Log prior minus half the log covariance determinant.
    };

    std::vector<std::string> classes_;              ///< Observed labels in sort order.
    std::vector<double> priors_;                   ///< Normalized prior probabilities.
    linalg::DenseMatrix<double> means_;             ///< Class means rounded to double precision.
    linalg::DenseMatrix<long double> mean_offsets_; ///< Extra precision for deviations from the means.
    std::vector<linalg::DenseMatrix<double>> covariances_; ///< Fitted covariances in original units.
    std::vector<ClassFactor> factors_;             ///< Cached factors used for prediction.

    friend RegularizedDiscriminantFit fit_regularized_discriminant(const linalg::DenseMatrix<double>&,
        const std::vector<std::string>&, const RegularizedDiscriminantOptions&);
};

/// @brief Fits Gaussian discriminant analysis with optional pooling and covariance shrinkage.
///
/// Blends each class's scatter with the pooled within-class scatter and
/// blends their divisors by the same amount. Then moves the covariance
/// toward a spherical covariance with the same average marginal variance.
/// With maximum-likelihood normalization this follows Friedman's
/// two-parameter estimator. The sample convention uses residual degrees of freedom.
/// @param predictors Finite training rows with one variable per column.
/// @param labels Nonempty class labels, one per row, with at least two distinct classes.
/// @param options Pooling, shrinkage, covariance convention and optional class prior weights.
/// @return An owned model that can report covariances and score new observations.
/// @throws std::invalid_argument Invalid dimensions, labels, priors or controls.
/// @throws std::domain_error A requested class covariance is numerically singular.
/// @throws std::length_error Working matrices exceed storage capacity.
/// @throws std::overflow_error A covariance entry exceeds the reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6,1,{0,1,2,4,6,8});
/// const auto fit = dm::stats::fit_regularized_discriminant(x,{"A","A","A","B","B","B"},{.pooling=1});
/// std::println("{:.1f}", fit.class_covariances()[0](0,0)); // 2.5
/// @endcode
/// @par References
/// - Friedman (1989), https://doi.org/10.1080/01621459.1989.10478752
[[nodiscard]] RegularizedDiscriminantFit fit_regularized_discriminant(const linalg::DenseMatrix<double>& predictors,
    const std::vector<std::string>& labels, const RegularizedDiscriminantOptions& options = {});

/// @brief Fits quadratic discriminant analysis with a separate sample covariance per class.
/// @param predictors Finite training predictors, one observation per row.
/// @param labels Nonempty class labels, one per row.
/// @param priors Positive class weights in sorted label order. Empty uses sample proportions.
/// @return The fitted model, as in fit_regularized_discriminant().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6,1,{0,1,2,4,6,8});
/// const auto fit = dm::stats::fit_quadratic_discriminant(x,{"A","A","A","B","B","B"});
/// std::println("{:.1f}", fit.class_covariances()[1](0,0)); // 4.0
/// @endcode
[[nodiscard]] RegularizedDiscriminantFit fit_quadratic_discriminant(const linalg::DenseMatrix<double>& predictors,
    const std::vector<std::string>& labels, const std::vector<double>& priors = {});

} // namespace datamunge::stats
