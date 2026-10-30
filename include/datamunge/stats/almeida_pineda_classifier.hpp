#pragma once

/// @file almeida_pineda_classifier.hpp
/// @brief A recurrent-network classifier trained by Almeida-Pineda backpropagation, fit from a
///        DataFrame and an R-style formula.

#include <datamunge/algorithms/almeida_pineda.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Options for AlmeidaPinedaClassifier.
struct AlmeidaPinedaClassifierOptions {
    std::size_t hidden{5};          ///< Hidden units.
    std::size_t epochs{300};        ///< Full-batch gradient steps.
    double learning_rate{0.1};      ///< Step size on the mean per-row error.
    double momentum{0.9};           ///< Heavy-ball momentum.
    double init_sd{0.1};            ///< Standard deviation of the initial weights.
    double target{0.8};             ///< Output target: +target for the true class, -target otherwise.
    std::uint64_t seed{1};          ///< Weight initialisation seed.
    algorithms::AlmeidaPinedaSettleOptions settle{500, 1e-9, 0.5};  ///< Forward and adjoint relaxation.
};

/// @brief Multiclass classifier whose prediction is the equilibrium of a fully recurrent tanh
///        network, trained with Almeida-Pineda gradients.
///
/// Unit 0 is a bias clamped to 1; the next units are clamped to the standardised predictors;
/// then come the hidden units and one output unit per class. Every free unit connects to every
/// unit except itself, so hidden and output units feed back into each other. A row's outputs are
/// the output units' values at the fixed point, and the prediction is the class with the largest
/// output. Training minimises the mean of \f$\tfrac12\sum_k(x_k-t_k)^2\f$ with targets
/// \f$\pm\f$`target`, by gradient descent with momentum. Rows with a missing value are dropped.
///
/// @par Example
/// @code
/// const auto [train, test] = dm::stats::train_test_split(dm::datasets::iris(), 0.3, 1, "Species");
/// const dm::stats::AlmeidaPinedaClassifier model(train, "Species ~ .");
/// std::print("test error {:.3f}\n", model.error(test));
/// @endcode
class AlmeidaPinedaClassifier {
 public:
    /// @brief Fits the network.
    /// @param data Training rows.
    /// @param formula Response (a string column with at least two classes) and numeric predictors.
    /// @param options Network size, training and relaxation settings.
    /// @throws std::invalid_argument if the response has fewer than two classes or epochs is 0.
    AlmeidaPinedaClassifier(const dstruct::DataFrame& data, const std::string& formula,
                            AlmeidaPinedaClassifierOptions options = {});
    /// @brief Same, in R's `(formula, data)` argument order.
    /// @param formula Model formula.
    /// @param data Training rows.
    /// @param options Network size, training and relaxation settings.
    AlmeidaPinedaClassifier(const Formula& formula, const dstruct::DataFrame& data,
                            AlmeidaPinedaClassifierOptions options = {})
        : AlmeidaPinedaClassifier(data, formula.text(), std::move(options)) {}

    /// @brief The classes, sorted; output unit k stands for classes()[k].
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    /// @brief The predictor columns, in input-unit order.
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    /// @brief Number of training rows used.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief The trained network.
    [[nodiscard]] const algorithms::AlmeidaPinedaNetwork& network() const { return net_; }
    /// @brief The options used.
    [[nodiscard]] const AlmeidaPinedaClassifierOptions& options() const { return options_; }

    /// @brief One row per epoch: `epoch`, `loss` (mean \f$\tfrac12\sum_k e_k^2\f$ per row),
    ///        `train_error`, `mean_iterations` and `max_iterations` of the forward relaxation, and
    ///        `unconverged` (rows whose forward or adjoint relaxation hit the iteration limit).
    [[nodiscard]] dstruct::DataFrame history() const;

    /// @brief The clamped input vector of each usable row of @p data: 1 for the bias, then the
    ///        predictors standardised with the training means and standard deviations.
    [[nodiscard]] std::vector<std::vector<double>> inputs(const dstruct::DataFrame& data) const;
    /// @brief The \f$\pm\f$target vector of each usable row of @p data.
    [[nodiscard]] std::vector<std::vector<double>> targets(const dstruct::DataFrame& data) const;
    /// @brief Equilibrium output of each class (one column per class), plus `iterations`.
    [[nodiscard]] dstruct::DataFrame outputs(const dstruct::DataFrame& data) const;
    /// @brief Predicted class of each usable row.
    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& data) const;
    /// @brief Misclassification rate on @p data (which must contain the response).
    [[nodiscard]] double error(const dstruct::DataFrame& data) const;
    /// @brief Counts by true class (`actual`, one row per class) and predicted class (one column
    ///        per class).
    [[nodiscard]] dstruct::DataFrame confusion(const dstruct::DataFrame& data) const;

 private:
    /// @brief Unstandardised predictor rows of @p data (and the class labels when @p response).
    std::vector<std::vector<double>> raw_rows(const dstruct::DataFrame& data, bool response,
                                              std::vector<std::string>* labels) const;

    Formula formula_;                                ///< The model formula.
    DesignInfo design_;                              ///< Resolved columns, for new data.
    AlmeidaPinedaClassifierOptions options_;         ///< Fitting options.
    std::vector<std::string> classes_;               ///< Sorted classes.
    std::vector<std::string> predictor_names_;       ///< Predictor columns.
    std::vector<double> mean_;                       ///< Training mean of each predictor.
    std::vector<double> sd_;                         ///< Training standard deviation of each predictor.
    std::size_t observations_{0};                    ///< Training rows used.
    algorithms::AlmeidaPinedaNetwork net_;           ///< The trained network.
    std::vector<double> epoch_;                      ///< history(): epoch numbers.
    std::vector<double> loss_;                       ///< history(): mean loss.
    std::vector<double> train_error_;                ///< history(): training error rate.
    std::vector<double> mean_iterations_;            ///< history(): mean forward sweeps.
    std::vector<double> max_iterations_;             ///< history(): largest forward sweeps.
    std::vector<double> unconverged_;                ///< history(): rows that hit the sweep limit.
};

} // namespace datamunge::stats
