#pragma once

/// @file adaboost_classifier.hpp
/// @brief Formula-and-DataFrame AdaBoost.M1 classifier over decision stumps, built on
///        algorithms::adaboost_train().

#include <datamunge/algorithms/adaboost.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <string>
#include <vector>

/// @brief Statistical models, hypothesis tests and resampling.
namespace datamunge::stats {

/// @brief Options for AdaBoostClassifier.
struct AdaBoostClassifierOptions {
    std::size_t n_rounds{100};  ///< Maximum number of boosting rounds (stumps).
};

/// @brief Binary AdaBoost.M1 classifier with decision stumps, fit from a DataFrame and an R-style
///        formula whose response is a two-level string column.
///
/// The alphabetically second class is coded \f$y=+1\f$ (positive_class()) and the first
/// \f$y=-1\f$. Each round fits the stump with the smallest weighted error \f$\varepsilon_t\f$,
/// gives it the vote \f$\alpha_t=\tfrac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}\f$ and reweights
/// the rows by \f$e^{-\alpha_t y_i h_t(x_i)}\f$; the classifier is the sign of
/// \f$F(x)=\sum_t\alpha_t h_t(x)\f$. Rows with a missing response or predictor are dropped.
///
/// @par Example
/// @code
/// const auto [train, test] = dm::stats::train_test_split(dm::datasets::breast_cancer(), 0.3, 1, "diagnosis");
/// const dm::stats::AdaBoostClassifier model(train, "diagnosis ~ .", {.n_rounds = 200});
/// std::print("test error {:.3f}\n", model.error(test));
/// std::print("{}\n", model.history().head(5).to_md());
/// @endcode
///
/// @par References
/// - Y. Freund, R. E. Schapire (1997). A decision-theoretic generalization of on-line learning and
///   an application to boosting. Journal of Computer and System Sciences 55(1), 119-139.
/// - R. E. Schapire, Y. Freund, P. Bartlett, W. S. Lee (1998). Boosting the margin. Annals of
///   Statistics 26(5), 1651-1686.
class AdaBoostClassifier {
 public:
    /// @brief Fits the ensemble.
    /// @param data Training rows.
    /// @param formula Response (a string column with exactly two classes) and numeric predictors.
    /// @param options Number of rounds.
    /// @throws std::invalid_argument if the response does not have exactly two classes, or
    ///         n_rounds is 0.
    AdaBoostClassifier(const dstruct::DataFrame& data, const std::string& formula,
                       AdaBoostClassifierOptions options = {});

    /// @brief Same, in R's `(formula, data)` argument order.
    /// @param formula Model formula.
    /// @param data Training rows.
    /// @param options Number of rounds.
    AdaBoostClassifier(const Formula& formula, const dstruct::DataFrame& data, AdaBoostClassifierOptions options = {})
        : AdaBoostClassifier(data, formula.text(), options) {}

    /// @brief The two classes, sorted; classes()[1] is coded +1.
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    /// @brief The class coded \f$y=+1\f$.
    [[nodiscard]] const std::string& positive_class() const { return classes_[1]; }
    /// @brief The predictor columns, in design-matrix order.
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    /// @brief Number of training rows used.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Number of stumps actually fitted (training stops early on a perfect or useless stump).
    [[nodiscard]] std::size_t n_rounds() const { return model_.stumps.size(); }
    /// @brief The underlying stumps, votes and per-round errors.
    [[nodiscard]] const algorithms::AdaBoostModel& model() const { return model_; }

    /// @brief One row per round: `round`, `feature`, `threshold`, `polarity`, `weak_error`
    ///        (\f$\varepsilon_t\f$), `alpha`, `Z` (\f$2\sqrt{\varepsilon_t(1-\varepsilon_t)}\f$),
    ///        `bound` (\f$\prod_{s\le t}Z_s\f$, which bounds the training error) and `train_error`.
    [[nodiscard]] dstruct::DataFrame history() const;

    /// @brief The vote \f$F(x)=\sum_t\alpha_t h_t(x)\f$ for each row of @p newdata (NaN for rows with a
    ///        missing predictor); positive means positive_class().
    /// @param rounds Use only the first @p rounds stumps; 0 means all.
    [[nodiscard]] std::vector<double> decision_function(const dstruct::DataFrame& newdata, std::size_t rounds = 0) const;

    /// @brief Predicted class of each row ("" for rows with a missing predictor).
    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;

    /// @brief Misclassification rate on the labelled rows of @p data.
    /// @param rounds Use only the first @p rounds stumps; 0 means all.
    /// @throws std::invalid_argument if @p data lacks the response column.
    [[nodiscard]] double error(const dstruct::DataFrame& data, std::size_t rounds = 0) const;

    /// @brief Misclassification rate on @p data after each round 1, 2, ..., n_rounds().
    [[nodiscard]] std::vector<double> staged_error(const dstruct::DataFrame& data) const;

    /// @brief Normalized margins \f$y_i F(x_i)/\sum_t\alpha_t \in [-1, 1]\f$ of the labelled rows of
    ///        @p data; a row is misclassified exactly when its margin is negative.
    /// @param rounds Use only the first @p rounds stumps; 0 means all.
    [[nodiscard]] std::vector<double> margins(const dstruct::DataFrame& data, std::size_t rounds = 0) const;

    /// @brief Sum of the votes \f$\alpha_t\f$ of each predictor's stumps, normalized to sum to 1.
    [[nodiscard]] dstruct::DataFrame feature_importance() const;

    /// @brief The exact decision regions of a two-predictor model with @p data drawn on top.
    ///
    /// Stumps only cut along the axes, so the prediction is constant on each cell of the grid
    /// formed by the stumps' thresholds; each cell is drawn as one shaded rectangle.
    /// @param data Rows to draw as points, coloured by their true class.
    /// @throws std::logic_error if the model does not have exactly two predictors.
    [[nodiscard]] graphics::RPlot plot_decision_regions(const dstruct::DataFrame& data) const;

 private:
    /// @brief Design rows (feature vectors) and +-1 labels (0 when unlabelled) of @p data.
    void design(const dstruct::DataFrame& data, bool require_response, std::vector<std::vector<double>>& X,
                std::vector<int>& y, std::vector<std::size_t>& rows) const;
    /// @brief The vote of the first @p rounds stumps (all when 0) on one feature vector.
    [[nodiscard]] double vote(const std::vector<double>& x, std::size_t rounds) const;

    Formula                   formula_;              ///< The model formula.
    DesignInfo                design_;               ///< Resolved columns, for new data.
    AdaBoostClassifierOptions options_;              ///< Fitting options.
    std::vector<std::string>  classes_;              ///< The two classes, sorted.
    std::vector<std::string>  predictor_names_;      ///< Design-matrix column names.
    std::vector<double>       feature_min_;          ///< Training minimum of each predictor.
    std::vector<double>       feature_max_;          ///< Training maximum of each predictor.
    algorithms::AdaBoostModel model_;                ///< The fitted stumps.
    std::size_t               observations_{0};      ///< Training rows used.
};

} // namespace datamunge::stats
