#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::stats {

struct ConformalClassifierOptions {
    /// @brief Target miscoverage level: the guarantee is that the true class lies in the prediction
    ///        set with probability at least 1 - alpha.
    double alpha{0.1};
    /// @brief Nonconformity score: "lac" (1 - predicted probability of the class -- the Least
    ///        Ambiguous set-valued Classifier / softmax-score method) or "aps" (Adaptive Prediction
    ///        Sets: the cumulative probability of all classes at least as likely as the class),
    ///        which produces more adaptive, input-dependent set sizes.
    std::string score{"lac"};
};

/// @brief Split (inductive) conformal prediction for classification. Wrapping /any/ base classifier
///        that outputs class probabilities, it turns point predictions into prediction /sets/ that
///        carry a distribution-free, finite-sample coverage guarantee: under only the assumption
///        that the calibration and test data are exchangeable, the true label is contained in the
///        set with probability at least 1 - alpha, regardless of the base model or the data
///        distribution. It works by scoring each calibration example's true class, taking a
///        finite-sample-corrected empirical quantile of those nonconformity scores as a threshold,
///        and, for a test point, returning every class whose score falls below the threshold.
class SplitConformalClassifier {
public:
    /// @param calibration_probabilities rows are calibration examples, columns are class
    ///        probabilities (each row should sum to ~1), column order matching @p classes.
    /// @param calibration_labels the true class index (into @p classes) of each calibration row.
    SplitConformalClassifier(const linalg::DenseMatrix<double>& calibration_probabilities,
                             const std::vector<std::size_t>& calibration_labels, std::vector<std::string> classes,
                             ConformalClassifierOptions options = {});

    [[nodiscard]] double            alpha() const { return options_.alpha; }
    [[nodiscard]] const std::string& score() const { return options_.score; }
    /// @brief The calibrated nonconformity threshold (scores at or below it are admitted).
    [[nodiscard]] double            quantile() const { return quantile_; }
    [[nodiscard]] std::size_t       n_calibration() const { return n_cal_; }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }

    /// @brief Per-class set membership (1/0), aligned with classes(), for one test probability row.
    [[nodiscard]] std::vector<char> membership(const std::vector<double>& probabilities) const;
    /// @brief The prediction set as class names.
    [[nodiscard]] std::vector<std::string> predict_set(const std::vector<double>& probabilities) const;

    struct Evaluation {
        double                   coverage{0.0};          // fraction of test points whose set contains the truth
        double                   average_set_size{0.0};
        double                   empty_rate{0.0};        // fraction of empty prediction sets
        std::vector<std::size_t> set_sizes;              // per test point
    };
    /// @brief Empirical coverage and set-size statistics on a labeled test set.
    [[nodiscard]] Evaluation evaluate(const linalg::DenseMatrix<double>& test_probabilities,
                                      const std::vector<std::size_t>& test_labels) const;

private:
    [[nodiscard]] double nonconformity(const std::vector<double>& probabilities, std::size_t k) const;

    ConformalClassifierOptions options_;
    std::vector<std::string>   classes_;
    std::size_t                n_cal_{0};
    double                     quantile_{0.0};
    bool                       include_all_{false};  // true when the finite-sample rank exceeds n
};

struct ConformalRegressorOptions {
    double alpha{0.1};
};

/// @brief Split conformal prediction for regression: wraps any point regressor to produce
///        prediction /intervals/ with the same distribution-free guarantee. The nonconformity score
///        is the absolute residual on a calibration set; its finite-sample (1 - alpha) quantile
///        q-hat becomes a fixed interval half-width, so the interval [y_hat - q, y_hat + q] contains
///        the true response with probability at least 1 - alpha under exchangeability.
class SplitConformalRegressor {
public:
    SplitConformalRegressor(const std::vector<double>& calibration_predictions,
                            const std::vector<double>& calibration_truth, ConformalRegressorOptions options = {});

    [[nodiscard]] double      alpha() const { return options_.alpha; }
    /// @brief The interval half-width (the calibrated absolute-residual quantile).
    [[nodiscard]] double      quantile() const { return quantile_; }
    [[nodiscard]] std::size_t n_calibration() const { return n_cal_; }

    /// @brief [prediction - q, prediction + q].
    [[nodiscard]] std::pair<double, double> predict_interval(double prediction) const;

    struct Evaluation {
        double coverage{0.0};
        double average_width{0.0};
    };
    [[nodiscard]] Evaluation evaluate(const std::vector<double>& test_predictions,
                                      const std::vector<double>& test_truth) const;

private:
    ConformalRegressorOptions options_;
    std::size_t               n_cal_{0};
    double                    quantile_{0.0};
    bool                      infinite_{false};
};

} // namespace datamunge::stats
