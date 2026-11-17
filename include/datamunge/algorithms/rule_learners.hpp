#pragma once

/// \file rule_learners.hpp
/// \brief Two baseline rule classifiers: ZeroR (majority-class) and OneR
///        (one-attribute rule) -- the sanity-check floors every classifier is
///        measured against.
///
/// Before deploying a complex model it is essential to know what *trivial* rules
/// already achieve on a data set. Two classic baselines do this:
///   - *ZeroR* (Zero-attribute Rule) ignores all features and always predicts the
///     *majority class*. Its accuracy is the class prior of the most common label --
///     a reference under unweighted zero-one loss, not a test of all useful learning.
///   - *OneR* (One-attribute Rule; Holte 1993) builds the best single-attribute rule:
///     for each attribute it forms one rule mapping each attribute value to the most
///     frequent class among rows with that value, scores the rule's training error,
///     and keeps the single attribute with the lowest error. Despite its simplicity
///     OneR is often within a few percent of far more elaborate learners, which is
///     exactly Holte's famous point: "very simple classification rules perform well
///     on most commonly used datasets".
///
/// Both operate on categorical (integer-coded) attributes and labels.

#include <datamunge/algorithms/one_r.hpp>
#include <cstddef>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

/// @brief Value model for a constant majority-class classifier on integer-coded classes.
///
/// Integer labels represent nominal categories, including negative integers.
/// A default model is empty; zero_r_train() fills its training counts.
/// @par Example
/// @code
/// const auto model = dm::algorithms::zero_r_train(std::vector<int>{0, 1, 0});
/// const int prediction = model.prediction; // 0
/// @endcode
/// @par References
/// - Weka ZeroR (nominal-class variant), https://weka.sourceforge.io/doc.stable/weka/classifiers/rules/ZeroR.html
/// @see zero_r_train(), zero_r_predict()
struct ZeroRModel {
    int         prediction = 0;    ///< The majority class.
    std::size_t majority_count = 0;///< How many training rows had that class.
    std::size_t total = 0;         ///< Training rows.

    /// @brief Fraction of training labels equal to prediction; this is not held-out accuracy.
    /// @return majority_count/total, or zero for an empty model.
    /// @par Example
    /// @code
    /// const auto model = dm::algorithms::zero_r_train(std::vector<int>{0, 1, 0});
    /// const double accuracy = model.accuracy(); // 2.0/3.0
    /// @endcode
    /// @see zero_r_train()
    double      accuracy() const { return total ? static_cast<double>(majority_count) / total : 0.0; }
};

/// @brief Train the optimal constant decision under empirical unweighted zero-one loss.
///
/// Tallies labels in a sorted map: O(n log K) time and O(K) storage. The
/// smallest label wins a count tie. Empty input returns an empty default model.
/// @param labels Nominal integer-coded classes, borrowed and unchanged.
/// @return Owned model with training size, selected class and majority count.
/// @par Example
/// @code
/// const auto model = dm::algorithms::zero_r_train(std::vector<int>{2, 1, 2, 1});
/// const int prediction = model.prediction; // 1 (tie)
/// @endcode
/// @par References
/// - Weka ZeroR (nominal-class variant), https://weka.sourceforge.io/doc.stable/weka/classifiers/rules/ZeroR.html
/// @see ZeroRModel, zero_r_predict()
inline ZeroRModel zero_r_train(const std::vector<int>& labels) {
    ZeroRModel               m;
    std::map<int, std::size_t> counts;
    for (int y : labels) ++counts[y];
    for (const auto& [cls, c] : counts)
        if (c > m.majority_count) {
            m.majority_count = c;
            m.prediction     = cls;
        }
    m.total = labels.size();
    return m;
}

/// @brief Return the constant class decision, irrespective of supplied features.
/// @param m Model value; an empty default model predicts its default label zero.
/// @param features Ignored; defaults to an empty vector.
/// @return m.prediction, without mutation or allocation, in O(1) time.
/// @par Example
/// @code
/// const auto model = dm::algorithms::zero_r_train(std::vector<int>{1, 1, 0});
/// const int prediction = dm::algorithms::zero_r_predict(model, {4, 9}); // 1
/// @endcode
/// @par References
/// - Weka ZeroR, https://weka.sourceforge.io/doc.stable/weka/classifiers/rules/ZeroR.html
/// @see zero_r_train()
inline int zero_r_predict(const ZeroRModel& m, const std::vector<int>& features = {}) {
    (void)features;
    return m.prediction;
}

}  // namespace datamunge::algorithms
