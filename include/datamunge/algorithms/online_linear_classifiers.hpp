#pragma once

/// \file online_linear_classifiers.hpp
/// \brief Mistake-driven online linear classifiers: the Perceptron (additive
///        weight updates) and Winnow (multiplicative weight updates).
///
/// Both algorithms learn a linear separator by streaming through labeled examples
/// and updating the weight vector only when the current weights misclassify an
/// example. They differ in *how* they update and in what regime they excel:
///
///   - The *Perceptron* (Rosenblatt 1958) uses an *additive* update
///     \f$w \leftarrow w + \eta\,y\,x\f$. On any linearly separable data set with
///     margin \f$\gamma\f$ and radius \f$R=\max_i\|x_i\|\f$, it makes at most
///     \f$(R/\gamma)^2\f$ mistakes before converging, regardless of dimension
///     (the Perceptron convergence theorem, Novikoff 1962).
///   - *Winnow* (Littlestone 1988) uses a *multiplicative* update
///     \f$w_i \leftarrow w_i\,\alpha^{\pm 1}\f$ on the active features. Its mistake
///     bound grows only *logarithmically* in the number of features, so it excels
///     when the target depends on a few relevant features among many irrelevant
///     ones (e.g. learning a \f$k\f$-literal disjunction over \f$n\f$ boolean
///     attributes in \f$O(k\log n)\f$ mistakes).
///
/// These complementary bounds -- additive/\f$L_2\f$ for the Perceptron,
/// multiplicative/\f$L_\infty\f$-to-\f$L_1\f$ for Winnow -- make the pair a classic
/// illustration of how the geometry of the update shapes sample efficiency.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// A trained Perceptron: a linear decision rule \f$\operatorname{sign}(w\cdot x + b)\f$.
struct PerceptronModel {
    std::vector<double> weights;              ///< Weight vector \f$w\f$.
    double              bias           = 0.0;  ///< Bias \f$b\f$.
    std::size_t         epochs_run     = 0;    ///< Passes over the data actually made.
    bool                converged      = false;///< True if an entire epoch made no mistake.
    std::size_t         total_mistakes = 0;    ///< Total updates across all epochs.
};

/// \brief Train a Perceptron by the mistake-driven additive rule.
///
/// Labels must be \f$\pm 1\f$. For each example in order, if
/// \f$y_i(w\cdot x_i + b) \le 0\f$ (a mistake or a point exactly on the boundary),
/// update \f$w \leftarrow w + \eta\,y_i\,x_i\f$ and \f$b \leftarrow b + \eta\,y_i\f$.
/// A full pass with no mistakes proves linear separation has been achieved and
/// training stops early (\p converged = true). On non-separable data it runs the
/// full \p max_epochs and returns the last weights.
///
/// \param features  Row-major design matrix (each row an example).
/// \param labels    Class labels, each \f$+1\f$ or \f$-1\f$.
/// \param max_epochs  Maximum passes over the data.
/// \param learning_rate  Step size \f$\eta>0\f$ (only rescales \f$w,b\f$ for the plain Perceptron).
inline PerceptronModel perceptron_train(const std::vector<std::vector<double>>& features,
                                        const std::vector<int>&                 labels,
                                        std::size_t                             max_epochs    = 100,
                                        double                                  learning_rate = 1.0) {
    if (features.size() != labels.size()) throw std::invalid_argument("perceptron: size mismatch");
    PerceptronModel model;
    if (features.empty()) return model;
    const std::size_t dim = features.front().size();
    model.weights.assign(dim, 0.0);

    for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
        bool mistake_this_epoch = false;
        for (std::size_t i = 0; i < features.size(); ++i) {
            double score = model.bias;
            for (std::size_t j = 0; j < dim; ++j) score += model.weights[j] * features[i][j];
            const double y = static_cast<double>(labels[i]);
            if (y * score <= 0.0) {  // mistake (or on the boundary)
                for (std::size_t j = 0; j < dim; ++j) model.weights[j] += learning_rate * y * features[i][j];
                model.bias += learning_rate * y;
                ++model.total_mistakes;
                mistake_this_epoch = true;
            }
        }
        model.epochs_run = epoch + 1;
        if (!mistake_this_epoch) {
            model.converged = true;
            break;
        }
    }
    return model;
}

/// Predict \f$+1\f$ or \f$-1\f$ for a feature vector with a trained Perceptron.
inline int perceptron_predict(const PerceptronModel& model, const std::vector<double>& x) {
    double score = model.bias;
    for (std::size_t j = 0; j < model.weights.size() && j < x.size(); ++j) score += model.weights[j] * x[j];
    return score > 0.0 ? 1 : -1;
}

/// @brief Owned positive-weight Boolean threshold rule learned by Winnow2.
///
/// Predicts one exactly when sum(weights[j]*x[j]) > threshold; equality predicts
/// zero. No bias or probability calibration is fitted. Concurrent read-only
/// predictions are safe when the instance is not mutated.
/// @par Example
/// @code
/// const auto model=dm::algorithms::winnow_train({{1,0},{0,1},{0,0}},{1,1,0});
/// const auto dimension=model.weights.size(); // 2
/// @endcode
/// @par References
/// - N. Littlestone (1988), Learning Quickly When Irrelevant Attributes Abound,
///   https://ai.stanford.edu/~pabbeel/depth_qual/littlestone1988.pdf
/// @see winnow_train(), winnow_predict()
struct WinnowModel {
    std::vector<double> weights; ///< Owned positive finite weights, initially all one.
    double threshold{0}; ///< Finite nonnegative decision threshold; a strict comparison is used.
    double alpha{2}; ///< Finite update multiplier greater than one.
    std::size_t epochs_run{0}; ///< Number of completed passes, including any final mistake-free pass.
    bool converged{false}; ///< True only after observing a complete pass without an online mistake.
    std::size_t total_mistakes{0}; ///< Cumulative online updates; not the final classification error count.
};

/// @brief Internal trace records shared by the ordinary and instrumented Winnow entry points.
namespace detail {
/// @brief One completed pass's online update counts and post-pass state summary.
struct WinnowEpoch {
    std::size_t epoch{0}; ///< One-based pass number.
    std::size_t false_positives{0}; ///< Online demotions during this pass.
    std::size_t false_negatives{0}; ///< Online promotions during this pass.
    std::size_t cumulative_mistakes{0}; ///< Updates through this pass.
    double accuracy{0}; ///< Accuracy of the final weights on the supplied training rows.
    double total_weight{0}; ///< Sum of weights after the pass, before any next-pass updates.
};
/// @brief Validated common training loop, optionally retaining pass records.
/// @param features Rectangular Boolean rows.
/// @param labels Boolean responses aligned with rows.
/// @param alpha Finite multiplier greater than one.
/// @param threshold Negative finite selects dimension; otherwise finite nonnegative.
/// @param max_epochs Maximum passes, possibly zero.
/// @param history Optional output vector, cleared before training and owned by caller.
/// @return Owned fitted model; inputs are unchanged.
/// @throws std::invalid_argument If options, shapes or Boolean values are invalid.
/// @throws std::overflow_error If arithmetic overflows or a positive weight underflows to zero.
WinnowModel winnow_train_impl(const std::vector<std::vector<int>>& features,
                              const std::vector<int>& labels, double alpha, double threshold,
                              std::size_t max_epochs, std::vector<WinnowEpoch>* history);
} // namespace detail

/// @brief Train Winnow2 by promoting active weights on false negatives and demoting on false positives.
///
/// Starts weights at one, visits rows in supplied order on each pass, and
/// stops after a complete mistake-free pass. Correct predictions never update
/// weights. Other attributes can be promoted on positive examples, including
/// attributes irrelevant to a target concept. The O(k log d) mistake bound
/// requires labels from a fixed k-literal monotone disjunction over d Boolean
/// attributes, alpha=2 and the default threshold d; it is not a guarantee on
/// arbitrary labels or generalisation to unseen rows. Finite-precision range
/// failures are reported, rather than silently producing zero/infinite weights.
/// O(epochs*n*d) time and O(d) model storage; input is borrowed and unchanged.
/// @param features Rectangular Boolean rows; an empty matrix returns an untrained empty model.
/// @param labels Each zero or one, one label per row.
/// @param alpha Finite multiplier greater than one, default two.
/// @param threshold Finite threshold; any negative value selects d, default -1.
/// @param max_epochs Maximum passes, default 100; zero returns the initial weights without fitting.
/// @return Owned model with pass/update counts; empty input is not marked converged.
/// @throws std::invalid_argument If shapes, Boolean values or options are invalid.
/// @throws std::overflow_error If weights/scores overflow or positive weights underflow to zero.
/// @par Example
/// @code
/// const auto model=dm::algorithms::winnow_train({{1,0},{0,1},{0,0}},{1,1,0});
/// const bool fitted=model.converged; // true
/// @endcode
/// @par References
/// - N. Littlestone (1988), Learning Quickly When Irrelevant Attributes Abound:
///   A New Linear-threshold Algorithm, Machine Learning 2, 285-318,
///   https://ai.stanford.edu/~pabbeel/depth_qual/littlestone1988.pdf
/// @see WinnowModel, winnow_score(), winnow_fit()
inline WinnowModel winnow_train(const std::vector<std::vector<int>>& features,
                                const std::vector<int>& labels, double alpha=2.0,
                                double threshold=-1.0, std::size_t max_epochs=100) {
    return detail::winnow_train_impl(features,labels,alpha,threshold,max_epochs,nullptr);
}

/// @brief Compute a checked Boolean row's weighted score without changing the model.
/// @param model Finite nonnegative threshold and positive finite weights; alpha metadata is unused.
/// @param x Boolean values, with exactly model.weights.size() entries.
/// @return Sum of weights of active attributes; zero for an empty row/model.
/// @throws std::invalid_argument If dimensions, Boolean values, weights or threshold are invalid.
/// @throws std::overflow_error If the score exceeds the finite double range.
/// @par Example
/// @code
/// const auto model=dm::algorithms::winnow_train({{1,0},{0,1},{0,0}},{1,1,0});
/// const double score=dm::algorithms::winnow_score(model,{0,0}); // 0
/// @endcode
/// @see winnow_predict()
double winnow_score(const WinnowModel& model,const std::vector<int>& x);

/// @brief Predict zero or one by the strict test winnow_score(model,x) > model.threshold.
/// @param model Read-only fitted model, with winnow_score()'s validity constraints.
/// @param x Boolean row with exactly as many entries as weights.
/// @return One above the threshold; zero at or below it.
/// @throws std::invalid_argument If model or row is invalid.
/// @throws std::overflow_error If the weighted score overflows.
/// @par Example
/// @code
/// const auto model=dm::algorithms::winnow_train({{1,0},{0,1},{0,0}},{1,1,0});
/// const int positive=dm::algorithms::winnow_predict(model,{1,0}); // 1
/// @endcode
/// @see winnow_score(), winnow_train()
inline int winnow_predict(const WinnowModel& model,const std::vector<int>& x) {
    return winnow_score(model,x)>model.threshold ? 1 : 0;
}

} // namespace datamunge::algorithms
