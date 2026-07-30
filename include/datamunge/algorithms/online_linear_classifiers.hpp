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

/// A trained Winnow classifier over boolean features: predicts \f$1\f$ iff \f$w\cdot x > \theta\f$.
struct WinnowModel {
    std::vector<double> weights;               ///< Positive multiplicative weights \f$w_i\f$.
    double              threshold      = 0.0;   ///< Threshold \f$\theta\f$.
    double              alpha          = 2.0;   ///< Update multiplier \f$\alpha>1\f$.
    std::size_t         epochs_run     = 0;
    bool                converged      = false;
    std::size_t         total_mistakes = 0;
};

/// \brief Train the Winnow2 algorithm by multiplicative updates.
///
/// Features are boolean (\f$0/1\f$) and labels are \f$0/1\f$. Weights start at 1 and
/// the threshold defaults to the number of features \f$n\f$. The prediction is
/// \f$\hat y = [\,w\cdot x > \theta\,]\f$; on a mistake only the *active* weights
/// (\f$x_i=1\f$) are changed:
///   - false negative (\f$\hat y=0, y=1\f$): *promote*, \f$w_i \leftarrow \alpha\,w_i\f$;
///   - false positive (\f$\hat y=1, y=0\f$): *demote*, \f$w_i \leftarrow w_i/\alpha\f$.
/// Correct weights are untouched. Winnow tolerates many irrelevant features because
/// their weights are never promoted; its mistake bound is \f$O(k\log n)\f$ for a
/// \f$k\f$-literal monotone disjunction over \f$n\f$ attributes.
///
/// \param features  Boolean design matrix.
/// \param labels    Class labels, each \f$0\f$ or \f$1\f$.
/// \param alpha     Multiplier \f$\alpha>1\f$.
/// \param threshold Decision threshold; a negative value selects the default \f$\theta=n\f$.
/// \param max_epochs Maximum passes over the data.
inline WinnowModel winnow_train(const std::vector<std::vector<int>>& features,
                                const std::vector<int>&              labels,
                                double                               alpha      = 2.0,
                                double                               threshold  = -1.0,
                                std::size_t                          max_epochs = 100) {
    if (features.size() != labels.size()) throw std::invalid_argument("winnow: size mismatch");
    WinnowModel model;
    if (features.empty()) return model;
    const std::size_t dim = features.front().size();
    model.weights.assign(dim, 1.0);
    model.alpha     = alpha;
    model.threshold = threshold < 0.0 ? static_cast<double>(dim) : threshold;

    for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
        bool mistake_this_epoch = false;
        for (std::size_t i = 0; i < features.size(); ++i) {
            double dot = 0.0;
            for (std::size_t j = 0; j < dim; ++j) dot += model.weights[j] * features[i][j];
            const int predicted = dot > model.threshold ? 1 : 0;
            if (predicted != labels[i]) {
                const bool promote = (labels[i] == 1);  // false negative -> promote active weights
                for (std::size_t j = 0; j < dim; ++j)
                    if (features[i][j] != 0) model.weights[j] = promote ? model.weights[j] * alpha : model.weights[j] / alpha;
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

/// Predict \f$0\f$ or \f$1\f$ for a boolean feature vector with a trained Winnow model.
inline int winnow_predict(const WinnowModel& model, const std::vector<int>& x) {
    double dot = 0.0;
    for (std::size_t j = 0; j < model.weights.size() && j < x.size(); ++j) dot += model.weights[j] * x[j];
    return dot > model.threshold ? 1 : 0;
}

}  // namespace datamunge::algorithms
