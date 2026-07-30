#pragma once

/// \file structured_svm.hpp
/// \brief Structured / multiclass support vector machine: the Crammer-Singer
///        formulation, trained by Pegasos-style stochastic subgradient descent.
///
/// A *structured* SVM generalizes the binary max-margin classifier to outputs with
/// internal structure by scoring input-output pairs with a joint feature map
/// \f$\Phi(x,y)\f$ and requiring the correct output to outscore every alternative by a
/// task-defined margin \f$\Delta(y,\hat y)\f$:
/// \f$\; w^\top\Phi(x_i,y_i) \ge w^\top\Phi(x_i,y) + \Delta(y_i,y) - \xi_i\f$ for all
/// \f$y\f$. The canonical special case -- and the one implemented here -- is *multiclass*
/// classification (Crammer & Singer 2001): one weight vector \f$w_c\f$ per class,
/// score \f$w_c^\top x\f$, and the 0/1 margin \f$\Delta(y,c)=[y\ne c]\f$. The multiclass
/// hinge loss is
/// \f[
///   \ell(x_i,y_i) = \max_{c}\big(w_c^\top x_i + [c\ne y_i]\big) - w_{y_i}^\top x_i,
/// \f]
/// minimized (with an \f$\tfrac{\lambda}{2}\sum_c\lVert w_c\rVert^2\f$ regularizer) by
/// the Pegasos subgradient step: find the most-violating class, then move \f$w_{y_i}\f$
/// toward and \f$w_r\f$ away from \f$x_i\f$, after an \f$L_2\f$ shrink. The exact same
/// machinery, with a different \f$\Phi\f$ and loss-augmented inference, handles
/// sequence and parse-tree outputs.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// A trained multiclass (Crammer-Singer) SVM: one weight vector per class (bias-augmented).
struct StructuredSVMModel {
    std::vector<std::vector<double>> weights;    ///< weights[c] has length dim+1 (last entry is the bias).
    std::size_t                      num_classes = 0;
    std::size_t                      dim = 0;

    /// Score of class \p c for feature vector \p x (bias handled internally).
    double score(std::size_t c, const std::vector<double>& x) const {
        double s = weights[c][dim];  // bias
        for (std::size_t j = 0; j < dim; ++j) s += weights[c][j] * x[j];
        return s;
    }
};

/// Hyper-parameters for the Pegasos multiclass-SVM solver.
struct StructuredSVMParameters {
    double        lambda      = 0.01;   ///< L2 regularization strength.
    std::size_t   epochs      = 50;     ///< Passes over the data.
    std::uint64_t seed        = 0;
};

/// \brief Train a Crammer-Singer multiclass SVM by stochastic subgradient descent.
///
/// \param features   Design matrix.
/// \param labels     Class labels in \f$\{0,\dots,K-1\}\f$.
/// \param num_classes  Number of classes \f$K\f$.
/// \param params     Regularization, epochs, seed.
inline StructuredSVMModel structured_svm_train(const std::vector<std::vector<double>>& features,
                                               const std::vector<int>& labels, std::size_t num_classes,
                                               const StructuredSVMParameters& params) {
    StructuredSVMModel model;
    const std::size_t  n = features.size();
    if (n == 0) return model;
    const std::size_t dim = features.front().size();
    model.num_classes     = num_classes;
    model.dim             = dim;
    model.weights.assign(num_classes, std::vector<double>(dim + 1, 0.0));

    std::mt19937_64                            rng(params.seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    std::size_t t = 1;
    for (std::size_t ep = 0; ep < params.epochs; ++ep) {
        for (std::size_t step = 0; step < n; ++step, ++t) {
            const std::size_t i   = pick(rng);
            const int         y   = labels[i];
            const auto&       x   = features[i];
            const double      eta = 1.0 / (params.lambda * static_cast<double>(t));

            // Loss-augmented inference: most-violating class r = argmax_c score(c) + [c != y].
            std::size_t r      = 0;
            double      best   = -1e300;
            for (std::size_t c = 0; c < num_classes; ++c) {
                const double aug = model.score(c, x) + (static_cast<int>(c) == y ? 0.0 : 1.0);
                if (aug > best) { best = aug; r = c; }
            }

            // L2 shrink (Pegasos).
            for (auto& wc : model.weights)
                for (double& v : wc) v *= (1.0 - eta * params.lambda);

            // Subgradient of the hinge: push true class up, violator down.
            if (static_cast<int>(r) != y) {
                for (std::size_t j = 0; j < dim; ++j) {
                    model.weights[y][j] += eta * x[j];
                    model.weights[r][j] -= eta * x[j];
                }
                model.weights[y][dim] += eta;  // bias terms
                model.weights[r][dim] -= eta;
            }
        }
    }
    return model;
}

/// Predict the class with the highest score.
inline std::size_t structured_svm_predict(const StructuredSVMModel& model, const std::vector<double>& x) {
    std::size_t best = 0;
    double      bs   = -1e300;
    for (std::size_t c = 0; c < model.num_classes; ++c) {
        const double s = model.score(c, x);
        if (s > bs) { bs = s; best = c; }
    }
    return best;
}

}  // namespace datamunge::algorithms
