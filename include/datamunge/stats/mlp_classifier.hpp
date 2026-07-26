#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct MLPClassifierOptions {
    /// @brief Widths of the hidden layers, input-to-output. Empty means no hidden layer at all --
    ///        a bare softmax, i.e. multinomial logistic regression with a linear decision boundary.
    std::vector<std::size_t> hidden_layer_sizes{16};
    /// @brief Hidden-unit nonlinearity: "relu", "tanh", or "sigmoid". (The output layer is always
    ///        softmax.)
    std::string activation{"relu"};
    /// @brief SGD step size.
    double learning_rate{0.05};
    /// @brief Number of passes over the training data.
    std::size_t max_epochs{400};
    /// @brief Mini-batch size; 0 means full-batch gradient descent.
    std::size_t batch_size{32};
    /// @brief L2 penalty on the weights (not the biases). 0 disables it.
    double l2{0.0};
    /// @brief Seed for weight initialization and mini-batch shuffling (the fit is otherwise
    ///        deterministic).
    std::uint64_t seed{42};
    /// @brief Standardize predictors to zero mean / unit variance before training.
    bool standardize{true};
};

struct MLPClassifierPrediction {
    std::vector<std::string>         class_label;   // "" where predictors were missing
    std::vector<std::vector<double>> probabilities; // [row][class], softmax outputs
};

/// @brief A feed-forward neural network (multi-layer perceptron) classifier, trained by
///        backpropagation with mini-batch stochastic gradient descent. Each layer computes an
///        affine map followed by a nonlinearity, z = W a + b, a' = act(z); the final layer's
///        outputs pass through a softmax to give class probabilities, and training minimizes the
///        cross-entropy loss. With one or more hidden layers the network learns nonlinear decision
///        boundaries no linear model can express (the classic XOR/two-moons problems); with an
///        empty hidden-layer list it reduces exactly to multinomial logistic regression. Uses the
///        same formula + DataFrame constructor style as the other classifiers. Rows with a null
///        value in any model column are dropped before fitting.
class MLPClassifier {
public:
    MLPClassifier(const dstruct::DataFrame& data, const std::string& formula, MLPClassifierOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }

    /// @brief Layer widths including input and output, e.g. {2, 16, 3} for 2 features, one hidden
    ///        layer of 16 units, and 3 classes.
    [[nodiscard]] std::vector<std::size_t> architecture() const;
    /// @brief Total number of trainable parameters (all weights and biases).
    [[nodiscard]] std::size_t n_parameters() const;

    /// @brief Resubstitution predictions on the training data (the network does have a genuine
    ///        training phase, so unlike KNN this is a real, if optimistic, in-sample fit).
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    /// @brief Rows = actual class, columns = predicted class, both ordered as classes().
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    /// @brief Mean cross-entropy loss on the training set after each epoch (length max_epochs).
    [[nodiscard]] const std::vector<double>& loss_curve() const { return loss_curve_; }
    [[nodiscard]] bool                       converged() const { return converged_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] MLPClassifierPrediction  predict_detail(const dstruct::DataFrame& newdata) const;
    /// @brief Accuracy on labeled newdata (its formula response column must be present).
    [[nodiscard]] double accuracy(const dstruct::DataFrame& newdata) const;

    /// @brief Background grid of predicted class regions in the (x_feature, y_feature) plane over
    ///        the training points; only valid for a two-predictor model.
    [[nodiscard]] plot::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                    std::size_t grid_resolution = 80) const;
    /// @brief The training loss curve versus epoch.
    [[nodiscard]] plot::RPlot plot_loss_curve() const;

private:
    void                fit(const dstruct::DataFrame& data);
    std::vector<double> forward_probs(const std::vector<double>& x_std) const; // softmax class probabilities

    Formula              formula_;
    DesignInfo           design_;
    MLPClassifierOptions options_;

    std::vector<std::string> classes_;
    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_mean_;
    std::vector<double>      feature_scale_;
    std::vector<double>      feature_min_;
    std::vector<double>      feature_max_;

    // Parameters: weights_[l] is (units_l x units_{l-1}), biases_[l] is (units_l), for each layer l.
    std::vector<linalg::DenseMatrix<double>> weights_;
    std::vector<std::vector<double>>          biases_;

    std::vector<std::string> training_labels_;
    std::vector<std::string> fitted_classes_;
    std::vector<double>      loss_curve_;
    bool                     converged_{false};
    std::size_t              observations_{0};
};

} // namespace datamunge::stats
