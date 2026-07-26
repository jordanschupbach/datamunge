#pragma once

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct CNNClassifierOptions {
    /// @brief Number of convolutional filters (feature maps) in the single conv layer.
    std::size_t n_filters{8};
    /// @brief Square convolution kernel side length (valid convolution, stride 1).
    std::size_t kernel_size{3};
    /// @brief Square max-pooling window side length (non-overlapping).
    std::size_t pool_size{2};
    /// @brief Width of the fully-connected hidden layer between the flattened feature maps and the
    ///        softmax output.
    std::size_t dense_hidden{32};
    double      learning_rate{0.05};
    std::size_t max_epochs{40};
    std::size_t batch_size{16};
    /// @brief L2 penalty on the convolution filters and dense weights (not biases).
    double        l2{0.0};
    std::uint64_t seed{42};
};

struct CNNClassifierPrediction {
    std::vector<std::string>         class_label;
    std::vector<std::vector<double>> probabilities; // [image][class]
};

/// @brief A small convolutional neural network image classifier, trained by backpropagation with
///        mini-batch stochastic gradient descent. The architecture is the classic LeNet-style
///        stack for a single-channel (grayscale) image:
///
///          conv (n_filters of kernel_size x kernel_size) -> ReLU -> max-pool
///            -> flatten -> dense(dense_hidden) -> ReLU -> softmax
///
///        Unlike a fully-connected network, a convolutional layer *shares* one small set of
///        weights across every spatial position, so it learns local, translation-equivariant
///        feature detectors (edges, corners) with far fewer parameters, and pooling then confers a
///        degree of translation *invariance*. Images are passed directly as grayscale DenseMatrix
///        objects (all the same height x width) rather than through the tabular formula interface
///        the other classifiers use, since image structure is intrinsically 2D.
class CNNClassifier {
public:
    CNNClassifier(const std::vector<linalg::DenseMatrix<double>>& images, const std::vector<std::string>& labels,
                  CNNClassifierOptions options = {});

    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t image_height() const { return height_; }
    [[nodiscard]] std::size_t image_width() const { return width_; }

    /// @brief Total number of trainable parameters (filters, biases, and dense weights).
    [[nodiscard]] std::size_t n_parameters() const;
    /// @brief The learned convolution filters, each a kernel_size x kernel_size matrix -- e.g. for
    ///        visualizing the edge detectors the network discovers.
    [[nodiscard]] const std::vector<linalg::DenseMatrix<double>>& filters() const { return filters_; }

    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    [[nodiscard]] linalg::DenseMatrix<double>     confusion_matrix() const;

    [[nodiscard]] const std::vector<double>& loss_curve() const { return loss_curve_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string> predict(const std::vector<linalg::DenseMatrix<double>>& images) const;
    [[nodiscard]] CNNClassifierPrediction   predict_detail(const std::vector<linalg::DenseMatrix<double>>& images) const;
    /// @brief Accuracy on a labeled image set.
    [[nodiscard]] double accuracy(const std::vector<linalg::DenseMatrix<double>>& images,
                                  const std::vector<std::string>& labels) const;

    /// @brief Layer-wise relevance propagation (Bach et al., 2015): explains one image's prediction
    ///        by propagating the target class's output score back through every layer to the input
    ///        pixels, returning an image_height x image_width heatmap of how much each pixel
    ///        supported the decision. Uses the z+ rule (LRP-alpha1beta0), which suits this ReLU
    ///        network on non-negative pixel data and yields a non-negative, approximately
    ///        conservation-preserving relevance map. @p target_class < 0 (the default) explains the
    ///        predicted class; otherwise it explains the given class index (into classes()).
    [[nodiscard]] linalg::DenseMatrix<double> relevance(const linalg::DenseMatrix<double>& image,
                                                        int target_class = -1) const;

    /// @brief Loss curve versus epoch.
    [[nodiscard]] plot::RPlot plot_loss_curve() const;

private:
    void                fit(const std::vector<linalg::DenseMatrix<double>>& images);
    std::vector<double> forward_probs(const linalg::DenseMatrix<double>& image) const;

    CNNClassifierOptions options_;

    std::vector<std::string> classes_;
    std::vector<std::string> training_labels_;
    std::vector<std::string> fitted_classes_;

    // Parameters.
    std::vector<linalg::DenseMatrix<double>> filters_; // n_filters of (k x k)
    std::vector<double>                       conv_bias_;
    linalg::DenseMatrix<double>               W1_, W2_; // dense hidden and output weights
    std::vector<double>                       b1_, b2_;

    std::size_t height_{0}, width_{0};
    std::size_t conv_h_{0}, conv_w_{0}, pool_h_{0}, pool_w_{0}, flat_dim_{0};

    std::vector<double> loss_curve_;
    std::size_t          observations_{0};
};

} // namespace datamunge::stats
