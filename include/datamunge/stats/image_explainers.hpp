#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief A black-box prediction function: maps an image to the (scalar) score of the class being
///        explained -- typically a class probability. Model-agnostic explainers see nothing of the
///        model beyond this callable.
using ImagePredict = std::function<double(const linalg::DenseMatrix<double>&)>;

struct ImageExplanationOptions {
    /// @brief Superpixel grid the image is partitioned into (the "features" being attributed).
    std::size_t patch_rows{4};
    std::size_t patch_cols{4};
    /// @brief Value substituted into a patch that is turned "off" (removed) in a perturbation.
    double baseline{0.0};
    /// @brief Number of perturbation samples for LIME.
    std::size_t n_samples{800};
    /// @brief LIME proximity-kernel width (on the fraction of patches turned off).
    double lime_kernel_width{0.25};
    /// @brief LIME ridge penalty on the surrogate coefficients.
    double lime_l2{1e-3};
    std::uint64_t seed{0};
};

/// @brief LIME (Ribeiro, Singh & Guestrin, 2016): explains one prediction by fitting a simple,
///        interpretable linear model in the neighborhood of the instance. It perturbs the image by
///        randomly switching superpixels on (original) or off (baseline), asks the black-box model
///        for each perturbation's score, and fits a proximity-weighted ridge regression of the
///        score on the on/off pattern; each superpixel's coefficient is its local importance. Fully
///        model-agnostic -- it only ever calls the supplied prediction function.
class LimeImageExplainer {
public:
    LimeImageExplainer(const linalg::DenseMatrix<double>& image, const ImagePredict& predict,
                       ImageExplanationOptions options = {});

    /// @brief patch_rows x patch_cols surrogate coefficients (per-superpixel importance).
    [[nodiscard]] const linalg::DenseMatrix<double>& patch_weights() const { return patch_weights_; }
    /// @brief image_height x image_width map assigning each pixel its superpixel's weight.
    [[nodiscard]] linalg::DenseMatrix<double> pixel_relevance() const;
    [[nodiscard]] double intercept() const { return intercept_; }
    /// @brief Weighted R^2 of the local linear surrogate -- how faithfully it fits the model nearby.
    [[nodiscard]] double local_r_squared() const { return local_r2_; }

private:
    ImageExplanationOptions      options_;
    std::size_t                  height_{0}, width_{0};
    linalg::DenseMatrix<double>  patch_weights_;
    double                       intercept_{0.0};
    double                       local_r2_{0.0};
};

/// @brief SHAP (Lundberg & Lee, 2017) via exact Shapley values from cooperative game theory: each
///        superpixel's attribution is its average marginal contribution to the model's score over
///        all coalitions of the other superpixels. Computed exactly by enumerating every one of the
///        2^P coalitions of the P = patch_rows*patch_cols superpixels (so P must stay small), which
///        makes the result independent of any sampling and exactly satisfies the efficiency axiom:
///        the attributions sum to f(full image) - f(empty image). Model-agnostic.
class ShapImageExplainer {
public:
    ShapImageExplainer(const linalg::DenseMatrix<double>& image, const ImagePredict& predict,
                       ImageExplanationOptions options = {});

    /// @brief patch_rows x patch_cols Shapley values (per-superpixel attribution).
    [[nodiscard]] const linalg::DenseMatrix<double>& patch_shapley() const { return patch_shapley_; }
    [[nodiscard]] linalg::DenseMatrix<double> pixel_relevance() const;
    /// @brief Model score with every superpixel removed (the empty coalition), f(baseline image).
    [[nodiscard]] double base_value() const { return base_value_; }
    /// @brief Model score with every superpixel present, f(original image).
    [[nodiscard]] double full_value() const { return full_value_; }

private:
    ImageExplanationOptions      options_;
    std::size_t                  height_{0}, width_{0};
    linalg::DenseMatrix<double>  patch_shapley_;
    double                       base_value_{0.0};
    double                       full_value_{0.0};
};

} // namespace datamunge::stats
