#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct KernelPCAOptions {
    std::size_t n_components{2};
    /// @brief One of "linear", "rbf", "polynomial".
    std::string kernel{"rbf"};
    /// @brief rbf: exp(-gamma*||x-y||^2); polynomial: scale on the dot product. Must be > 0 for
    ///        rbf/polynomial.
    double gamma{1.0};
    /// @brief polynomial kernel exponent.
    double degree{3.0};
    /// @brief polynomial kernel offset.
    double coef0{1.0};
};

/// @brief Kernel PCA (Scholkopf, Smola & Muller, 1998): principal component analysis generalized
///        to a nonlinear feature space via the "kernel trick" -- eigendecomposes the centered
///        n x n kernel (Gram) matrix instead of the p x p covariance matrix used by plain PCA,
///        so it can capture nonlinear structure that PCA/MDS (both linear) cannot. With a linear
///        kernel it reproduces plain unscaled PCA's scores (up to a sign flip per component).
///        Rows with a null value in any feature column are dropped before fitting.
class KernelPCA {
public:
    KernelPCA(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
              KernelPCAOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

    /// @brief Eigenvalues of the centered kernel matrix corresponding to the used components
    ///        (length == n_components()), descending.
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] plot::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    [[nodiscard]] plot::RPlot plot_embedding(const std::vector<std::string>& group_labels,
                                             std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    KernelPCAOptions          options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       eigenvalues_;
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
};

} // namespace datamunge::stats
