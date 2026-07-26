#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct FactorAnalysisOptions {
    /// @brief Number of common latent factors to extract (k). Must be >= 1 and small enough that
    ///        the model is identifiable: k <= p, and in practice (p - k)^2 >= p + k.
    std::size_t n_factors{2};
    /// @brief Fit on the correlation matrix (each feature standardized to unit variance) rather
    ///        than the covariance matrix. Almost always left true for factor analysis, so that
    ///        loadings, communalities and uniquenesses are on a common, unit-total-variance scale.
    bool standardize{true};
    /// @brief Orthogonal rotation applied to the extracted loadings for interpretability. One of
    ///        "none" or "varimax" (Kaiser-normalized). Rotation leaves communalities and the model
    ///        fit unchanged; it only redistributes the common variance across factors.
    std::string rotation{"varimax"};
    /// @brief Maximum principal-axis-factoring iterations (communality refinement).
    std::size_t max_iter{1000};
    /// @brief Convergence tolerance on the largest per-variable communality change between
    ///        successive PAF iterations.
    double tol{1e-6};
};

/// @brief Exploratory Factor Analysis (Spearman 1904; Thurstone 1947) fit by principal axis
///        factoring. Unlike PCA -- which rotates the axes of *total* variance and keeps every
///        component -- factor analysis posits a generative latent-variable model,
///        x = mu + Lambda f + epsilon, with k common factors f ~ N(0, I), a p x k loading matrix
///        Lambda, and variable-specific noise epsilon ~ N(0, Psi) with Psi diagonal. It therefore
///        splits each variable's variance into a *communality* (shared, explained by the common
///        factors) and a *uniqueness* (variable-specific), and models only the off-diagonal
///        covariance structure. Principal axis factoring estimates Lambda by iteratively
///        eigendecomposing the correlation matrix with its diagonal replaced by the current
///        communality estimates, until the communalities stop changing. An optional varimax
///        rotation then rotates Lambda toward "simple structure" (each variable loading heavily on
///        few factors) without changing the fit. Rows with a null value in any feature column are
///        dropped before fitting.
class FactorAnalysis {
public:
    FactorAnalysis(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                   FactorAnalysisOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t num_factors() const { return loadings_.cols(); }

    /// @brief Row indices (into the original `data`) that survived null-dropping, in fitted order.
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief p x k loading matrix Lambda (post-rotation if a rotation was requested): entry (j, c)
    ///        is variable j's loading on factor c.
    [[nodiscard]] const linalg::DenseMatrix<double>& loadings() const { return loadings_; }
    [[nodiscard]] std::vector<double> factor_loadings(std::size_t factor_index) const;

    /// @brief Communality of each variable, h_j^2 = sum_c Lambda_{jc}^2: the fraction of variable
    ///        j's (standardized) variance explained by the common factors. Length p.
    [[nodiscard]] const std::vector<double>& communalities() const { return communalities_; }
    /// @brief Uniqueness of each variable, psi_j = 1 - h_j^2 (on the correlation scale): the
    ///        variable-specific variance the common factors do not explain. Length p.
    [[nodiscard]] const std::vector<double>& uniquenesses() const { return uniquenesses_; }

    /// @brief Sum of squared loadings on each factor (post-rotation), i.e. the amount of total
    ///        (standardized) variance that factor accounts for. Length k.
    [[nodiscard]] const std::vector<double>& variance_explained() const { return variance_explained_; }
    /// @brief variance_explained() divided by the number of variables p. Length k.
    [[nodiscard]] std::vector<double> proportion_variance() const;

    /// @brief n x k factor score matrix, estimated by the Thomson/regression method
    ///        F = Z R^{-1} Lambda (Z the standardized data, R the sample correlation matrix).
    [[nodiscard]] const linalg::DenseMatrix<double>& scores() const { return scores_; }
    [[nodiscard]] std::vector<double> factor_scores(std::size_t factor_index) const;

    /// @brief Number of principal-axis-factoring iterations actually run, and whether the
    ///        communalities converged within max_iter.
    [[nodiscard]] std::size_t iterations() const { return iterations_; }
    [[nodiscard]] bool        converged() const { return converged_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    /// @brief Scatter of factor scores in the (factor_x, factor_y) plane, colored by an external
    ///        grouping vector with one entry per fitted observation (in kept_row_indices() order).
    [[nodiscard]] plot::RPlot plot_scores(const std::vector<std::string>& group_labels, std::size_t factor_x = 0,
                                          std::size_t factor_y = 1) const;
    /// @brief Loading plot: each variable placed at its (factor_x, factor_y) loadings and labeled,
    ///        with the unit circle drawn for reference -- the standard way to read simple structure.
    [[nodiscard]] plot::RPlot plot_loadings(std::size_t factor_x = 0, std::size_t factor_y = 1) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    FactorAnalysisOptions     options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       communalities_;
    std::vector<double>       uniquenesses_;
    std::vector<double>       variance_explained_;
    linalg::DenseMatrix<double> loadings_;
    linalg::DenseMatrix<double> scores_;
    std::size_t                 observations_{0};
    std::size_t                 iterations_{0};
    bool                        converged_{false};
};

} // namespace datamunge::stats
