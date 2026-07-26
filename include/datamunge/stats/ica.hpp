#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct ICAOptions {
    /// @brief Number of independent components to extract. Must be >= 1 and <= the number of
    ///        features (you cannot recover more independent sources than you have mixtures).
    std::size_t n_components{2};
    /// @brief Contrast function whose expected value FastICA maximizes as a proxy for
    ///        non-Gaussianity. One of "logcosh" (g(u)=tanh u; robust, the usual default),
    ///        "exp" (g(u)=u exp(-u^2/2); good for very super-Gaussian sources), or
    ///        "cube" (g(u)=u^3; the classic excess-kurtosis contrast).
    std::string contrast{"logcosh"};
    /// @brief Maximum fixed-point iterations per component.
    std::size_t max_iter{200};
    /// @brief Convergence tolerance: a component is converged when |w_new . w_old| exceeds
    ///        1 - tol between successive iterates.
    double tol{1e-6};
    /// @brief Seed for the random initialization of the unmixing vectors (results are otherwise
    ///        deterministic).
    std::uint64_t seed{42};
};

/// @brief Independent Component Analysis via the deflationary FastICA algorithm (Hyvarinen & Oja,
///        1997). ICA solves blind source separation: given observations modeled as unknown linear
///        mixtures x = A s of statistically *independent* latent sources s, it estimates an
///        unmixing matrix W so that W x recovers the sources. Where PCA/factor analysis only
///        remove second-order correlation (yielding *uncorrelated* components), ICA removes all
///        higher-order dependence too by maximizing each component's *non-Gaussianity* -- justified
///        by the central limit theorem, since any mixture of independent variables is more Gaussian
///        than its constituents, so the least-Gaussian projections are the original sources. The
///        data is first centered and PCA-whitened (decorrelated to unit variance), after which the
///        remaining unmixing rotation is found one component at a time by a fixed-point iteration,
///        each new direction orthogonalized against those already found. Sources are recovered only
///        up to sign, scale, and permutation. Rows with a null value in any feature column are
///        dropped before fitting.
class ICA {
public:
    ICA(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, ICAOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return sources_.cols(); }

    /// @brief Row indices (into the original `data`) that survived null-dropping, in fitted order.
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief n x k matrix of recovered independent components: column c is the c-th estimated
    ///        source signal, standardized to zero mean and unit variance.
    [[nodiscard]] const linalg::DenseMatrix<double>& sources() const { return sources_; }
    [[nodiscard]] std::vector<double> component(std::size_t index) const;

    /// @brief k x p unmixing matrix W mapping a centered observation to its source estimates
    ///        (source row = W (x - mean)).
    [[nodiscard]] const linalg::DenseMatrix<double>& unmixing_matrix() const { return unmixing_; }
    /// @brief p x k estimated mixing matrix A (the pseudo-inverse of W): column c is the direction
    ///        in feature space along which source c is expressed.
    [[nodiscard]] const linalg::DenseMatrix<double>& mixing_matrix() const { return mixing_; }
    /// @brief Per-feature mean subtracted before fitting.
    [[nodiscard]] const std::vector<double>& mean() const { return mean_; }

    /// @brief Fixed-point iterations used to extract each component, and whether every component
    ///        converged within max_iter.
    [[nodiscard]] const std::vector<std::size_t>& iterations() const { return iterations_; }
    [[nodiscard]] bool                            converged() const { return converged_; }

    /// @brief Excess kurtosis of each recovered source -- a measure of non-Gaussianity (0 for a
    ///        Gaussian; negative for sub-Gaussian, e.g. a sine or uniform; positive for
    ///        super-Gaussian, e.g. a spiky signal). Length k.
    [[nodiscard]] const std::vector<double>& source_kurtosis() const { return source_kurtosis_; }

    /// @brief Projects new data onto the fitted unmixing directions using the training mean.
    [[nodiscard]] linalg::DenseMatrix<double> transform(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    /// @brief Scatter of two recovered sources, colored by an external grouping vector with one
    ///        entry per fitted observation (in kept_row_indices() order).
    [[nodiscard]] plot::RPlot plot_sources(const std::vector<std::string>& group_labels, std::size_t component_x = 0,
                                           std::size_t component_y = 1) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    ICAOptions                options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       mean_;
    std::vector<double>       source_kurtosis_;
    std::vector<std::size_t> iterations_;
    linalg::DenseMatrix<double> sources_;
    linalg::DenseMatrix<double> unmixing_;
    linalg::DenseMatrix<double> mixing_;
    std::size_t                 observations_{0};
    bool                        converged_{false};
};

} // namespace datamunge::stats
