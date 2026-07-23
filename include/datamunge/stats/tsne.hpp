#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/knn_classifier.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct TSNEOptions {
    std::size_t n_components{2};
    double perplexity{30.0};
    std::size_t max_iterations{1000};
    double learning_rate{200.0};
    double early_exaggeration{12.0};
    std::size_t early_exaggeration_iterations{250};
    double initial_momentum{0.5};
    double final_momentum{0.8};
    std::size_t momentum_switch_iteration{250};
    DistanceMetric metric{DistanceMetric::Euclidean};
    std::uint64_t seed{42};
};

/// @brief t-distributed Stochastic Neighbor Embedding (van der Maaten & Hinton, 2008): converts
///        pairwise distances into conditional probabilities (calibrated per-point via a
///        perplexity-targeted binary search over a Gaussian bandwidth), symmetrizes them into a
///        joint distribution P, and iteratively moves points in a low-dimensional space via
///        momentum-based gradient descent so that a Student-t-distributed joint distribution Q
///        over the embedded points matches P as closely as possible (minimizing their KL
///        divergence). Unlike PCA/MDS, t-SNE preserves local neighborhood structure rather than
///        global distances or variance, and is primarily useful for visualization (typically
///        n_components 2 or 3). Rows with a null value in any feature column are dropped before
///        fitting.
class TSNE {
public:
    TSNE(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, TSNEOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with embedding()/plot_embedding().
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief The achieved perplexity for each point after the per-point binary search over the
    ///        Gaussian bandwidth beta_i -- should be close to options().perplexity for every
    ///        point if the calibration converged well.
    [[nodiscard]] const std::vector<double>& achieved_perplexity() const { return achieved_perplexity_; }

    /// @brief n x n_components matrix of embedded coordinates.
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] plot::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    [[nodiscard]] plot::RPlot plot_embedding(const std::vector<std::string>& group_labels, std::size_t dimension_x = 0,
                                             std::size_t dimension_y = 1) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    TSNEOptions                 options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       achieved_perplexity_;
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
};

} // namespace datamunge::stats
