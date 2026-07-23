#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct PCAOptions {
    /// @brief Mean-center each feature before computing covariance. Almost always left true.
    bool center{true};
    /// @brief Divide each (centered) feature by its sample standard deviation before computing
    ///        covariance -- i.e. run PCA on the correlation matrix rather than the covariance
    ///        matrix, matching R's prcomp(scale. = TRUE). Recommended whenever features are on
    ///        different scales, which is why it defaults to true here (R itself defaults to
    ///        FALSE but documents TRUE as the generally-recommended choice).
    bool scale{true};
};

/// @brief Principal Component Analysis via eigendecomposition of the (correlation or covariance)
///        matrix, fit from a DataFrame and a list of numeric feature columns. Rows with a null
///        value in any feature column are dropped before fitting. Always computes all
///        min(observations - 1, features) non-trivial components -- callers pick how many to
///        use via explained_variance_ratio()/cumulative_explained_variance_ratio() rather than
///        requesting a truncated fit up front.
class PCA {
public:
    PCA(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, PCAOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t num_components() const { return loadings_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with scores()/plot_scores().
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief Eigenvalues of the covariance/correlation matrix, one per component, descending.
    [[nodiscard]] const std::vector<double>& explained_variance() const { return explained_variance_; }
    [[nodiscard]] std::vector<double> explained_variance_ratio() const;
    [[nodiscard]] std::vector<double> cumulative_explained_variance_ratio() const;

    /// @brief p x k "rotation" matrix: column j is the j-th principal axis, expressed in the
    ///        original (centered/scaled) feature space.
    [[nodiscard]] const linalg::DenseMatrix<double>& loadings() const { return loadings_; }
    /// @brief n x k matrix: row i is observation i's coordinates in principal-component space.
    [[nodiscard]] const linalg::DenseMatrix<double>& scores() const { return scores_; }

    [[nodiscard]] std::vector<double> component_loadings(std::size_t component_index) const;
    [[nodiscard]] std::vector<double> component_scores(std::size_t component_index) const;

    /// @brief Projects new data onto the already-fitted components, using the training mean/
    ///        scale (not newdata's own). Throws if any feature column is missing or null.
    [[nodiscard]] linalg::DenseMatrix<double> transform(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    /// @brief Scatter of scores in the (component_x, component_y) plane, single series.
    [[nodiscard]] plot::RPlot plot_scores(std::size_t component_x = 0, std::size_t component_y = 1) const;
    /// @brief Same, colored by an external grouping vector (e.g. a withheld response column).
    ///        `group_labels` must have one entry per fitted observation, in kept_row_indices() order.
    [[nodiscard]] plot::RPlot plot_scores(const std::vector<std::string>& group_labels, std::size_t component_x = 0,
                                          std::size_t component_y = 1) const;
    /// @brief Scree plot: percent of variance explained by each component, as a bar chart.
    [[nodiscard]] plot::RPlot plot_scree() const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    PCAOptions                options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       mean_;
    std::vector<double>       scale_;
    std::vector<double>       explained_variance_;
    linalg::DenseMatrix<double> loadings_;
    linalg::DenseMatrix<double> scores_;
    std::size_t                 observations_{0};
};

} // namespace datamunge::stats
