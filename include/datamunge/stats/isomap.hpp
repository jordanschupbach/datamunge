#pragma once

/// @file isomap.hpp
/// @brief Union-kNN Isomap, inspectable graph/geodesic geometry and embedding diagnostics.

#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/knn_classifier.hpp>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Embedding dimension and union-kNN geometry on unscaled numeric features.
/// @par Example
/// @code
/// const dm::stats::IsomapOptions opt{2,25,dm::stats::DistanceMetric::Euclidean};
/// const dm::stats::Isomap model(dm::datasets::iris(),
///   {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"},opt);
/// @endcode
/// @see Isomap
struct IsomapOptions {
  std::size_t    n_components{2}; ///< Positive axis count, less than retained observations.
  std::size_t    n_neighbors{10}; ///< Exactly k other neighbours per directed row; 2<=k<n.
  DistanceMetric metric{DistanceMetric::Euclidean}; ///< Euclidean or Manhattan input metric; no
                                                    ///< automatic standardization.
};

/// @brief Owned union-kNN construction result, including disconnected graphs.
/// Fields align to retained-row order. An undirected edge exists when either endpoint
/// selects the other. Self-edges are excluded; ties use increasing retained-row index.
/// Zero-length edges between duplicates are retained. Diagnostics/plots assume the fields
/// have not been modified from isomap_graph() output. Const methods use no mutable caches.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,2,4});
/// const auto g=dm::stats::isomap_graph(d,{"x"},2);
/// std::print("{}\n",g.component_sizes.size()); // 1
/// @endcode
/// @see isomap_graph(), Isomap
struct IsomapGraph {
  std::vector<std::string>    feature_names;  ///< Numeric input names, in metric order.
  linalg::DenseMatrix<double> feature_values; ///< Retained numeric points, n by p.
  std::vector<std::size_t>    kept_rows;      ///< Original zero-based retained row indices.
  std::vector<std::vector<std::size_t>>
      neighbors; ///< Directed nearest neighbours, n lists of k fitted indices.
  linalg::DenseMatrix<double> distances; ///< Full input distances, finite symmetric n by n.
  linalg::DenseMatrix<double>
      adjacency; ///< Union edge weights; non-edges are infinity, diagonal zero.
  std::vector<std::size_t>
      component_labels; ///< Zero-based component number per fitted row, ordered by first member.
  std::vector<std::size_t>
              component_sizes; ///< Row counts per component, including zero-length connectivity.
  std::size_t n_neighbors{0};  ///< Directed neighbour count used in construction.

  /// @brief Unique undirected edges with original row indices and input weights.
  /// @return Owned row_i/row_j/distance/mutual columns; mutual is one if both endpoints selected
  /// the edge.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,2,4});
  /// const auto g=dm::stats::isomap_graph(d,{"x"},2);
  /// std::print("{}\n",g.edge_table().nrows()); // 5
  /// @endcode
  [[nodiscard]] dstruct::DataFrame edge_table() const;

  /// @brief Connectivity, degree and edge-length summary without an eigensolve.
  /// @return Owned one-row n/k/components/edges/min_degree/max_degree/mean_degree/longest_edge
  /// frame.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,2,4});
  /// const auto g=dm::stats::isomap_graph(d,{"x"},2);
  /// std::print("{:.0f}\n",g.diagnostics().double_at("components",0)); // 1
  /// @endcode
  [[nodiscard]] dstruct::DataFrame diagnostics() const;

  /// @brief Plots union edges projected on two selected input columns, coloured by component.
  /// @param feature_x Zero-based input-feature index.
  /// @param feature_y Zero-based input-feature index; may equal feature_x.
  /// @return Owned editable plot. Projections can hide edge crossings/separation in other
  /// coordinates.
  /// @throws std::out_of_range Feature index outside the input design.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,1,0});
  /// d.add_column("y",std::vector<double>{0,0,1,1});
  /// const auto g=dm::stats::isomap_graph(d,{"x","y"},2);
  /// g.plot().size(700,450).save_svg("isomap_graph.svg");
  /// @endcode
  [[nodiscard]] graphics::RPlot plot(std::size_t feature_x = 0, std::size_t feature_y = 1) const;
};

/// @brief Constructs an inspectable union-kNN graph without requiring connectivity.
/// Nulls are dropped jointly; retained numeric values and distances must be finite.
/// Uses brute-force distances and partial row sorting: O(n^2*p+n^2*log(k)) work and
/// O(n^2+n*p+n*k) storage. The caller owns all returned fields; input is unchanged.
/// @param data Input frame.
/// @param feature_columns Nonempty numeric columns; no automatic rescaling.
/// @param n_neighbors Number of other neighbours per row, 2<=k<n after null omission.
/// @param metric Euclidean (default) or Manhattan; unknown enum values are rejected.
/// @return Owned graph result, including component labels before any shortest-path fitting.
/// @throws std::invalid_argument Missing/non-numeric columns, nonfinite retained values,
/// invalid metric/k, too few complete rows or distance overflow.
/// @par Example
/// See IsomapGraph's runnable example.
/// @par References
/// - J. B. Tenenbaum, V. de Silva and J. C. Langford (2000), A Global Geometric Framework
///   for Nonlinear Dimensionality Reduction. https://doi.org/10.1126/science.290.5500.2319
/// @see Isomap, IsomapGraph
[[nodiscard]] IsomapGraph isomap_graph(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& feature_columns,
    std::size_t                     n_neighbors,
    DistanceMetric                  metric = DistanceMetric::Euclidean);

/// @brief Isomap: classical scaling of shortest-path distances on a union-kNN graph.
///
/// Requires a connected graph; it never inserts artificial cross-component bridges.
/// isomap_graph() inspects disconnected inputs before fitting. No feature standardization
/// or out-of-sample transform is performed. Finite graph distances need not be Euclidean:
/// negative centred-Gram eigenvalues are reported and zeroed only for coordinates.
/// Eigenvector signs and rotations in repeated eigenspaces are arbitrary; compare Gram
/// matrices or distances rather than individual coordinates across implementations.
///
/// Dense Floyd-Warshall and symmetric Householder/QL eigendecomposition cost O(n^3),
/// in addition to graph construction. Storage is O(n^2+n*p+n*k). Diagnostics are in-sample;
/// a high positive-inertia fraction does not establish true-geodesic recovery. Returned
/// references live as long as the model; const methods have no mutable shared caches.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,1,0});
/// d.add_column("y",std::vector<double>{0,0,1,1});
/// const dm::stats::Isomap model(d,{"x","y"},{2,2});
/// std::print("{:.1f}\n",model.goodness_of_fit()); // 1.0
/// @endcode
/// @par References
/// - J. B. Tenenbaum, V. de Silva and J. C. Langford (2000), A Global Geometric Framework
///   for Nonlinear Dimensionality Reduction, Science 290(5500), 2319-2323.
///   https://doi.org/10.1126/science.290.5500.2319
/// - W. S. Torgerson (1952), Multidimensional scaling: I. Theory and method.
///   https://doi.org/10.1007/BF02288916
/// @see isomap_graph(), MDS, LLE
class Isomap {
 public:
  /// @brief Fits immediately to selected raw numeric columns.
  /// @param data Training points; rows with any selected-feature null are omitted jointly.
  /// @param feature_columns Nonempty numeric feature names.
  /// @param options Axis count, neighbour count and input distance metric.
  /// @throws std::invalid_argument Invalid graph inputs, component count, disconnected graph,
  /// or shortest/squared distances outside representable range.
  /// @throws std::runtime_error Symmetric eigensolver fails to converge.
  /// @par Example
  /// See the class example; original labels can be aligned with kept_row_indices().
  Isomap(
      const dstruct::DataFrame&       data,
      const std::vector<std::string>& feature_columns,
      IsomapOptions                   options = {});

  /// @brief Selected raw feature names.
  /// @return Borrowed names in distance-coordinate order.
  [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
  /// @brief Number of retained observations.
  /// @return Row count of the fitted graph and embedding.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Number of requested axes, including zero axes beyond positive rank.
  /// @return Coordinate column count.
  [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }
  /// @brief Original zero-based rows retained after joint null omission.
  /// @return Borrowed increasing indices aligned to embedding().
  [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const {
    return graph_.kept_rows;
  }
  /// @brief Full descending centred-Gram spectrum, including negative values.
  /// @return Borrowed eigenvalues, length observations().
  [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }
  /// @brief Retained nonnegative eigenvalue sum divided by all positive eigenvalues.
  /// @return Positive-inertia fraction, or zero if all points coincide; not a distance-error score.
  [[nodiscard]] double goodness_of_fit() const { return goodness_of_fit_; }
  /// @brief Fitted coordinates in arbitrary orthogonal orientation.
  /// @return Borrowed n-by-n_components matrix; negative eigenmodes contribute zero coordinates.
  [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
  /// @brief Retained graph, directed neighbour lists and original distances.
  /// @return Borrowed connected graph result.
  [[nodiscard]] const IsomapGraph& neighborhood_graph() const { return graph_; }
  /// @brief All-pairs shortest-path distances; these estimate, rather than certify, manifold
  /// geodesics.
  /// @return Borrowed finite symmetric n-by-n matrix.
  [[nodiscard]] const linalg::DenseMatrix<double>& geodesic_distances() const { return geodesic_; }
  /// @brief Double-centred squared-geodesic Gram matrix B=-J*Dgeo^2*J/2.
  /// @return Borrowed symmetric n-by-n matrix, centred up to rounding.
  [[nodiscard]] const linalg::DenseMatrix<double>& gram_matrix() const { return gram_; }

  /// @brief Copies one fitted coordinate axis.
  /// @param index Zero-based axis index.
  /// @return Owned vector of retained-row coordinates.
  /// @throws std::out_of_range Axis index >=n_components().
  /// @par Example
  /// See spectrum(); `model.dimension(0)` is its first coordinate vector.
  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

  /// @brief Spectrum and cumulative positive-inertia shares for every mode.
  /// @return Owned
  /// dimension/eigenvalue/positive_share/cumulative_positive_share/significant_negative table.
  /// Negative flags use lambda<-1e-10*max(abs(lambda)).
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,1,0});
  /// d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::Isomap model(d,{"x","y"},{2,2});
  /// std::print("{} {}\n",model.spectrum().nrows(),model.dimension(0).size()); // 4 4
  /// @endcode
  [[nodiscard]] dstruct::DataFrame spectrum() const;

  /// @brief Spectral and geodesic-distance diagnostics for every fitted prefix, without refitting.
  /// @return Owned dimensions/gof_positive/negative_mass/relative_strain/geodesic_rmse/
  /// relative_geodesic_error/residual_variance table. negative_mass is sum(abs(negative
  /// eigenvalues))/sum(abs(all eigenvalues)); strain is normalized Gram Frobenius error.
  /// Distance sums use unordered pairs. residual_variance=1-Pearson(Dgeo,Dembedded)^2;
  /// it is NaN when either pair-distance vector is constant. Zero norm denominators give zero.
  /// O(n^2*n_components) work and O(n^2) workspace.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,1,0});
  /// d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::Isomap model(d,{"x","y"},{2,2});
  /// std::print("{:.1f}\n",model.dimension_diagnostics().double_at("negative_mass",1)); // 0.2
  /// @endcode
  [[nodiscard]] dstruct::DataFrame dimension_diagnostics() const;

  /// @brief Original-feature, graph-geodesic and embedded distances of every unordered pair.
  /// @param dimensions Prefix length; zero means all fitted axes.
  /// @return Owned row_i/row_j/ambient_distance/geodesic_distance/embedded_distance table,
  /// using original row indices, in lexicographic fitted-pair order. O(n^2*dimensions) work.
  /// @throws std::out_of_range Prefix exceeds n_components().
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,1,0});
  /// d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::Isomap model(d,{"x","y"},{2,2});
  /// std::print("{}\n",model.pair_distances().nrows()); // 6
  /// @endcode
  [[nodiscard]] dstruct::DataFrame pair_distances(std::size_t dimensions = 0) const;

  /// @brief Compares graph/embedded distances with independent Euclidean reference coordinates.
  /// @param reference Original-row-aligned frame, with exactly the training input row count.
  /// @param columns Nonempty finite numeric reference coordinates at every retained row.
  /// @return Owned n/pairs/geodesic_relative_error/embedding_relative_error/geodesic_correlation/
  /// embedding_correlation/max_edge_stretch row. max_edge_stretch is the largest reference
  /// Euclidean edge length divided by its input edge weight; it is meaningful when both
  /// use comparable units. Zero/zero edge ratios are zero; positive/zero is infinity.
  /// All fitted embedding axes are used, no alignment or rescaling.
  /// Relative errors divide by the norm of reference pair distances (zero/zero is zero,
  /// positive/zero is infinity); correlations are NaN when either distance vector is constant.
  /// @throws std::invalid_argument Row mismatch, missing/non-numeric columns, retained nulls,
  /// nonfinite values or reference distance overflow.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,1,0});
  /// d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::Isomap model(d,{"x","y"},{2,2});
  /// std::print("{:.0f}\n",model.reference_diagnostics(d,{"x","y"}).double_at("n",0)); // 4
  /// @endcode
  [[nodiscard]] dstruct::DataFrame reference_diagnostics(
      const dstruct::DataFrame&       reference,
      const std::vector<std::string>& columns) const;

  /// @brief Formats graph connectivity and the positive-inertia fraction.
  /// @return Owned text; no I/O occurs.
  /// @par Example
  /// See print_summary().
  [[nodiscard]] std::string summary() const;
  /// @brief Writes summary() to a caller-owned stream.
  /// @param os Destination stream, mutated by insertion.
  /// @par Example
  /// See the standard-output overload.
  void print_summary(std::ostream& os) const;
  /// @brief Writes summary() to standard output.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,2,4});
  /// const dm::stats::Isomap model(d,{"x"},{1,2});std::ostringstream out;model.print_summary(out);
  /// std::print("{}\n",out.str()==model.summary()); // true
  /// @endcode
  void print_summary() const;

  /// @brief Scatter plot of two fitted coordinate columns.
  /// @param dimension_x Zero-based horizontal fitted axis.
  /// @param dimension_y Zero-based vertical fitted axis; may equal dimension_x.
  /// @return Owned editable plot, without rendering.
  /// @throws std::out_of_range Invalid fitted axis.
  /// @par Example
  /// See the grouped overload; `model.plot_embedding()` omits labels.
  [[nodiscard]] graphics::RPlot plot_embedding(
      std::size_t dimension_x = 0,
      std::size_t dimension_y = 1) const;
  /// @brief Scatter plot with categorical labels aligned to retained observations.
  /// @param group_labels Exactly observations() labels, after applying kept_row_indices().
  /// @param dimension_x Zero-based horizontal axis.
  /// @param dimension_y Zero-based vertical axis.
  /// @return Owned editable plot; sorted labels receive recycled colours.
  /// @throws std::invalid_argument Label count differs from retained row count.
  /// @throws std::out_of_range Invalid fitted axis.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,1,0});
  /// d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::Isomap model(d,{"x","y"},{2,2});
  /// const auto plain=model.plot_embedding();
  /// model.plot_embedding(std::vector<std::string>{"A","A","B","B"}).save_svg("isomap_embedding.svg");
  /// @endcode
  [[nodiscard]] graphics::RPlot plot_embedding(
      const std::vector<std::string>& group_labels,
      std::size_t                     dimension_x = 0,
      std::size_t                     dimension_y = 1) const;

  /// @brief Plots embedded versus graph-geodesic distances, with an equality line.
  /// @param dimensions Prefix length; zero means all fitted axes.
  /// @return Owned editable plot of all unordered pairs.
  /// @throws std::out_of_range Prefix exceeds n_components().
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,1,0});
  /// d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::Isomap model(d,{"x","y"},{2,2});
  /// model.plot_shepard().save_svg("isomap_shepard.svg");
  /// @endcode
  [[nodiscard]] graphics::RPlot plot_shepard(std::size_t dimensions = 0) const;

 private:
  /// @brief Constructs the graph, shortest paths and centred eigensystem in place.
  /// @param data Training frame.
  void                        fit(const dstruct::DataFrame& data);
  std::vector<std::string>    feature_columns_; ///< Selected raw input names.
  IsomapOptions               options_;         ///< Validated fitting options.
  IsomapGraph                 graph_;           ///< Owned graph geometry and retained indices.
  linalg::DenseMatrix<double> geodesic_;        ///< Finite all-pairs graph distances.
  linalg::DenseMatrix<double> gram_;            ///< Double-centred squared-geodesic Gram matrix.
  std::vector<double>         eigenvalues_;     ///< Full descending signed spectrum.
  double                      goodness_of_fit_{0.0}; ///< Retained fraction of positive inertia.
  linalg::DenseMatrix<double> embedding_;            ///< Fitted coordinate matrix.
  std::size_t                 observations_{0};      ///< Retained row count.
  std::size_t source_rows_{0}; ///< Original input row count, for reference alignment.
};

} // namespace datamunge::stats
