#pragma once

/// @file embedding_quality.hpp
/// @brief Rank-based measures of how well a low-dimensional embedding keeps neighbourhoods:
///        trustworthiness and continuity (Venna and Kaski 2001).
///
/// Both compare the \f$k\f$ nearest neighbours of each point in the original space and in the
/// embedding, and lie in [0, 1], with 1 meaning every neighbourhood is kept exactly.
/// *Trustworthiness* penalises points that are neighbours in the embedding but far apart in the
/// data (false neighbours, as when a fold is flattened onto itself); *continuity* penalises
/// data neighbours that the embedding pulls apart (tears). Distances are Euclidean; ties in
/// distance are broken by row order, as in scikit-learn.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Trustworthiness of an embedding (Venna and Kaski 2001), as scikit-learn's
///        `sklearn.manifold.trustworthiness`.
///
/// \f$T(k)=1-\frac{2}{nk(2n-3k-1)}\sum_{i=1}^n\sum_{j\in U_k(i)}\big(r(i,j)-k\big)\f$, where
/// \f$U_k(i)\f$ are the points among the \f$k\f$ nearest neighbours of \f$i\f$ in the embedding
/// but not in the data, and \f$r(i,j)\f$ is the rank of \f$j\f$ by distance from \f$i\f$ in the
/// data (nearest = 1). Costs \f$O(n^2\log n)\f$.
///
/// @param data  Original points, one per row.
/// @param embedding  Embedded points, one per row, in the same order.
/// @param k  Neighbourhood size; 1 <= k < n/2.
/// @return \f$T(k)\in[0,1]\f$.
/// @throws std::invalid_argument  If the row counts differ or @p k is out of range.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> X(4, 1, {0.0, 1.0, 2.0, 10.0});
/// const dm::linalg::DenseMatrix<double> Y(4, 1, {0.0, 2.0, 1.0, 10.0});
/// std::print("{:.3f}\n", dm::stats::trustworthiness(X, Y, 1));  // 0.500
/// @endcode
///
/// @par References
/// - J. Venna and S. Kaski (2001). "Neighborhood preservation in nonlinear projection methods:
///   an experimental study". ICANN 2001, Lecture Notes in Computer Science 2130, 485-491.
///   doi:10.1007/3-540-44668-0_68
/// - scikit-learn `trustworthiness`:
///   https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
///
/// @see continuity()
[[nodiscard]] double trustworthiness(const linalg::DenseMatrix<double>& data,
                                     const linalg::DenseMatrix<double>& embedding, std::size_t k);

/// @brief Continuity of an embedding (Venna and Kaski 2001): trustworthiness() with the roles of
///        the data and the embedding swapped.
///
/// It penalises the data's \f$k\f$ nearest neighbours that are not among the embedding's, by how
/// far down the embedding's ranking they fall.
///
/// @param data  Original points, one per row.
/// @param embedding  Embedded points, one per row, in the same order.
/// @param k  Neighbourhood size; 1 <= k < n/2.
/// @return \f$C(k)\in[0,1]\f$.
/// @throws std::invalid_argument  As trustworthiness().
///
/// @par Example
/// See trustworthiness().
///
/// @par References
/// - J. Venna and S. Kaski (2001). "Neighborhood preservation in nonlinear projection methods:
///   an experimental study". ICANN 2001, LNCS 2130, 485-491. doi:10.1007/3-540-44668-0_68
///
/// @see trustworthiness()
[[nodiscard]] double continuity(const linalg::DenseMatrix<double>& data,
                                const linalg::DenseMatrix<double>& embedding, std::size_t k);

/// @brief trustworthiness() of an embedding of numeric columns of a DataFrame.
///
/// Row \f$i\f$ of @p embedding must be row \f$i\f$ of @p data, so @p data must have no nulls
/// in @p columns (drop them first, or select the rows of the model's `kept_row_indices()`).
///
/// @param data  Original data.
/// @param columns  Numeric columns spanning the original space.
/// @param embedding  Embedded points, one per row of @p data.
/// @param k  Neighbourhood size; 1 <= k < n/2.
/// @return \f$T(k)\in[0,1]\f$.
/// @throws std::invalid_argument  If a column is missing, not numeric or has a null, or as
///         trustworthiness().
///
/// @par Example
/// @code
/// const auto roll = dm::datasets::swiss_roll(300, 0.0, 1);
/// const dm::stats::LLE lle(roll, {"x", "y", "z"}, {.n_neighbors = 12});
/// std::print("{:.3f}\n", dm::stats::trustworthiness(roll, {"x", "y", "z"}, lle.embedding(), 10));  // 0.847
/// @endcode
///
/// @see continuity(const dstruct::DataFrame&, const std::vector<std::string>&, const linalg::DenseMatrix<double>&, std::size_t)
[[nodiscard]] double trustworthiness(const dstruct::DataFrame& data, const std::vector<std::string>& columns,
                                     const linalg::DenseMatrix<double>& embedding, std::size_t k);

/// @brief continuity() of an embedding of numeric columns of a DataFrame.
///
/// @param data  Original data, with no nulls in @p columns.
/// @param columns  Numeric columns spanning the original space.
/// @param embedding  Embedded points, one per row of @p data.
/// @param k  Neighbourhood size; 1 <= k < n/2.
/// @return \f$C(k)\in[0,1]\f$.
/// @throws std::invalid_argument  As the DataFrame overload of trustworthiness().
///
/// @par Example
/// @code
/// const auto roll = dm::datasets::swiss_roll(300, 0.0, 1);
/// const dm::stats::LLE lle(roll, {"x", "y", "z"}, {.n_neighbors = 12});
/// std::print("{:.3f}\n", dm::stats::continuity(roll, {"x", "y", "z"}, lle.embedding(), 10));  // 0.971
/// @endcode
[[nodiscard]] double continuity(const dstruct::DataFrame& data, const std::vector<std::string>& columns,
                                const linalg::DenseMatrix<double>& embedding, std::size_t k);

} // namespace datamunge::stats
