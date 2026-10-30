/// @file embedding_quality.cpp
/// @brief Trustworthiness and continuity of embeddings.

#include <datamunge/stats/embedding_quality.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Helpers private to this file.
namespace {

/// @brief Squared Euclidean distances from row @p i of @p A to every row.
/// @param A  Points, one per row.
/// @param i  Reference row.
/// @return n squared distances.
std::vector<double> squared_distances(const linalg::DenseMatrix<double>& A, std::size_t i) {
    std::vector<double> out(A.rows(), 0.0);
    for (std::size_t j = 0; j < A.rows(); ++j) {
        double s = 0.0;
        for (std::size_t c = 0; c < A.cols(); ++c) {
            const double d = A(i, c) - A(j, c);
            s += d * d;
        }
        out[j] = s;
    }
    return out;
}

/// @brief Other points ordered by distance from @p i (ties by index), excluding @p i itself.
/// @param A  Points, one per row.
/// @param i  Reference row.
/// @return n - 1 row indices, nearest first.
std::vector<std::size_t> ordered_neighbours(const linalg::DenseMatrix<double>& A, std::size_t i) {
    const auto               d = squared_distances(A, i);
    std::vector<std::size_t> idx(A.rows());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    idx.erase(idx.begin() + static_cast<std::ptrdiff_t>(i));
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return d[a] < d[b]; });
    return idx;
}

/// @brief The Venna-Kaski penalty: neighbours of each point in @p near but not in @p far,
///        weighted by their rank in @p far.
/// @param far  Space whose ranks are used.
/// @param near  Space whose k nearest neighbours are checked.
/// @param k  Neighbourhood size.
/// @return \f$1-\frac{2}{nk(2n-3k-1)}\sum(r-k)\f$.
double rank_penalty(const linalg::DenseMatrix<double>& far, const linalg::DenseMatrix<double>& near, std::size_t k) {
    const std::size_t n = far.rows();
    if (near.rows() != n) throw std::invalid_argument("trustworthiness: data and embedding have different row counts");
    if (k < 1 || 2 * k >= n) throw std::invalid_argument("trustworthiness: k must satisfy 1 <= k < n/2");
    double                   penalty = 0.0;
    std::vector<std::size_t> rank(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto far_order = ordered_neighbours(far, i);
        for (std::size_t r = 0; r < far_order.size(); ++r) rank[far_order[r]] = r + 1;
        const auto near_order = ordered_neighbours(near, i);
        for (std::size_t a = 0; a < k; ++a) {
            const std::size_t j = near_order[a];
            if (rank[j] > k) penalty += static_cast<double>(rank[j] - k);
        }
    }
    const double nd = static_cast<double>(n), kd = static_cast<double>(k);
    return 1.0 - 2.0 / (nd * kd * (2.0 * nd - 3.0 * kd - 1.0)) * penalty;
}

/// @brief The numeric @p columns of @p data as a matrix, one row per row.
/// @param data  Data frame.
/// @param columns  Column names.
/// @return n x columns.size() matrix.
/// @throws std::invalid_argument  If a column is missing, not numeric or has a null.
linalg::DenseMatrix<double> column_matrix(const dstruct::DataFrame& data, const std::vector<std::string>& columns) {
    if (columns.empty()) throw std::invalid_argument("trustworthiness: no columns");
    linalg::DenseMatrix<double> X(data.nrows(), columns.size(), 0.0);
    for (std::size_t j = 0; j < columns.size(); ++j) {
        if (!data.has_column(columns[j])) throw std::invalid_argument("trustworthiness: column '" + columns[j] + "' not found");
        if (data.column_type(columns[j]) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("trustworthiness: column '" + columns[j] + "' must be numeric");
        for (std::size_t i = 0; i < data.nrows(); ++i) {
            if (data.is_null(columns[j], i))
                throw std::invalid_argument("trustworthiness: column '" + columns[j] + "' has a null");
            X(i, j) = data.double_at(columns[j], i);
        }
    }
    return X;
}

} // namespace

double trustworthiness(const dstruct::DataFrame& data, const std::vector<std::string>& columns,
                       const linalg::DenseMatrix<double>& embedding, std::size_t k) {
    return rank_penalty(column_matrix(data, columns), embedding, k);
}

double continuity(const dstruct::DataFrame& data, const std::vector<std::string>& columns,
                  const linalg::DenseMatrix<double>& embedding, std::size_t k) {
    return rank_penalty(embedding, column_matrix(data, columns), k);
}

double trustworthiness(const linalg::DenseMatrix<double>& data, const linalg::DenseMatrix<double>& embedding,
                       std::size_t k) {
    return rank_penalty(data, embedding, k);
}

double continuity(const linalg::DenseMatrix<double>& data, const linalg::DenseMatrix<double>& embedding,
                  std::size_t k) {
    return rank_penalty(embedding, data, k);
}

} // namespace datamunge::stats
