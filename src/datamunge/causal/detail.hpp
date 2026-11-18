#pragma once

/// @file detail.hpp
/// @brief Internal helpers shared by causal graph and numerical methods.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

/// @brief Private causal fitting and search primitives.
namespace datamunge::causal::detail {

/// @brief Enumerates fixed-size subsets in lexicographic position order.
/// @tparam Visit Callable returning true to stop the search.
/// @param nodes Ordered elements available for selection.
/// @param size Desired subset size.
/// @param visit Visitor for each complete subset.
/// @return True when the visitor stops the search, false after exhaustion.
template<class Visit>
bool subsets(const std::vector<std::size_t>& nodes, std::size_t size, Visit visit) {
    if (size > nodes.size()) return false;
    std::vector<std::size_t> index(size), selected(size);
    std::iota(index.begin(), index.end(), 0);
    for (;;) {
        for (std::size_t i = 0; i < size; ++i) selected[i] = nodes[index[i]];
        if (visit(selected)) return true;
        auto j = size;
        while (j > 0 && index[j - 1] == nodes.size() - size + j - 1) --j;
        if (j == 0) return false;
        ++index[j - 1];
        for (std::size_t i = j; i < size; ++i) index[i] = index[i - 1] + 1;
    }
}

/// @brief Rejects nonfinite matrix entries.
/// @param matrix Numeric data to validate.
/// @throws std::invalid_argument If any entry is nonfinite.
void finite_matrix(const linalg::DenseMatrix<double>& matrix);

} // namespace datamunge::causal::detail
