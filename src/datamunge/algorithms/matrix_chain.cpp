#include <datamunge/algorithms/matrix_chain.hpp>

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Recursively build the parenthesization string from the split table s (1-indexed matrices).
std::string build(const std::vector<std::vector<std::size_t>>& s, std::size_t i, std::size_t j) {
    if (i == j) return "A" + std::to_string(i);
    const std::size_t k = s[i][j];
    return "(" + build(s, i, k) + build(s, k + 1, j) + ")";
}

} // namespace

MatrixChainResult matrix_chain_order(const std::vector<int>& dims) {
    const std::size_t n = dims.size() - 1; // number of matrices
    MatrixChainResult result;
    if (n == 0) return result;
    if (n == 1) {
        result.parenthesization = "A1";
        return result;
    }

    // m[i][j] = min cost to multiply A_i..A_j (1-indexed); s[i][j] = optimal split point.
    std::vector<std::vector<long long>>   m(n + 1, std::vector<long long>(n + 1, 0));
    std::vector<std::vector<std::size_t>> s(n + 1, std::vector<std::size_t>(n + 1, 0));

    for (std::size_t len = 2; len <= n; ++len) {              // chain length
        for (std::size_t i = 1; i + len - 1 <= n; ++i) {
            const std::size_t j = i + len - 1;
            m[i][j]             = std::numeric_limits<long long>::max();
            for (std::size_t k = i; k < j; ++k) {
                const long long cost = m[i][k] + m[k + 1][j] +
                                       static_cast<long long>(dims[i - 1]) * dims[k] * dims[j];
                if (cost < m[i][j]) {
                    m[i][j] = cost;
                    s[i][j] = k;
                }
            }
        }
    }

    result.cost            = m[1][n];
    result.parenthesization = build(s, 1, n);
    return result;
}

} // namespace datamunge::algorithms
