#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::geometry {

struct DTWResult {
    /// @brief The cumulative warping cost -- sum of |a[i] - b[j]| along the optimal
    ///        alignment path.
    double distance{0.0};
    /// @brief The optimal alignment: path_a[k]/path_b[k] give the index into a/b at step k of
    ///        the path, from (0, 0) to (a.size()-1, b.size()-1), monotonically non-decreasing
    ///        in both. Two parallel int vectors rather than a vector<pair<size_t,size_t>> (or
    ///        even vector<size_t>) -- a deliberately binding-friendly shape reusing the
    ///        already-proven plain vector<int> machinery. vector<size_t> as either a struct
    ///        field or nested inside another template has repeatedly hit real SWIG codegen
    ///        conflicts in this codebase (duplicate scalar-traits definitions); plain `int`
    ///        sidesteps that category of bug entirely, and is more than large enough for any
    ///        realistic sequence length here.
    std::vector<int> path_a;
    std::vector<int> path_b;
};

/// @brief Dynamic Time Warping distance between two (possibly different-length) scalar
///        sequences: the minimum-cost monotone alignment between them under absolute
///        difference as the per-step cost, found via the classic O(n*m) dynamic program. This
///        is the unrestricted (no Sakoe-Chiba band or other windowing) version -- fine for the
///        modest sequence lengths this module targets; a banded variant would be needed to
///        scale to long sequences.
[[nodiscard]] inline DTWResult dynamic_time_warping(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    if (n == 0 || m == 0) {
        throw std::invalid_argument("dynamic_time_warping: both sequences must be non-empty");
    }

    std::vector<std::vector<double>> dtw(n + 1, std::vector<double>(m + 1, std::numeric_limits<double>::infinity()));
    dtw[0][0] = 0.0;
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= m; ++j) {
            const double cost = std::abs(a[i - 1] - b[j - 1]);
            dtw[i][j] = cost + std::min({dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1]});
        }
    }

    DTWResult result;
    result.distance = dtw[n][m];

    std::size_t i = n, j = m;
    while (i > 0 || j > 0) {
        result.path_a.push_back(static_cast<int>(i > 0 ? i - 1 : 0));
        result.path_b.push_back(static_cast<int>(j > 0 ? j - 1 : 0));
        if (i == 0) {
            --j;
        } else if (j == 0) {
            --i;
        } else {
            const double diag = dtw[i - 1][j - 1];
            const double up = dtw[i - 1][j];
            const double left = dtw[i][j - 1];
            if (diag <= up && diag <= left) {
                --i;
                --j;
            } else if (up <= left) {
                --i;
            } else {
                --j;
            }
        }
    }
    std::reverse(result.path_a.begin(), result.path_a.end());
    std::reverse(result.path_b.begin(), result.path_b.end());

    return result;
}

} // namespace datamunge::geometry
