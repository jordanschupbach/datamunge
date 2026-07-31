#pragma once

/// \file summed_area_table.hpp
/// \brief The summed-area table (integral image) for O(1) rectangle sums.
///
/// A *summed-area table* (Crow, 1984; rediscovered in vision as the *integral image*,
/// Viola-Jones 2001) precomputes, for every cell \f$(r,c)\f$, the sum of all grid values in
/// the rectangle from the origin to that cell:
/// \f[
///   S(r,c) = \sum_{i\le r,\ j\le c} A(i,j).
/// \f]
/// It is built in one pass with the recurrence
/// \f$S(r,c) = A(r,c) + S(r-1,c) + S(r,c-1) - S(r-1,c-1)\f$. Afterwards the sum over *any*
/// axis-aligned rectangle is four table lookups by inclusion-exclusion:
/// \f[
///   \sum_{r_0\le r\le r_1,\ c_0\le c\le c_1} A = S(r_1,c_1) - S(r_0-1,c_1) - S(r_1,c_0-1)
///     + S(r_0-1,c_0-1),
/// \f]
/// independent of the rectangle's size. This turns box filters, Haar-feature evaluation, and
/// local-average computations from \f$O(\text{area})\f$ into \f$O(1)\f$ per query, which is
/// what made real-time face detection feasible.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief Precomputed prefix-sum table over a 2D grid, giving O(1) rectangle-sum queries.
class SummedAreaTable {
   public:
    /// \brief Build the table from a row-major grid of \p rows x \p cols values.
    SummedAreaTable(const std::vector<std::vector<double>>& grid)
        : rows_(grid.size()), cols_(grid.empty() ? 0 : grid[0].size()) {
        // (rows+1) x (cols+1) with a zero border to avoid bounds checks in queries.
        s_.assign(rows_ + 1, std::vector<double>(cols_ + 1, 0.0));
        for (std::size_t r = 0; r < rows_; ++r)
            for (std::size_t c = 0; c < cols_; ++c)
                s_[r + 1][c + 1] =
                    grid[r][c] + s_[r][c + 1] + s_[r + 1][c] - s_[r][c];
    }

    /// \brief Sum over the inclusive rectangle rows [r0, r1], cols [c0, c1] in O(1).
    double rect_sum(std::size_t r0, std::size_t c0, std::size_t r1, std::size_t c1) const {
        // Uses the 1-based, zero-bordered table: indices shift by +1.
        return s_[r1 + 1][c1 + 1] - s_[r0][c1 + 1] - s_[r1 + 1][c0] + s_[r0][c0];
    }

    /// \brief Mean value over the inclusive rectangle (box filter) in O(1).
    double rect_mean(std::size_t r0, std::size_t c0, std::size_t r1, std::size_t c1) const {
        double n = static_cast<double>((r1 - r0 + 1) * (c1 - c0 + 1));
        return rect_sum(r0, c0, r1, c1) / n;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

   private:
    std::size_t rows_, cols_;
    std::vector<std::vector<double>> s_;  // (rows+1) x (cols+1) prefix sums
};

}  // namespace datamunge::algorithms
