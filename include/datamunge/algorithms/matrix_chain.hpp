#pragma once

#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief The result of matrix-chain ordering: the minimum number of scalar multiplications
///        @ref cost and an optimal fully-parenthesized product @ref parenthesization (e.g.
///        @c "((A1(A2A3))A4)").
struct MatrixChainResult {
    long long   cost{0};          ///< minimum scalar multiplications over all parenthesizations.
    std::string parenthesization; ///< one optimal parenthesization, matrices named A1..An.
};

/// @brief *Matrix-chain multiplication ordering* by dynamic programming. Multiplying a chain
///        @c A1·A2·…·An is associative, but the *order* dramatically changes the cost: multiplying a
///        @c p×q by a @c q×r matrix costs @c p·q·r scalar multiplications, so a good parenthesization
///        can be orders of magnitude cheaper. Given the dimension sequence @p dims (matrix @c i is
///        @c dims[i-1]×dims[i]), the DP @c m[i][j] = min over split @c k of
///        @c m[i][k]+m[k+1][j]+dims[i-1]·dims[k]·dims[j] finds the cheapest order in @c O(n^3) time,
///        versus the exponential (Catalan) number of parenthesizations.
///
/// @param dims the @c n+1 dimensions of the @c n matrices (must have size >= 2).
/// @return the minimum @ref MatrixChainResult::cost and an optimal @ref MatrixChainResult::parenthesization.
MatrixChainResult matrix_chain_order(const std::vector<int>& dims);

} // namespace datamunge::algorithms
