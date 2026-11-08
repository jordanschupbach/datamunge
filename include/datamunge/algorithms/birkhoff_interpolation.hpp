#pragma once
/// @file birkhoff_interpolation.hpp
/// @brief Polynomial interpolation from possibly lacunary derivative constraints.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief One condition p^(order)(node)=value for ascending polynomial coefficients.
/// Derivative orders at a node may have gaps. Orders above the proposed degree
/// produce zero matrix rows and cannot establish uniqueness.
/// @par Example
/// @code
/// const dm::algorithms::BirkhoffConstraint curvature{1,2,30};
/// std::print("{} {}\n",curvature.order,curvature.value); // 2 30
/// @endcode
/// @see birkhoff_polynomial(), birkhoff_matrix()
struct BirkhoffConstraint {
  double node{0}; ///< Finite coordinate at which the derivative is prescribed.
  int order{0}; ///< Nonnegative derivative order; zero denotes a value.
  double value{0}; ///< Finite prescribed derivative value, in its derivative units.
};

/// @brief Derivative-monomial coefficient used by Birkhoff matrix assembly.
namespace detail {
/// @brief Evaluate the order-th derivative of x^k at node.
/// @param k Nonnegative monomial degree.
/// @param order Nonnegative derivative order.
/// @param node Finite coordinate.
/// @return Falling factorial k!/(k-order)! times node^(k-order), or zero when k<order.
/// @throws std::overflow_error If the entry is not a finite double.
double birkhoff_entry(int k,int order,double node);
}

/// @brief Assemble the square derivative-monomial system for Birkhoff interpolation.
/// N constraints give an N-by-N nested vector, with columns for 1,x,...,x^(N-1).
/// Empty input returns an empty matrix. Values are validated but omitted from the
/// matrix. O(N^2) storage and up to O(N^3) work for falling factorials; no mutation
/// or shared state.
/// @param constraints Finite node/value conditions with nonnegative derivative orders.
/// @return Owned matrix A such that A*c equals the constraint values.
/// @throws std::invalid_argument For negative orders or nonfinite node/value.
/// @throws std::length_error For a count exceeding supported integer indexing.
/// @throws std::overflow_error For a nonrepresentable matrix entry.
/// @par Example
/// @code
/// const auto matrix=dm::algorithms::birkhoff_matrix({{0,0,1},{1,1,2}});
/// std::print("{} {}\n",matrix[1][0],matrix[1][1]); // 0 1
/// @endcode
/// @par References
/// - G. D. Birkhoff (1906), General Mean Value and Remainder Theorems with
///   Applications to Mechanical Differentiation and Quadrature,
///   Transactions of the American Mathematical Society 7(1), 107–136,
///   https://www.jstor.org/stable/1986339.
/// @see birkhoff_polynomial(), BirkhoffConstraint
std::vector<std::vector<double>> birkhoff_matrix(const std::vector<BirkhoffConstraint>& constraints);

/// @brief Find a polynomial of degree at most N-1 satisfying N derivative conditions.
/// Derivative orders may have gaps. Equilibrates rows and columns of
/// birkhoff_matrix(), then applies pivoted Gauss–Jordan elimination. Pivots at
/// most 64*N*epsilon on the equilibrated matrix are treated as numerically
/// singular. Empty input or numerical singularity returns empty, whether a
/// singular system has no solution or infinitely many. A nonempty result is not
/// a proof of mathematical poisedness or good conditioning. Coefficients are
/// restored to the original monomial units, lowest degree first. O(N^3) time,
/// O(N^2) storage, no mutation or shared state.
/// @param constraints Finite conditions with nonnegative derivative orders.
/// @return Owned ascending coefficients, or empty if numerical uniqueness is unresolved.
/// @throws std::invalid_argument For invalid constraints, as in birkhoff_matrix().
/// @throws std::length_error For an unsupported constraint count.
/// @throws std::overflow_error For nonrepresentable entries, arithmetic or coefficients.
/// @par Example
/// @code
/// const auto c=dm::algorithms::birkhoff_polynomial({{-1,0,-2},{-1,2,-18},{1,0,10},{1,2,30}});
/// std::print("{:.0f} {:.0f}\n",c[0],c[3]); // 1 4
/// @endcode
/// @par References
/// - G. D. Birkhoff (1906), General Mean Value and Remainder Theorems with
///   Applications to Mechanical Differentiation and Quadrature,
///   https://www.jstor.org/stable/1986339. Scaling and numerical pivot tolerance
///   are implementation choices, separate from mathematical poisedness.
/// @see birkhoff_matrix(), birkhoff_eval(), birkhoff_eval_derivative()
std::vector<double> birkhoff_polynomial(const std::vector<BirkhoffConstraint>& constraints);

/// @brief Evaluate ascending polynomial coefficients with Horner's rule.
/// Empty coefficients represent zero. O(N) work and constant auxiliary storage;
/// no mutation or shared state. Finite input can still overflow.
/// @param coeffs Finite coefficients c0,c1,... in increasing degree order.
/// @param x Finite coordinate, including outside interpolation nodes.
/// @return Polynomial value, or zero for empty coefficients.
/// @throws std::invalid_argument For nonfinite coefficients or x.
/// @throws std::overflow_error If Horner arithmetic becomes nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::birkhoff_eval({1,2,3,4},1)); // 10
/// @endcode
/// @see birkhoff_eval_derivative(), birkhoff_polynomial()
double birkhoff_eval(const std::vector<double>& coeffs,double x);

/// @brief Evaluate a derivative directly from ascending polynomial coefficients.
/// Uses Horner evaluation of derivative coefficients. Order zero gives the
/// polynomial value. Orders above the degree and empty vectors give zero after
/// validation. O(N*(order+1)) work and constant auxiliary storage; no mutation
/// or shared state. Results retain the corresponding derivative units.
/// @param coeffs Finite ascending coefficients.
/// @param x Finite coordinate.
/// @param order Nonnegative derivative order.
/// @return The requested derivative value.
/// @throws std::invalid_argument For negative order or nonfinite inputs.
/// @throws std::overflow_error For nonfinite derivative coefficients or arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::birkhoff_eval_derivative({1,2,3,4},1,2)); // 30
/// @endcode
/// @par References
/// - G. D. Birkhoff (1906), General Mean Value and Remainder Theorems with
///   Applications to Mechanical Differentiation and Quadrature,
///   https://www.jstor.org/stable/1986339.
/// @see birkhoff_eval(), BirkhoffConstraint
double birkhoff_eval_derivative(const std::vector<double>& coeffs,double x,int order);
} // namespace datamunge::algorithms
