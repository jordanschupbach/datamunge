#pragma once
/// @file binary_splitting.hpp
/// @brief Exact uint64 exponential-series fractions with a balanced product tree.
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {
/// @brief Exact, possibly unreduced exponential-series fraction and its double conversion.
/// num and den are owned unsigned integers; value rounds the ratio after converting
/// both integers to double. Equality to a rounded exp(1) is not exact equality to e.
/// @par Example
/// @code
/// const auto fraction=dm::algorithms::binary_splitting_e(4);
/// std::print("{}/{}\n",fraction.num,fraction.den); // 16/6
/// @endcode
/// @see binary_splitting_e(), BinSplitTrace
struct BinSplitResult {
  std::uint64_t num{0}; ///< Numerator of the exact finite series; not automatically reduced.
  std::uint64_t den{1}; ///< Positive denominator, equal to (terms-1)! for a computed result.
  double value{0}; ///< Floating conversion double(num)/double(den), with rounding.
};

/// @brief One owned node of the balanced exponential-series product tree.
/// For q(0)=1 and q(k)=k otherwise, den is the product of q(k) over [begin,end).
/// num/den equals the sum over k in that interval of 1/product(q(j),j=begin..k).
/// This is a normalised subtree, not generally the original sum of 1/k! there.
/// Children are indices into the owning trace's postorder node vector; leaves use -1.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::binary_splitting_e_trace(4);
/// const auto& node=trace.nodes.back();
/// std::print("{} {}\n",node.begin,node.end); // 0 4
/// @endcode
/// @see BinSplitTrace
struct BinSplitNode {
  int begin{0}; ///< Inclusive factor index.
  int end{1}; ///< Exclusive factor index.
  std::uint64_t num{1}; ///< Exact normalised subtree numerator.
  std::uint64_t den{1}; ///< Exact subtree factor product.
  int left{-1}; ///< Left child index, or -1 for a leaf.
  int right{-1}; ///< Right child index, or -1 for a leaf.
};

/// @brief Exact result and owned postorder tree from the canonical binary-splitting routine.
/// The last node is the root. N effective terms give N leaves and 2N-1 nodes.
/// No pointers into inputs or shared mutable state survive the call.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::binary_splitting_e_trace(4);
/// std::print("{} {}\n",trace.terms,trace.nodes.size()); // 4 7
/// @endcode
/// @see binary_splitting_e_trace()
struct BinSplitTrace {
  BinSplitResult result; ///< Same exact fraction and conversion as the unrecorded call.
  int terms{1}; ///< Effective number of terms after the legacy lower clamp to one.
  std::vector<BinSplitNode> nodes; ///< Leaves and merges in postorder, with root last.
};

/// @brief Evaluate sum(k=0..N-1,1/k!) exactly as an unreduced uint64 fraction.
/// N is max(1,terms), preserving the historical lower clamp. N must be at most 21:
/// the root numerator and denominator both fit uint64 through N=21; the raw
/// denominator 21! for N=22 does not. The count is checked before recursion.
/// Every merge uses P=P_left*Q_right+P_right and Q=Q_left*Q_right on normalised
/// subtrees. Only the root represents the original factorial series. No reduction
/// or multiprecision arithmetic is performed. O(N) fixed-word operations and
/// O(log(N+1)) stack storage; no allocation, mutation or shared state.
/// @param terms Requested count of terms, clamped to one when less than one; maximum 21.
/// @return Exact finite-series numerator/denominator and rounded double conversion.
/// @throws std::overflow_error If terms is greater than 21, before any arithmetic or allocation.
/// @par Example
/// @code
/// const auto fraction=dm::algorithms::binary_splitting_e(3);
/// std::print("{}/{} = {}\n",fraction.num,fraction.den,fraction.value); // 5/2 = 2.5
/// @endcode
/// @par References
/// - R. P. Brent and P. Zimmermann, Modern Computer Arithmetic, section 4.9,
///   https://members.loria.fr/PZimmermann/mca/pub226.html (general multiprecision
///   method; this implementation is restricted to uint64 and the exponential series).
/// @see binary_splitting_e_trace(), BinSplitResult
BinSplitResult binary_splitting_e(int terms);

/// @brief Record the same balanced exponential-series tree as binary_splitting_e().
/// Uses identical fractions and the same [1,21] effective term domain. Recording
/// adds O(N) owned storage; the unrecorded routine retains only its recursion stack.
/// @param terms Requested number of terms, with values below one clamped to one.
/// @return Owned fraction and postorder nodes, with the root last.
/// @throws std::overflow_error If terms is greater than 21.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::binary_splitting_e_trace(4);
/// std::print("{}/{}\n",trace.nodes.back().num,trace.nodes.back().den); // 16/6
/// @endcode
/// @par References
/// - R. P. Brent and P. Zimmermann, Modern Computer Arithmetic, section 4.9,
///   https://members.loria.fr/PZimmermann/mca/pub226.html.
/// @see binary_splitting_e(), BinSplitNode
BinSplitTrace binary_splitting_e_trace(int terms);
} // namespace datamunge::algorithms
