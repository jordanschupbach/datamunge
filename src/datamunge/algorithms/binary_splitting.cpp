/// @file binary_splitting.cpp
/// @brief Balanced exact exponential-series evaluation with an explicit word-size limit.
#include <datamunge/algorithms/binary_splitting.hpp>
#include <algorithm>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private product-tree recursion and checked effective term count.
namespace {
/// @brief Largest term count whose unreduced root numerator and factorial denominator fit uint64.
constexpr int max_terms=21;

/// @brief Check the upper arithmetic limit, retaining the legacy lower clamp.
/// @param terms Requested count.
/// @return Effective count in [1,21].
/// @throws std::overflow_error If the requested count exceeds 21.
int effective_terms(int terms) {
  if (terms>max_terms) throw std::overflow_error("binary_splitting_e: at most 21 terms fit uint64");
  return std::max(1,terms);
}

/// @brief Compute a normalised factor-tree node and optionally append its postorder record.
/// @param a Inclusive factor index in the validated effective range.
/// @param b Exclusive factor index, greater than a and at most 21.
/// @param nodes Nullable borrowed postorder output vector.
/// @return Exact node value; products fit by the validated root-domain bound.
BinSplitNode split(int a,int b,std::vector<BinSplitNode>* nodes) {
  BinSplitNode node{a,b,1,static_cast<std::uint64_t>(a==0?1:a),-1,-1};
  if (b-a>1) {
    const int middle=a+(b-a)/2;
    const auto left=split(a,middle,nodes);
    node.left=nodes?static_cast<int>(nodes->size())-1:-1;
    const auto right=split(middle,b,nodes);
    node.right=nodes?static_cast<int>(nodes->size())-1:-1;
    node.num=left.num*right.den+right.num;
    node.den=left.den*right.den;
  }
  if (nodes) nodes->push_back(node);
  return node;
}
} // namespace

BinSplitResult binary_splitting_e(int terms) {
  const auto root=split(0,effective_terms(terms),nullptr);
  return {root.num,root.den,static_cast<double>(root.num)/static_cast<double>(root.den)};
}

BinSplitTrace binary_splitting_e_trace(int terms) {
  BinSplitTrace trace;
  trace.terms=effective_terms(terms);
  trace.nodes.reserve(2*trace.terms-1);
  const auto root=split(0,trace.terms,&trace.nodes);
  trace.result={root.num,root.den,static_cast<double>(root.num)/static_cast<double>(root.den)};
  return trace;
}
} // namespace datamunge::algorithms
