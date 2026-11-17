/// @file tree_sort.cpp
/// @brief Canonical producing kernels and native report diagnostics.
#include <datamunge/algorithms/tree_sort.hpp>
namespace datamunge::algorithms {
/// @brief Implement the checked canonical tree_sort_work producer; contracts are declared in the matching header.
TreeSortWork tree_sort_work(const std::vector<int>& input, std::size_t cap) {
  if (input.size() > 1000000)
    throw std::length_error("tree_sort: key limit");
  TreeSortWork r;
  r.values = input;
  r.left.assign(input.size(), -1);
  r.right = r.left;
  r.depth.resize(input.size());
  for (std::size_t i = 1; i < input.size(); ++i) {
    std::size_t u = 0;
    for (;;) {
      if (r.comparisons == cap)
        throw std::length_error("tree_sort: comparison limit");
      ++r.comparisons;
      bool left  = input[i] < input[u];
      int  child = left ? r.left[u] : r.right[u];
      if (child < 0) {
        (left ? r.left[u] : r.right[u]) = static_cast<int>(i);
        r.depth[i]                      = r.depth[u] + 1;
        break;
      }
      u = child;
    }
  }
  std::vector<int> stack;
  int              u = input.empty() ? -1 : 0;
  while (u >= 0 || !stack.empty()) {
    while (u >= 0) {
      stack.push_back(u);
      u = r.left[u];
    }
    u = stack.back();
    stack.pop_back();
    r.inorder.push_back(u);
    r.sorted.push_back(r.values[u]);
    u = r.right[u];
  }
  return r;
}
/// @brief Implement the checked canonical tree_sort producer; contracts are declared in the matching header.
std::vector<int> tree_sort(std::vector<int> a) {
  return tree_sort_work(a).sorted;
}
} // namespace datamunge::algorithms
