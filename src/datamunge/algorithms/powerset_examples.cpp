/// @file powerset_examples.cpp
/// @brief Original binary suffix-position NFA generator with checked complete geometry.
#include <datamunge/algorithms/powerset_examples.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
NFA nth_from_end_nfa(int n, std::size_t max_states) {
  if (n < 1 || n == std::numeric_limits<int>::max())
    throw std::invalid_argument("suffix NFA positive representable n");
  auto count = std::size_t(n) + 1;
  if (max_states && count > max_states)
    throw std::length_error("suffix NFA state cap");
  NFA a;
  a.num_states  = n + 1;
  a.num_symbols = 2;
  a.delta.assign(count, std::vector<std::vector<int>>(2));
  a.delta[0][0] = {0};
  a.delta[0][1] = {0, 1};
  for (int i = 1; i < n; ++i) {
    a.delta[i][0] = {i + 1};
    a.delta[i][1] = {i + 1};
  }
  a.start_states = {0};
  a.accept.assign(count, 0);
  a.accept[n] = 1;
  return a;
}
} // namespace datamunge::algorithms
