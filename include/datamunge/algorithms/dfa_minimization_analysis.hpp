#pragma once
/// @file dfa_minimization_analysis.hpp
/// @brief Public six-state minimization fixture and bounded unary threshold languages.
#include <datamunge/algorithms/dfa_minimization.hpp>
namespace datamunge::algorithms {
/// @brief Return the original six-state, two-symbol Hopcroft report fixture.
/// Accepting states 2,3,4 have identical future behavior; states 0,1 are equivalent
/// and state 5 is a rejecting sink. All six input states are reachable from 0.
/// The reachable minimal quotient has three states. Fixed owned storage; no
/// shared/input state, randomness or retained references.
/// @return Owned complete DFA over symbols 0,1.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto d=dm::algorithms::hopcroft_example_dfa();
/// std::print("{} {}\n",d.num_states,dm::algorithms::hopcroft_minimize(d).num_states); // 6 3
/// @endcode
/// @see hopcroft_minimize_trace()
inline DFA hopcroft_example_dfa() {
  return {6, 2, {{1, 2}, {0, 3}, {4, 5}, {4, 5}, {4, 5}, {5, 5}}, 0, {0, 0, 1, 1, 1, 0}};
}
/// @brief Build the minimal unary DFA accepting words of length at least a threshold.
/// States 0..minimum_length advance on symbol0, and the last state accepts and
/// self-loops. Every state is reachable; any pair is distinguished by reaching
/// the accepting threshold from the later one first. Threshold 0 accepts epsilon
/// and every unary word. Full state/count policy admitted before allocation.
/// O(minimum_length+1) time and owned storage, no randomness/shared state.
/// @param minimum_length Minimum accepted word length, zero allowed.
/// @param max_states Complete threshold+1 state cap, default 100000; zero disables.
/// @return Owned complete one-symbol DFA with threshold+1 states and start0.
/// @throws std::length_error On state cap, int/count/container representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto d=dm::algorithms::dfa_unary_threshold(2);
/// std::print("{} {}\n",dm::algorithms::dfa_accepts(d,{0}),dm::algorithms::dfa_accepts(d,{0,0})); // false true
/// @endcode
/// @see hopcroft_minimize_run(), moore_minimize()
inline DFA dfa_unary_threshold(std::size_t minimum_length, std::size_t max_states = 100000) {
  const auto count = hopcroft_detail::add(minimum_length, 1);
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DFA threshold int count");
  hopcroft_detail::admit(count, max_states);
  DFA d;
  d.num_states  = static_cast<int>(count);
  d.num_symbols = 1;
  d.start       = 0;
  d.delta.resize(count);
  d.accept.assign(count, 0);
  for (std::size_t i = 0; i < count; ++i)
    d.delta[i] = {static_cast<int>(i + 1 < count ? i + 1 : i)};
  d.accept.back() = 1;
  return d;
}
} // namespace datamunge::algorithms
