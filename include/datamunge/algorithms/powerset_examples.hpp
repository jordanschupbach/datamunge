#pragma once
/// @file powerset_examples.hpp
/// @brief Reusable original suffix-language NFA for actual reachable-subset demonstrations.
#include <datamunge/algorithms/powerset_construction.hpp>
#include <cstddef>
namespace datamunge::algorithms {
/// @brief Construct the epsilon-free binary NFA accepting words whose nth symbol from the end is1.
/// The original nondeterministic guess is preserved: state0 loops on both symbols and may advance to state1 on1; n-1 remaining symbols lead to the sole accepting state n. Words shorter than n reject. O(n) generator work/storage; it does not determinize or allocate2^n states. Returned storage is owned and no state is shared.
/// @param n Positive suffix position, with n+1 representable by int.
/// @param max_states Admitted NFA states, defaultone million; zero disables the policy.
/// @return Owned valid binary NFA with n+1 states and start0.
/// @throws std::invalid_argument If n<1 or n+1 is not representable.
/// @throws std::length_error If the requested generator state count exceeds its policy or storage.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nth_from_end_nfa(2).num_states); // 3
/// @endcode
/// @par References
/// - M. O. Rabin and D. Scott (1959), Finite Automata and Their Decision Problems, doi:10.1147/rd.32.0114; this is the original report's controlled suffix-language construction.
/// @see powerset_construction_trace()
NFA nth_from_end_nfa(int n, std::size_t max_states = 1000000);
} // namespace datamunge::algorithms
