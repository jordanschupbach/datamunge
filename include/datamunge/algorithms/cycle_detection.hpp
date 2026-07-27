#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace datamunge::algorithms {

/// @brief The result of detecting a cycle in the sequence x0, f(x0), f(f(x0)), ... . Any such
///        "functional graph" iteration over a finite state space is eventually periodic, tracing
///        a rho shape: a tail of @c mu transient states leading into a cycle of length @c lambda.
struct CycleResult {
    /// @brief The cycle length lambda: the smallest lambda >= 1 with x_{mu + lambda} == x_{mu}.
    std::size_t lambda{0};
    /// @brief The start index mu of the cycle (the tail length): the smallest mu >= 0 such that
    ///        x_{mu} recurs, i.e. x_{mu + lambda} == x_{mu}.
    std::size_t mu{0};
};

/// @brief Brent's cycle-finding algorithm (Brent 1980). Detects the period @c lambda and the
///        cycle-start index @c mu of the iterated sequence x_i = f^i(x0) using only two state
///        variables and O(1) memory, comparing values taken at successive powers of two rather
///        than at a fixed lag (Floyd's approach). It needs roughly lambda + mu applications of
///        @p f -- typically fewer than Floyd's tortoise-and-hare -- and is the cycle detector
///        underlying Pollard's rho factorization and period-finding for pseudorandom sequences.
///
/// @param x0 the starting state.
/// @param f  the transition function; the sequence iterated is x0, f(x0), f(f(x0)), ...
/// @return the cycle length @c lambda and start index @c mu.
CycleResult brent_cycle_detection(std::uint64_t x0, const std::function<std::uint64_t(std::uint64_t)>& f);

/// @brief Floyd's cycle-finding algorithm -- the classic "tortoise and the hare" (Floyd,
///        published by Knuth 1969). Detects the period @c lambda and cycle-start index @c mu of
///        the iterated sequence x_i = f^i(x0) in O(1) memory, by running one pointer at single
///        and another at double speed until they collide inside the cycle, then walking two
///        pointers in lockstep to locate the cycle's entry and once more around to measure its
///        length. Same asymptotics as Brent's algorithm but more applications of @p f in
///        practice, because the hare re-treads ground at double speed.
///
/// @param x0 the starting state.
/// @param f  the transition function; the sequence iterated is x0, f(x0), f(f(x0)), ...
/// @return the cycle length @c lambda and start index @c mu.
CycleResult floyd_cycle_detection(std::uint64_t x0, const std::function<std::uint64_t(std::uint64_t)>& f);

} // namespace datamunge::algorithms
