#include <datamunge/algorithms/cycle_detection.hpp>

namespace datamunge::algorithms {

CycleResult brent_cycle_detection(std::uint64_t x0, const std::function<std::uint64_t(std::uint64_t)>& f) {
    // ---- Phase 1: find the cycle length lambda ----
    // The "tortoise" is teleported to the hare's position each time the step budget (a power of
    // two) is exhausted; lambda counts the hare's steps since that teleport. When the hare's value
    // meets the tortoise's, the number of steps taken is exactly the cycle length.
    std::size_t power = 1;
    std::size_t lambda = 1;
    std::uint64_t tortoise = x0;
    std::uint64_t hare = f(x0);
    while (tortoise != hare) {
        if (power == lambda) { // step budget exhausted: teleport tortoise, double the budget
            tortoise = hare;
            power *= 2;
            lambda = 0;
        }
        hare = f(hare);
        ++lambda;
    }

    // ---- Phase 2: find the cycle-start index mu ----
    // Put both pointers at x0, advance the hare lambda steps ahead, then move both in lockstep;
    // they first coincide exactly at x_mu, the first state of the cycle.
    tortoise = x0;
    hare = x0;
    for (std::size_t i = 0; i < lambda; ++i) hare = f(hare);
    std::size_t mu = 0;
    while (tortoise != hare) {
        tortoise = f(tortoise);
        hare = f(hare);
        ++mu;
    }

    return CycleResult{lambda, mu};
}

} // namespace datamunge::algorithms
