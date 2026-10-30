/// @file banker.cpp
/// @brief Safe-sequence counting, deadlock detection and allocation simulation for the banker's algorithm.

#include <datamunge/algorithms/banker.hpp>

#include <random>
#include <stdexcept>

namespace datamunge::algorithms {

std::uint64_t banker_count_safe_sequences(const std::vector<std::vector<int>>& allocation,
                                          const std::vector<std::vector<int>>& max_demand,
                                          const std::vector<int>&              available) {
    const std::size_t n = allocation.size(), m = available.size();
    if (n > 24) throw std::invalid_argument("banker_count_safe_sequences: at most 24 processes");
    if (max_demand.size() != n) throw std::invalid_argument("banker_count_safe_sequences: shape mismatch");
    for (std::size_t i = 0; i < n; ++i)
        if (allocation[i].size() != m || max_demand[i].size() != m)
            throw std::invalid_argument("banker_count_safe_sequences: shape mismatch");
    const std::size_t full = (std::size_t{1} << n);
    std::vector<std::uint64_t> ways(full, 0);
    std::vector<int>           work(m);
    ways[0] = 1;
    for (std::size_t mask = 0; mask < full; ++mask) {
        if (ways[mask] == 0) continue;
        for (std::size_t j = 0; j < m; ++j) {
            work[j] = available[j];
            for (std::size_t i = 0; i < n; ++i)
                if (mask >> i & 1) work[j] += allocation[i][j];
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (mask >> i & 1) continue;
            bool fits = true;
            for (std::size_t j = 0; j < m && fits; ++j) fits = max_demand[i][j] - allocation[i][j] <= work[j];
            if (fits) ways[mask | (std::size_t{1} << i)] += ways[mask];
        }
    }
    return ways[full - 1];
}

std::vector<std::size_t> deadlocked_processes(const std::vector<std::vector<int>>& allocation,
                                              const std::vector<std::vector<int>>& requests,
                                              const std::vector<int>&              available) {
    const std::size_t n = allocation.size(), m = available.size();
    std::vector<int>  work = available;
    std::vector<char> done(n, 0);
    for (bool progress = true; progress;) {
        progress = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (done[i]) continue;
            bool fits = true;
            for (std::size_t j = 0; j < m && fits; ++j) fits = requests[i][j] <= work[j];
            if (!fits) continue;
            for (std::size_t j = 0; j < m; ++j) work[j] += allocation[i][j];
            done[i]  = 1;
            progress = true;
        }
    }
    std::vector<std::size_t> stuck;
    for (std::size_t i = 0; i < n; ++i)
        if (!done[i]) stuck.push_back(i);
    return stuck;
}

AllocationRun simulate_resource_allocation(const std::vector<std::vector<int>>& max_demand,
                                           const std::vector<int>& total, AllocationPolicy policy,
                                           std::size_t max_steps, std::uint64_t seed) {
    const std::size_t n = max_demand.size(), m = total.size();
    for (const auto& row : max_demand) {
        if (row.size() != m) throw std::invalid_argument("simulate_resource_allocation: shape mismatch");
        for (std::size_t j = 0; j < m; ++j)
            if (row[j] > total[j] || row[j] < 0) throw std::invalid_argument("simulate_resource_allocation: claim exceeds total");
    }
    AllocationRun                 run;
    std::vector<std::vector<int>> alloc(n, std::vector<int>(m, 0));
    std::vector<int>              available = total;
    std::vector<long>             pending(n, -1);  // resource type requested, or -1
    std::mt19937_64               rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    long   units = 0;
    for (int t : total) units += t;
    double used_sum = 0.0;

    auto holds_claim = [&](std::size_t i) {
        for (std::size_t j = 0; j < m; ++j)
            if (alloc[i][j] < max_demand[i][j]) return false;
        return true;
    };
    // Deadlock: no process can ever move again. A process can move if it holds its full claim,
    // or if a unit it is waiting for (its pending type, or any type it still needs) is free.
    auto stuck = [&]() {
        for (std::size_t i = 0; i < n; ++i) {
            if (holds_claim(i)) return false;
            for (std::size_t j = 0; j < m; ++j) {
                const bool wanted = pending[i] >= 0 ? static_cast<std::size_t>(pending[i]) == j : alloc[i][j] < max_demand[i][j];
                if (wanted && available[j] > 0) return false;
            }
        }
        return true;
    };

    for (run.steps = 0; run.steps < max_steps; ++run.steps) {
        long used = 0;
        for (std::size_t j = 0; j < m; ++j) used += total[j] - available[j];
        used_sum += units ? static_cast<double>(used) / static_cast<double>(units) : 0.0;
        const std::size_t i = pick(rng);
        if (holds_claim(i)) {
            for (std::size_t j = 0; j < m; ++j) {
                available[j] += alloc[i][j];
                alloc[i][j] = 0;
            }
            pending[i] = -1;
            ++run.jobs_finished;
            continue;
        }
        if (pending[i] < 0) {
            std::vector<std::size_t> needed;
            for (std::size_t j = 0; j < m; ++j)
                for (int k = alloc[i][j]; k < max_demand[i][j]; ++k) needed.push_back(j);  // weight by units needed
            pending[i] = static_cast<long>(needed[std::uniform_int_distribution<std::size_t>(0, needed.size() - 1)(rng)]);
        }
        const auto j = static_cast<std::size_t>(pending[i]);
        if (available[j] == 0) {
            ++run.blocked;
            if (policy == AllocationPolicy::GrantIfAvailable && stuck()) {
                run.deadlocked = true;
                ++run.steps;
                break;
            }
            continue;
        }
        if (policy == AllocationPolicy::Banker) {
            std::vector<int> request(m, 0);
            request[j] = 1;
            if (!banker_request_grantable(i, request, alloc, max_demand, available)) {
                ++run.denials;
                continue;
            }
        }
        ++alloc[i][j];
        --available[j];
        pending[i] = -1;
        ++run.grants;
    }
    run.utilisation = run.steps ? used_sum / static_cast<double>(run.steps) : 0.0;
    return run;
}

}  // namespace datamunge::algorithms
