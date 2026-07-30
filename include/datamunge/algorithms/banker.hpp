#pragma once

/// \file banker.hpp
/// \brief Banker's algorithm for deadlock avoidance (Dijkstra 1965).
///
/// The banker's algorithm decides whether granting a resource request keeps the system in a
/// *safe state* -- one in which there exists an ordering of the processes (a *safe sequence*)
/// such that each can obtain its maximum remaining need from the currently available
/// resources plus those released by earlier-finishing processes. A state is safe iff such a
/// sequence exists; a request is granted only if the resulting state is still safe, which
/// guarantees the system never deadlocks. This module provides the safety algorithm (does a
/// safe sequence exist, and what is it) and the resource-request test.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Result of the safety check: whether the state is safe and, if so, a safe sequence.
struct SafetyResult {
    bool                     safe;      ///< True iff a safe sequence exists.
    std::vector<std::size_t> sequence;  ///< A safe order of process ids (empty if unsafe).
};

/// \brief Run the safety algorithm on an allocation state.
///
/// \param allocation  n x m matrix: resources of each type currently held by each process.
/// \param max_demand  n x m matrix: maximum resources each process may ever request.
/// \param available   length-m vector: units of each resource currently free.
/// \return whether the state is safe and a safe sequence if so.
///
/// Need = max_demand - allocation. Repeatedly find an unfinished process whose remaining
/// need fits in the working set of free resources; "run" it and reclaim its allocation.
/// If every process can finish, the state is safe.
inline SafetyResult banker_safety(const std::vector<std::vector<int>>& allocation,
                                  const std::vector<std::vector<int>>& max_demand,
                                  const std::vector<int>&              available) {
    const std::size_t n = allocation.size();
    const std::size_t m = available.size();

    std::vector<int>  work = available;
    std::vector<bool> finished(n, false);
    std::vector<std::size_t> sequence;

    for (std::size_t count = 0; count < n; ++count) {
        bool progressed = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (finished[i]) continue;
            bool fits = true;
            for (std::size_t j = 0; j < m; ++j)
                if (max_demand[i][j] - allocation[i][j] > work[j]) { fits = false; break; }
            if (!fits) continue;
            for (std::size_t j = 0; j < m; ++j) work[j] += allocation[i][j];  // process finishes, releases
            finished[i] = true;
            sequence.push_back(i);
            progressed = true;
        }
        if (!progressed) break;  // no process can proceed -> unsafe
    }

    if (sequence.size() == n) return {true, sequence};
    return {false, {}};
}

/// \brief Test whether a resource request by one process can be safely granted.
///
/// \param process    id of the requesting process.
/// \param request    length-m request vector.
/// \param allocation current allocation matrix (modified copy used internally).
/// \param max_demand maximum-demand matrix.
/// \param available  currently-free resources.
/// \return true iff the request is <= the process's need and available, and the state that
///         would result from granting it is safe.
inline bool banker_request_grantable(std::size_t                          process,
                                     const std::vector<int>&              request,
                                     const std::vector<std::vector<int>>& allocation,
                                     const std::vector<std::vector<int>>& max_demand,
                                     const std::vector<int>&              available) {
    const std::size_t m = available.size();
    for (std::size_t j = 0; j < m; ++j) {
        if (request[j] > max_demand[process][j] - allocation[process][j]) return false;  // exceeds need
        if (request[j] > available[j]) return false;                                     // exceeds available
    }
    // Tentatively grant and test safety.
    std::vector<std::vector<int>> alloc2 = allocation;
    std::vector<int>              avail2 = available;
    for (std::size_t j = 0; j < m; ++j) {
        alloc2[process][j] += request[j];
        avail2[j]          -= request[j];
    }
    return banker_safety(alloc2, max_demand, avail2).safe;
}

}  // namespace datamunge::algorithms
