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
#include <cstdint>
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

/// \brief Counts every safe sequence of the state (orderings in which each process's remaining
///        need fits in the free resources plus the allocations of the processes before it).
///
/// The free pool after finishing a set of processes depends only on the set, so a dynamic
/// program over subsets counts the orderings in \f$O(2^n n m)\f$ time.
/// \throws std::invalid_argument if there are more than 24 processes or the shapes disagree.
std::uint64_t banker_count_safe_sequences(const std::vector<std::vector<int>>& allocation,
                                          const std::vector<std::vector<int>>& max_demand,
                                          const std::vector<int>&              available);

/// \brief Deadlock detection (Holt's reduction): processes with outstanding @p requests that
///        can never be satisfied, even if every satisfiable process runs and releases.
///
/// Repeatedly marks a process whose current request fits in the free pool as able to finish and
/// returns its allocation to the pool. The processes left unmarked are deadlocked.
/// \return the ids of the deadlocked processes (empty if none).
std::vector<std::size_t> deadlocked_processes(const std::vector<std::vector<int>>& allocation,
                                              const std::vector<std::vector<int>>& requests,
                                              const std::vector<int>&              available);

/// How requests are decided in simulate_resource_allocation().
enum class AllocationPolicy {
    Banker,           ///< grant a request only if the resulting state is safe.
    GrantIfAvailable  ///< grant whenever the units are free.
};

/// Outcome of a simulated run of processes acquiring resources one unit at a time.
struct AllocationRun {
    bool          deadlocked    = false;  ///< whether the run ended in deadlock.
    std::size_t   steps         = 0;      ///< scheduler steps taken.
    std::size_t   jobs_finished = 0;      ///< jobs that reached their maximum claim and released.
    std::size_t   grants        = 0;      ///< unit requests granted.
    std::size_t   denials       = 0;      ///< unit requests refused although the unit was free (banker only).
    std::size_t   blocked       = 0;      ///< unit requests that had to wait because no unit was free.
    double        utilisation   = 0.0;    ///< mean fraction of all units allocated, over the steps.
};

/// \brief Simulates processes that each repeatedly run jobs: a job acquires its maximum claim one
///        random unit at a time, then releases everything and starts again.
///
/// At each step a random process acts. A process holding its full claim finishes the job and
/// releases. Otherwise it asks for one unit of a resource type it still needs (keeping the same
/// pending request until granted), and the @p policy decides. The run stops after @p max_steps
/// steps or when no process can ever act again (deadlock).
/// \throws std::invalid_argument if a claim exceeds the total of its resource type.
AllocationRun simulate_resource_allocation(const std::vector<std::vector<int>>& max_demand,
                                           const std::vector<int>& total, AllocationPolicy policy,
                                           std::size_t max_steps, std::uint64_t seed);

}  // namespace datamunge::algorithms
