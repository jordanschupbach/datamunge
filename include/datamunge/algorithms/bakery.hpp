#pragma once

/// \file bakery.hpp
/// \brief Lamport's bakery algorithm for mutual exclusion (Lamport 1974).
///
/// The bakery algorithm coordinates \f$n\f$ threads' access to a critical section using
/// only atomic reads and writes -- no special hardware instructions. Like a bakery
/// counter, each thread wanting in draws a *number* one greater than the maximum it sees,
/// then waits until it holds the smallest outstanding \f$(\text{number}, \text{id})\f$
/// pair (ties broken by id) before entering; on exit it resets its number to 0. That total
/// order on (number, id) guarantees *mutual exclusion* (at most one thread inside) and
/// *fairness* (first-come-first-served, no starvation). This module computes the service
/// order the protocol induces from a set of drawn tickets and checks the mutual-exclusion
/// invariant.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A request to enter the critical section: a drawn ticket number and the thread id.
struct BakeryTicket {
    int         number;  ///< Ticket drawn (0 means "not requesting").
    std::size_t id;      ///< Thread id (tie-breaker).
};

/// \brief Compute the critical-section service order for a set of bakery tickets.
///
/// Requesting threads (number > 0) are served in increasing \f$(\text{number}, \text{id})\f$
/// order -- exactly the order the bakery algorithm enforces. Returns the thread ids in the
/// order they enter the critical section.
inline std::vector<std::size_t> bakery_service_order(const std::vector<BakeryTicket>& tickets) {
    std::vector<BakeryTicket> req;
    for (const auto& t : tickets)
        if (t.number > 0) req.push_back(t);
    std::sort(req.begin(), req.end(), [](const BakeryTicket& a, const BakeryTicket& b) {
        if (a.number != b.number) return a.number < b.number;
        return a.id < b.id;  // ties broken by id
    });
    std::vector<std::size_t> order;
    order.reserve(req.size());
    for (const auto& t : req) order.push_back(t.id);
    return order;
}

/// \brief The next thread that may enter the critical section (smallest (number,id)), or
///        SIZE_MAX if none is requesting.
inline std::size_t bakery_next(const std::vector<BakeryTicket>& tickets) {
    auto order = bakery_service_order(tickets);
    return order.empty() ? static_cast<std::size_t>(-1) : order.front();
}

/// \brief Check the mutual-exclusion invariant: all requesting tickets have distinct
///        \f$(\text{number}, \text{id})\f$ keys, so exactly one thread is "first".
inline bool bakery_mutual_exclusion_holds(const std::vector<BakeryTicket>& tickets) {
    std::vector<std::pair<int, std::size_t>> keys;
    for (const auto& t : tickets)
        if (t.number > 0) keys.push_back({t.number, t.id});
    std::sort(keys.begin(), keys.end());
    for (std::size_t i = 1; i < keys.size(); ++i)
        if (keys[i] == keys[i - 1]) return false;  // a duplicate key would break exclusion
    return true;
}

/// \brief Lamport's bakery lock for a fixed number of threads, built from sequentially
///        consistent atomic loads and stores only (no read-modify-write instructions).
///
/// Thread @c i calls lock(i) and unlock(i). Ticket numbers are unbounded in principle; the
/// lock records the largest one drawn.
class BakeryLock {
public:
    /// \brief A lock for @p threads threads with ids 0 .. threads-1.
    /// \throws std::invalid_argument if @p threads is 0.
    explicit BakeryLock(std::size_t threads);

    /// \brief Enters the critical section as thread @p id (doorway, then wait for the smallest ticket).
    void lock(std::size_t id);

    /// \brief Leaves the critical section as thread @p id (resets its ticket to 0).
    void unlock(std::size_t id);

    /// \brief Number of threads.
    std::size_t size() const noexcept { return n_; }

    /// \brief Largest ticket number drawn so far.
    std::uint64_t max_ticket() const noexcept { return max_ticket_.load(); }

private:
    std::size_t                                n_;          ///< thread count.
    std::unique_ptr<std::atomic<bool>[]>       choosing_;   ///< doorway flags.
    std::unique_ptr<std::atomic<std::uint64_t>[]> number_;  ///< tickets (0 = not requesting).
    std::atomic<std::uint64_t>                 max_ticket_{0};  ///< largest ticket drawn.
};

/// Where a process is in the bakery protocol.
enum class BakeryPhase : std::uint8_t {
    Idle,      ///< outside the protocol (non-critical section).
    Doorway,   ///< setting choosing, reading the other tickets and writing its own.
    Waiting,   ///< scanning the other processes until it holds the smallest ticket.
    Critical   ///< inside the critical section.
};

/// Options for the bakery model checker and simulator.
struct BakeryModelOptions {
    std::size_t   processes    = 2;     ///< number of processes.
    bool          use_choosing = true;  ///< whether the doorway sets and waits on the choosing flags.
    std::uint32_t max_ticket   = 6;     ///< tickets above this are cut off to keep the state space finite.
};

/// One atomic step of a process in a counterexample.
struct BakeryStep {
    std::size_t process;  ///< which process moved.
    std::string action;   ///< what it did, for example "number[1] = 1".
};

/// Result of exhaustively exploring the bakery protocol's interleavings.
struct BakeryCheck {
    std::uint64_t           states            = 0;     ///< distinct reachable states (within the ticket bound).
    std::uint64_t           transitions       = 0;     ///< atomic steps explored.
    std::uint64_t           cut_by_bound      = 0;     ///< ticket draws above max_ticket that were not followed.
    bool                    mutual_exclusion  = true;  ///< false if some state has two processes critical.
    std::size_t             max_critical      = 0;     ///< most processes simultaneously critical.
    std::vector<BakeryStep> counterexample;            ///< shortest schedule reaching a violation, if any.
};

/// \brief Explores every interleaving of the bakery protocol's atomic steps (one shared-variable
///        read or write per step) by breadth-first search, and checks mutual exclusion.
///
/// Each process repeatedly runs: choosing = true; read every number[j] one at a time to find
/// the maximum; number = max + 1; choosing = false; for each other j, wait while choosing[j],
/// then wait while number[j] != 0 and (number[j], j) < (number[i], i); enter the critical
/// section; number = 0. With @c use_choosing false the choosing writes and waits are skipped.
/// \throws std::invalid_argument if processes is 0 or above 4, or max_ticket is 0.
BakeryCheck bakery_model_check(const BakeryModelOptions& options);

/// A random interleaving of the bakery protocol.
struct BakerySimulation {
    std::vector<std::vector<BakeryPhase>>   phase;    ///< phase[t][i] after step t.
    std::vector<std::vector<std::uint32_t>> number;   ///< number[t][i] after step t.
    std::vector<std::size_t>                entries;  ///< critical-section entries per process.
    std::vector<std::size_t>                order;    ///< process ids in the order they entered.
    std::size_t                             violations = 0;  ///< steps with two or more processes critical.
};

/// \brief Runs @p steps atomic steps of the protocol, choosing the process to move uniformly at
///        random each step. The ticket bound is ignored.
BakerySimulation bakery_simulate(const BakeryModelOptions& options, std::size_t steps, std::uint64_t seed);

}  // namespace datamunge::algorithms
