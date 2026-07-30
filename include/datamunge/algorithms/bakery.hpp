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
#include <cstddef>
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

}  // namespace datamunge::algorithms
