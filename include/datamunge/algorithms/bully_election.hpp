#pragma once

/// \file bully_election.hpp
/// \brief The Bully algorithm for leader election in a distributed system
///        (Garcia-Molina 1982).
///
/// When the coordinator of a distributed system fails, the survivors must *elect* a new
/// one. The *Bully algorithm* elects the highest-id alive process: a process that notices
/// the leader is gone sends an ELECTION message to all *higher*-id processes; if none
/// respond (they are down), it declares itself coordinator; if a higher one responds, it
/// takes over and repeats the process, so the largest id ultimately "bullies" the rest
/// into submission and broadcasts COORDINATOR. This simulation drives the protocol from an
/// initiator and returns the elected leader and the messages exchanged.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Outcome of a Bully election.
struct BullyResult {
    int         leader = -1;         ///< Elected coordinator (highest-id alive process), -1 if none.
    std::size_t election_messages = 0;  ///< ELECTION messages sent.
    std::size_t answer_messages   = 0;  ///< OK/ANSWER responses.
    std::size_t coordinator_messages = 0;  ///< COORDINATOR announcements.
};

/// \brief Run a Bully election among \p num_processes, given who is \p alive, starting at \p initiator.
///
/// \param num_processes  Process ids are 0..num_processes-1 (higher id = higher priority).
/// \param alive          alive[i] = true if process i is up.
/// \param initiator      The process that detected the failure and starts the election.
inline BullyResult bully_elect(std::size_t num_processes, const std::vector<char>& alive, int initiator) {
    BullyResult r;
    // Simulate the cascade: each initiator contacts higher-id processes; alive ones answer
    // and re-run from themselves. We follow the chain to the highest alive id.
    std::vector<char> visited(num_processes, 0);
    int               current = initiator;
    while (true) {
        visited[static_cast<std::size_t>(current)] = 1;
        // Contact all higher-id processes.
        int highest_alive_above = -1;
        for (int j = current + 1; j < static_cast<int>(num_processes); ++j) {
            ++r.election_messages;
            if (alive[static_cast<std::size_t>(j)]) {
                ++r.answer_messages;  // it answers OK
                if (j > highest_alive_above) highest_alive_above = j;
            }
        }
        if (highest_alive_above < 0) break;  // no higher alive process -> current wins... unless current is dead
        current = highest_alive_above;       // the highest responder takes over and re-runs
        if (visited[static_cast<std::size_t>(current)]) break;
    }
    // The leader is the highest-id alive process overall (what Bully converges to).
    for (int j = static_cast<int>(num_processes) - 1; j >= 0; --j)
        if (alive[static_cast<std::size_t>(j)]) { r.leader = j; break; }
    // The new coordinator announces to all lower-id processes.
    if (r.leader >= 0) r.coordinator_messages = static_cast<std::size_t>(r.leader);
    return r;
}

}  // namespace datamunge::algorithms
