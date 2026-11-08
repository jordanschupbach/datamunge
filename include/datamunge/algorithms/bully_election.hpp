#pragma once
/// @file bully_election.hpp
/// @brief Checked static bully-election models: legacy highest-responder shortcut and full responder fanout.

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief Leader and attempted-message counts for one explicitly chosen election model.
/// Field order/types retain the original aggregate layout. Attempts to failed
/// destinations count as ELECTION/COORDINATOR messages; only live responders send
/// ANSWER. These are logical sends, not bytes, timings or network rounds.
/// @par Example
/// @code
/// const auto r=dm::algorithms::bully_elect(6,{1,1,1,1,0,0},1);
/// std::print("{} {} {} {}\n",r.leader,r.election_messages,r.answer_messages,r.coordinator_messages); // 3 6 2 3
/// @endcode
/// @see bully_elect(), bully_elect_fanout()
struct BullyResult {
  int leader{-1}; ///< Highest live id after an active election, or -1 if no process is up.
  std::size_t election_messages{0}; ///< Attempts from candidates to every higher id, including failed destinations.
  std::size_t answer_messages{0}; ///< Successful higher-to-lower replies to delivered ELECTION messages.
  std::size_t coordinator_messages{0}; ///< Announcement attempts from the winner to every lower id, including failed ones.
};

/// @brief Logical message categories in the static three-message teaching model.
/// This abstraction does not reproduce the original paper's halt/reorganization/
/// ready phases or simulate crash recovery and communication failures.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bully_elect_fanout_trace(2,{1,1},0);
/// std::print("{}\n",t.messages[0].kind==dm::algorithms::BullyMessageKind::Election); // true
/// @endcode
/// @par References
/// - P. Krzyzanowski, Distributed Mutual Exclusion and Leader Election,
///   https://people.cs.rutgers.edu/~pxk/classes/417/notes/mutex.html (three-message teaching model).
/// @see BullyMessage
enum class BullyMessageKind {
  Election, ///< Candidate contacts a known higher id, whether up or down.
  Answer, ///< Live recipient acknowledges a lower candidate's ELECTION.
  Coordinator ///< Winner announces itself to a known lower id, whether up or down.
};

/// @brief One actual logical send in deterministic kernel order.
/// ELECTION sends visit higher ids in ascending order; a live recipient's ANSWER
/// follows immediately. Candidate phases use FIFO activation order. Announcements
/// are sent in ascending lower-id order after all candidate phases. phase indexes
/// candidate processing, not physical time; announcements use the number of phases.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bully_elect_fanout_trace(2,{1,1},0);
/// std::print("{} {} {}\n",t.messages[0].sender,t.messages[0].receiver,t.messages[0].delivered); // 0 1 true
/// @endcode
/// @see BullyTrace
struct BullyMessage {
  BullyMessageKind kind{BullyMessageKind::Election}; ///< ELECTION, ANSWER or COORDINATOR send.
  int sender{-1}; ///< Live sending process id.
  int receiver{-1}; ///< Destination id, possibly down.
  bool delivered{false}; ///< Whether the fixed liveness snapshot permits delivery.
  std::size_t phase{0}; ///< Zero-based candidate phase, or final announcement phase.
};

/// @brief Actual work and highest responder for one processed candidate.
/// A phase sends to all higher ids and gathers their immediate logical answers.
/// Fanout activates every previously inactive live responder once; the shortcut
/// instead activates only the highest responder after completing this phase.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bully_elect_fanout_trace(6,{1,1,1,1,0,0},1);
/// std::print("{} {} {}\n",t.rounds[0].process,t.rounds[0].highest_responder,t.rounds[0].candidates_started); // 1 3 2
/// @endcode
/// @see BullyTrace
struct BullyRound {
  int process{-1}; ///< Live process whose election is processed in this phase.
  int highest_responder{-1}; ///< Highest live higher id answering, or -1 if none.
  std::size_t election_messages{0}; ///< Higher-id contact attempts in this phase.
  std::size_t answer_messages{0}; ///< Live replies to this phase's contacts.
  std::size_t candidates_started{0}; ///< New candidates actually activated by this phase.
};

/// @brief Message-recording and complete-call budget options for either trace model.
/// max_messages=0 means unlimited; a positive cap counts every attempted logical
/// send, including undelivered ones, and throws before the next send would exceed
/// the cap. No partial trace or leader is returned. record_messages=false omits the
/// O(M) message vector while retaining actual O(n) phase/learned-state records and
/// all counters. Input flags are borrowed only during the call.
/// @par Example
/// @code
/// dm::algorithms::BullyTraceOptions o;o.record_messages=false;
/// const auto t=dm::algorithms::bully_elect_fanout_trace(2,{1,1},0,o);
/// std::print("{} {}\n",t.total_messages,t.messages.size()); // 3 0
/// @endcode
/// @see bully_elect_trace(), bully_elect_fanout_trace()
struct BullyTraceOptions {
  std::size_t max_messages{0}; ///< Complete-call logical-send limit, zero for unlimited.
  bool record_messages{true}; ///< Whether to own individual send records, default true.
};

/// @brief Owned actual candidate phases, logical sends and final learned leaders.
/// fanout distinguishes the full responder model from the legacy shortcut.
/// result and counters come from the same kernel as the corresponding ordinary
/// function; no election is replayed. learned_leader has one entry per process:
/// live nodes learn the winner, down nodes remain -1. An all-down/empty system
/// has no phases or messages and no leader. Records own their storage and inputs
/// are unchanged; calls share no mutable state. See BullyTraceOptions for storage.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bully_elect_fanout_trace(6,{1,1,1,1,0,0},1);
/// std::print("{} {}\n",t.total_messages,t.learned_leader); // 15 [3, 3, 3, 3, -1, -1]
/// @endcode
/// @see BullyMessage, BullyRound, bully_elect_fanout_trace()
struct BullyTrace {
  BullyResult result; ///< Elected leader and category counts for this model.
  int initiator{-1}; ///< Requested starting process, or -1 for an inactive system.
  bool fanout{false}; ///< Whether every live responder can start an election once.
  std::size_t total_messages{0}; ///< All ELECTION+ANSWER+COORDINATOR attempts.
  std::size_t delivered_messages{0}; ///< Logical sends reaching live destinations.
  std::vector<BullyRound> rounds; ///< Actual FIFO candidate-processing sequence.
  std::vector<BullyMessage> messages; ///< Actual send sequence, or empty in lean mode.
  std::vector<int> learned_leader; ///< Final per-process coordinator knowledge; -1 for down nodes.
};

/// @brief Shared validation and actual send/candidate processing for both models.
namespace bully_detail {
/// @brief Validate id representation, liveness storage and an active initiator.
/// Extra liveness entries are ignored; any nonzero char means up. An initiator
/// of -1 is accepted only when every used process is down, including n=0.
/// @param n Number of used process ids, at most INT_MAX.
/// @param alive Borrowed flags with at least n entries.
/// @param initiator Live in-range id, or -1 for an entirely inactive prefix.
/// @return Whether a live process starts an election.
/// @throws std::length_error If n cannot be represented by the int-id API.
/// @throws std::invalid_argument If flags or initiator are invalid.
inline bool validate(std::size_t n,const std::vector<char>& alive,int initiator) {
  if (n>static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("bully process ids do not fit int");
  if (alive.size()<n) throw std::invalid_argument("bully liveness vector is shorter than process count");
  if (initiator==-1) {
    for (std::size_t i=0;i<n;++i) if (alive[i])
      throw std::invalid_argument("bully election needs a live initiator when any process is up");
    return false;
  }
  if (initiator<0 || static_cast<std::size_t>(initiator)>=n || !alive[initiator])
    throw std::invalid_argument("bully initiator must be a live in-range process");
  return true;
}

/// @brief Execute a static candidate cascade and record the actual work when requested.
/// Fixed liveness, reliable immediate logical delivery to live destinations and
/// one activation per process are explicit model assumptions. This is not a timed
/// network simulator or a complete crash/recovery/reorganization implementation.
/// @tparam Record Whether to own phase/message/learned-state records.
/// @tparam Fanout Whether every live responder activates, rather than only the highest.
/// @param n Number of processes with ids in [0,n).
/// @param alive Borrowed liveness prefix.
/// @param initiator Live starting id, or -1 for an all-down/empty prefix.
/// @param options Recording and complete-call send budget.
/// @return Result and optional actual records, with no partial return on failure.
/// @throws std::invalid_argument For invalid flags/initiator.
/// @throws std::length_error For unrepresentable ids or vector capacity.
/// @throws std::runtime_error Before a positive message budget is exceeded.
/// @throws std::overflow_error Before a size_t message counter could wrap.
/// @throws std::bad_alloc If local work/trace storage cannot be allocated.
template<bool Record,bool Fanout>
BullyTrace run(std::size_t n,const std::vector<char>& alive,int initiator,BullyTraceOptions options) {
  const bool active=validate(n,alive,initiator);
  BullyTrace trace;trace.initiator=initiator;trace.fanout=Fanout;
  if constexpr (Record) trace.learned_leader.assign(n,-1);
  if (!active) return trace;
  std::vector<char> activated(n,0);activated[initiator]=1;
  std::vector<int> candidates{initiator};
  /// @brief Check a whole-call send limit, emit one actual logical message and update counters.
  /// @param kind Message category.
  /// @param sender Live sender.
  /// @param receiver Destination id.
  /// @param delivered Whether the fixed destination is up.
  /// @param phase Candidate index or final announcement phase, not elapsed time.
  auto send=[&](BullyMessageKind kind,int sender,int receiver,bool delivered,std::size_t phase) {
    if (options.max_messages && trace.total_messages>=options.max_messages)
      throw std::runtime_error("bully message budget exceeded");
    if (trace.total_messages==std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("bully message count overflow");
    if constexpr (Record) if (options.record_messages)
      trace.messages.push_back({kind,sender,receiver,delivered,phase});
    ++trace.total_messages;
    if (delivered) ++trace.delivered_messages;
    if (kind==BullyMessageKind::Election) ++trace.result.election_messages;
    else if (kind==BullyMessageKind::Answer) ++trace.result.answer_messages;
    else ++trace.result.coordinator_messages;
  };
  for (std::size_t phase=0;phase<candidates.size();++phase) {
    const int current=candidates[phase];
    BullyRound round;round.process=current;
    for (int higher=current+1;higher<static_cast<int>(n);++higher) {
      send(BullyMessageKind::Election,current,higher,alive[higher]!=0,phase);++round.election_messages;
      if (alive[higher]) {
        send(BullyMessageKind::Answer,higher,current,true,phase);++round.answer_messages;
        round.highest_responder=higher;
        if constexpr (Fanout) if (!activated[higher]) {
          candidates.push_back(higher);activated[higher]=1;++round.candidates_started;
        }
      }
    }
    if (round.highest_responder<0) trace.result.leader=current;
    else if constexpr (!Fanout) {
      candidates.push_back(round.highest_responder);activated[round.highest_responder]=1;++round.candidates_started;
    }
    if constexpr (Record) trace.rounds.push_back(round);
  }
  const int winner=trace.result.leader;
  if constexpr (Record) trace.learned_leader[winner]=winner;
  for (int lower=0;lower<winner;++lower) {
    send(BullyMessageKind::Coordinator,winner,lower,alive[lower]!=0,candidates.size());
    if constexpr (Record) if (alive[lower]) trace.learned_leader[lower]=winner;
  }
  return trace;
}
} // namespace bully_detail

/// @brief Preserve the original highest-responder shortcut and its valid-input message counts.
/// Each candidate contacts every higher id and receives live replies; only the
/// highest responder runs the next election. Thus at most two candidates run.
/// Winner announcements target every lower id, including failed destinations.
/// This O(n) shortcut is not the quadratic all-responder teaching model. O(n)
/// local work storage; no trace records are allocated. Flags are not mutated.
/// @param num_processes Number of ids, at most INT_MAX; zero is supported with initiator=-1.
/// @param alive At least num_processes flags; nonzero means up, extras ignored.
/// @param initiator Live in-range id, or -1 only if all used processes are down.
/// @return Original-layout result, with leader=-1 and zero sends for an inactive system.
/// @throws std::invalid_argument If flags/initiator are invalid.
/// @throws std::length_error If ids/vector sizes cannot be represented.
/// @throws std::overflow_error If a size_t send count cannot be represented.
/// @throws std::bad_alloc If local work storage cannot be allocated.
/// @par Example
/// @code
/// const auto r=dm::algorithms::bully_elect(6,{1,1,1,1,0,0},1);
/// std::print("{} {}\n",r.election_messages,r.answer_messages); // 6 2
/// @endcode
/// @par References
/// - H. Garcia-Molina (1982), Elections in a Distributed Computing System,
///   IEEE Transactions on Computers C-31(1), 48-59,
///   https://homepage.divms.uiowa.edu/~ghosh/Bully.pdf.
///   This function is a highest-responder shortcut, not that paper's full protocol.
/// @see bully_elect_trace(), bully_elect_fanout()
inline BullyResult bully_elect(std::size_t num_processes,const std::vector<char>& alive,int initiator) {
  return bully_detail::run<false,false>(num_processes,alive,initiator,{}).result;
}

/// @brief Trace the actual legacy highest-responder cascade without replay.
/// Shares the bully_elect() kernel and preserves its result fields. See that
/// function for input and attempted-send conventions, and BullyTraceOptions for
/// lean records and a positive complete-call message budget. O(n+M) time/storage
/// with messages retained, O(n) trace storage in lean mode; here M=O(n).
/// @param num_processes Number of ids, at most INT_MAX.
/// @param alive Borrowed liveness prefix; extras ignored and nonzero means up.
/// @param initiator Live id, or -1 only for an inactive prefix.
/// @param options Recording/budget choices, default full records and no cap.
/// @return Owned actual shortcut phases, counters, sends and learned leaders.
/// @throws std::invalid_argument If flags/initiator are invalid.
/// @throws std::length_error If ids or vector capacity cannot be represented.
/// @throws std::runtime_error Before a positive complete-call send cap is exceeded.
/// @throws std::overflow_error Before a size_t message count could wrap.
/// @throws std::bad_alloc If local work/trace storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bully_elect_trace(6,{1,1,1,1,0,0},1);
/// std::print("{} {}\n",t.rounds.size(),t.total_messages); // 2 11
/// @endcode
/// @see bully_elect(), bully_elect_fanout_trace()
inline BullyTrace bully_elect_trace(std::size_t num_processes,const std::vector<char>& alive,int initiator,
                                   BullyTraceOptions options={}) {
  return bully_detail::run<true,false>(num_processes,alive,initiator,options);
}

/// @brief Run the static three-message model where every live higher responder activates once.
/// A delivered ELECTION elicits ANSWER and schedules the live recipient if not
/// already active in this epoch. Candidates execute in FIFO order; higher targets
/// are visited in ascending id order. The no-higher-responder candidate wins and
/// announces to every lower known id. Liveness never changes; no delays, restarts,
/// partitions or recovery are simulated. The original paper's reorganization
/// phases are omitted. O(n+M) time and O(n) work storage for M logical sends;
/// worst-case M=n*n-1 for all-up nodes initiated by zero. No trace storage.
/// @param num_processes Number of process ids, at most INT_MAX.
/// @param alive Borrowed prefix with at least num_processes flags; extras ignored.
/// @param initiator Live in-range id, or -1 only for an all-down/empty prefix.
/// @return Leader and actual attempted-send counts in the original result layout.
/// @throws std::invalid_argument If flags/initiator are invalid.
/// @throws std::length_error If ids/vector capacity cannot be represented.
/// @throws std::overflow_error Before a size_t message count could wrap.
/// @throws std::bad_alloc If local work storage cannot be allocated.
/// @par Example
/// @code
/// const auto r=dm::algorithms::bully_elect_fanout(6,{1,1,1,1,0,0},1);
/// std::print("{} {} {} {}\n",r.leader,r.election_messages,r.answer_messages,r.coordinator_messages); // 3 9 3 3
/// @endcode
/// @par References
/// - P. Krzyzanowski, Distributed Mutual Exclusion and Leader Election,
///   https://people.cs.rutgers.edu/~pxk/classes/417/notes/mutex.html.
/// - H. Garcia-Molina (1982), Elections in a Distributed Computing System,
///   https://homepage.divms.uiowa.edu/~ghosh/Bully.pdf (origin and broader failure assumptions).
/// @see bully_elect_fanout_trace(), bully_elect()
inline BullyResult bully_elect_fanout(std::size_t num_processes,const std::vector<char>& alive,int initiator) {
  return bully_detail::run<false,true>(num_processes,alive,initiator,{}).result;
}

/// @brief Record actual full-responder candidate phases and messages in one static epoch.
/// Shares the bully_elect_fanout() kernel without replay. Full recording needs
/// O(n+M) storage, lean recording O(n), while both still process every logical
/// send and enforce the same complete-call budget. phase ids do not measure time.
/// @param num_processes Number of process ids, at most INT_MAX.
/// @param alive Borrowed liveness prefix; any nonzero char is up, extras ignored.
/// @param initiator Live in-range id, or -1 only when every used node is down.
/// @param options Record individual sends by default; zero budget means unlimited.
/// @return Owned actual fanout trace, including final learned leaders for all nodes.
/// @throws std::invalid_argument If flags/initiator are invalid.
/// @throws std::length_error If ids or vector capacity cannot be represented.
/// @throws std::runtime_error Before a positive complete-call message cap is exceeded.
/// @throws std::overflow_error Before a size_t message count could wrap.
/// @throws std::bad_alloc If local work/trace storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bully_elect_fanout_trace(6,{1,1,1,1,0,0},1);
/// std::print("{} {}\n",t.rounds.size(),t.delivered_messages); // 3 9
/// @endcode
/// @par References
/// - P. Krzyzanowski, Distributed Mutual Exclusion and Leader Election,
///   https://people.cs.rutgers.edu/~pxk/classes/417/notes/mutex.html.
/// @see bully_elect_fanout(), BullyTraceOptions
inline BullyTrace bully_elect_fanout_trace(std::size_t num_processes,const std::vector<char>& alive,int initiator,
                                          BullyTraceOptions options={}) {
  return bully_detail::run<true,true>(num_processes,alive,initiator,options);
}

} // namespace datamunge::algorithms
