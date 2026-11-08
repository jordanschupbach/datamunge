/// @file general_problem_solver.cpp
/// @brief One heap-based bounded planner with best-depth reopening, actual events and a separate
/// plan checker.
#include <algorithm>
#include <datamunge/algorithms/general_problem_solver.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace datamunge::algorithms {
/// @brief Implementation details of deterministic finite fact planning.
namespace {
/// @brief Increment an actual search counter with checked arithmetic.
/// @param count Counter changed only within this execution.
/// @throws std::length_error The actual next count cannot fit size_t.
void increment(std::size_t& count) {
  if (count == std::numeric_limits<std::size_t>::max())
    throw std::length_error("planning: actual counter overflow");
  ++count;
}

/// @brief Capture the next actual fixed-schema event when a trace is requested.
/// @param trace Optional owned trace receiving the event.
/// @param cap Positive actual record quota; zero disables quota.
/// @param event Fully formed event from this actual control-flow point.
/// @throws std::length_error The next actual event exceeds quota or vector capacity.
void capture(PlanningTrace* trace, std::size_t cap, PlanningEvent event) {
  if (!trace)
    return;
  if ((cap && trace->events.size() >= cap) || trace->events.size() == trace->events.max_size())
    throw std::length_error("planning: actual event cap or capacity");
  trace->events.push_back(std::move(event));
}

/// @brief Construct an actual non-operator event for the current frame.
/// @param kind Control-flow kind.
/// @param state Current fact mask.
/// @param depth Actual selected-operator count.
/// @return Fixed-schema event with next_state=state and no-operator sentinel.
PlanningEvent event_at(PlanningEventKind kind, std::uint64_t state, std::size_t depth) {
  PlanningEvent event;
  event.kind  = kind;
  event.state = event.next_state = state;
  event.depth                    = depth;
  return event;
}

/// @brief One actual DFS activation retained on the heap instead of the C++ call stack.
struct Frame {
  std::uint64_t state{};       ///< Current finite fact mask.
  std::size_t   depth{};       ///< Actual selected-operator count.
  std::size_t   next_action{}; ///< Next operator to test once entered.
  bool          entered{};     ///< Whether terminal/map checks have already passed.
  std::size_t   incoming{
      std::numeric_limits<std::size_t>::max()}; ///< Operator selected by the parent.
};

/// @brief Pop one failed frame and its selected parent operator, recording the actual backtrack.
/// @param stack Active heap frames; nonempty on entry.
/// @param run Actual ordinary output/work and selected path.
/// @param trace Optional actual event recorder.
/// @param cap Positive online event quota; zero disables quota.
/// @throws std::length_error Actual counter or event quota cannot increment.
void failed_child(
    std::vector<Frame>& stack,
    PlanningRun&        run,
    PlanningTrace*      trace,
    std::size_t         cap) {
  const auto incoming = stack.back().incoming;
  stack.pop_back();
  if (stack.empty())
    return;
  run.action_indices.pop_back();
  increment(run.work.backtracks);
  auto event = event_at(PlanningEventKind::backtrack, stack.back().state, stack.back().depth);
  event.action_index = incoming;
  capture(trace, cap, event);
}

/// @brief Execute the single ordered DFS with safe best-depth pruning and optional actual capture.
/// @param initial Starting fact mask.
/// @param goals Required positive terminal mask.
/// @param actions Operators in immutable input order.
/// @param max_steps Complete-step budget; goal is checked first at every visit.
/// @param trace Optional recorder; already owns its independent input copy.
/// @param cap Positive online actual event quota; zero disables quota.
/// @return Owned first successful path/work, or legacy failure placeholders.
/// @throws std::length_error Actual work counter, event quota or container capacity overflows.
PlanningRun execute(
    std::uint64_t                      initial,
    std::uint64_t                      goals,
    const std::vector<PlanningAction>& actions,
    std::size_t                        max_steps,
    PlanningTrace*                     trace = nullptr,
    std::size_t                        cap   = 0) {
  PlanningRun                                    out;
  std::unordered_map<std::uint64_t, std::size_t> best;
  std::vector<Frame>                             stack{{initial, 0, 0, false}};
  out.work.peak_stack = 1;
  while (!stack.empty()) {
    auto& frame = stack.back();
    if (!frame.entered) {
      increment(out.work.visits);
      capture(trace, cap, event_at(PlanningEventKind::visit, frame.state, frame.depth));
      if ((frame.state & goals) == goals) {
        capture(trace, cap, event_at(PlanningEventKind::goal, frame.state, frame.depth));
        out.result.found       = true;
        out.result.final_state = frame.state;
        for (std::size_t i = stack.size() - 1; i > 0; --i) {
          auto event = event_at(PlanningEventKind::finish, stack[i - 1].state, stack[i - 1].depth);
          event.action_index = stack[i].incoming;
          capture(trace, cap, event);
        }
        for (auto index : out.action_indices)
          out.result.actions.push_back(actions[index].name);
        return out;
      }
      if (frame.depth == max_steps) {
        increment(out.work.depth_cutoffs);
        capture(trace, cap, event_at(PlanningEventKind::cutoff, frame.state, frame.depth));
        failed_child(stack, out, trace, cap);
        continue;
      }
      const auto previous = best.find(frame.state);
      if (previous != best.end()) {
        auto event = event_at(
            previous->second <= frame.depth ? PlanningEventKind::repeat : PlanningEventKind::reopen,
            frame.state,
            frame.depth);
        event.has_previous_depth = true;
        event.previous_depth     = previous->second;
        if (previous->second <= frame.depth) {
          increment(out.work.repeat_prunes);
          capture(trace, cap, event);
          failed_child(stack, out, trace, cap);
          continue;
        }
        increment(out.work.reopenings);
        capture(trace, cap, event);
        previous->second = frame.depth;
      } else {
        best.emplace(frame.state, frame.depth);
        increment(out.work.distinct_states);
      }
      increment(out.work.expanded);
      frame.entered = true;
    }
    if (frame.next_action == actions.size()) {
      capture(trace, cap, event_at(PlanningEventKind::exhausted, frame.state, frame.depth));
      failed_child(stack, out, trace, cap);
      continue;
    }
    const auto  index  = frame.next_action++;
    const auto& action = actions[index];
    increment(out.work.operator_checks);
    auto event         = event_at(PlanningEventKind::test, frame.state, frame.depth);
    event.action_index = index;
    event.applicable   = (frame.state & action.require) == action.require;
    capture(trace, cap, event);
    if (!event.applicable)
      continue;
    increment(out.work.applicable);
    event              = event_at(PlanningEventKind::apply, frame.state, frame.depth);
    event.action_index = index;
    event.next_state   = (frame.state & ~action.remove) | action.add;
    capture(trace, cap, event);
    out.action_indices.push_back(index);
    stack.push_back({event.next_state, frame.depth + 1, 0, false, index});
    out.work.peak_stack = std::max(out.work.peak_stack, stack.size());
  }
  return out;
}
} // namespace

std::string planning_event_name(PlanningEventKind kind) {
  switch (kind) {
    case PlanningEventKind::visit:
      return "visit";
    case PlanningEventKind::goal:
      return "goal";
    case PlanningEventKind::cutoff:
      return "cutoff";
    case PlanningEventKind::repeat:
      return "repeat";
    case PlanningEventKind::reopen:
      return "reopen";
    case PlanningEventKind::test:
      return "test";
    case PlanningEventKind::apply:
      return "apply";
    case PlanningEventKind::backtrack:
      return "backtrack";
    case PlanningEventKind::finish:
      return "finish";
    case PlanningEventKind::exhausted:
      return "exhausted";
  }
  throw std::invalid_argument("planning: unknown event kind");
}
PlanningResult general_problem_solver(
    std::uint64_t                      initial,
    std::uint64_t                      goals,
    const std::vector<PlanningAction>& actions,
    std::size_t                        max_steps) {
  return execute(initial, goals, actions, max_steps).result;
}
PlanningRun general_problem_solver_run(
    std::uint64_t                      initial,
    std::uint64_t                      goals,
    const std::vector<PlanningAction>& actions,
    std::size_t                        max_steps) {
  return execute(initial, goals, actions, max_steps);
}
PlanningTrace general_problem_solver_trace(
    std::uint64_t                      initial,
    std::uint64_t                      goals,
    const std::vector<PlanningAction>& actions,
    std::size_t                        max_steps,
    std::size_t                        max_events) {
  PlanningTrace trace;
  trace.initial   = initial;
  trace.goals     = goals;
  trace.operators = actions;
  trace.run       = execute(initial, goals, trace.operators, max_steps, &trace, max_events);
  return trace;
}
PlanningCertificate check_planning_plan(
    std::uint64_t                      initial,
    std::uint64_t                      goals,
    const std::vector<PlanningAction>& actions,
    const std::vector<std::size_t>&    action_indices,
    std::size_t                        max_states) {
  for (auto index : action_indices)
    if (index >= actions.size())
      throw std::out_of_range("planning: certificate operator index");
  if (action_indices.size() == std::numeric_limits<std::size_t>::max())
    throw std::length_error("planning: certificate size overflow");
  const auto count = action_indices.size() + 1;
  if ((max_states && count > max_states) || count > std::vector<std::uint64_t>{}.max_size())
    throw std::length_error("planning: whole certificate state cap");
  PlanningCertificate out;
  out.final_state = initial;
  out.states.push_back(initial);
  for (auto index : action_indices) {
    const auto& op = actions[index];
    if ((out.final_state & op.require) != op.require) {
      out.applicable = false;
      break;
    }
    out.final_state &= ~op.remove;
    out.final_state |= op.add;
    out.states.push_back(out.final_state);
    ++out.applied;
  }
  out.reaches_goals = (out.final_state & goals) == goals;
  return out;
}
std::vector<PlanningAction> planning_binary_counter(std::size_t bits) {
  if (!bits || bits > 64)
    throw std::invalid_argument("planning: binary counter needs1..64 bits");
  std::vector<PlanningAction> actions;
  actions.reserve(bits);
  for (std::size_t k = 0; k < bits; ++k) {
    const auto bit   = std::uint64_t{1} << k;
    const auto lower = bit - 1;
    actions.push_back({"carry-" + std::to_string(k), lower, bit, lower});
  }
  return actions;
}
} // namespace datamunge::algorithms
