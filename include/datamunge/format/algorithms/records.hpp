#pragma once

/// @file records.hpp
/// @brief std::formatter specializations for the plain result, option and data structs of datamunge::algorithms.
///
/// Each struct prints on one line as `TypeName(field=value, ...)`, fields in declaration order:
/// vectors via a capped preview (`[1, 2, 3, 4, 5, ..., 99] (n=100)`), nested vectors recursively,
/// maps as `{key: value, ...}` (unordered maps sorted), strings quoted, flag bytes as 0/1.
/// Included by <datamunge/format/algorithms.hpp>.

#include <datamunge/algorithms/algorithms.hpp>
#include <datamunge/algorithms/mst_result.hpp>
#include <datamunge/format/algorithms/common.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <format>
#include <string>

// ---- adaboost.hpp ----

/// @brief Prints an algorithms::DecisionStump as one line, `DecisionStump(feature=..., threshold=..., polarity=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::DecisionStump> : datamunge::format_detail::text_formatter<datamunge::algorithms::DecisionStump> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DecisionStump& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DecisionStump", {
                   {"feature", value(v.feature)},
                   {"threshold", value(v.threshold)},
                   {"polarity", value(v.polarity)},
                   {"alpha", value(v.alpha)}});
  }
};

/// @brief Prints an algorithms::AdaBoostModel as one line, `AdaBoostModel(stumps=..., training_error=..., weak_error=...)`.
template <>
struct std::formatter<datamunge::algorithms::AdaBoostModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::AdaBoostModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AdaBoostModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AdaBoostModel", {
                   {"stumps", value(v.stumps)},
                   {"training_error", value(v.training_error)},
                   {"weak_error", value(v.weak_error)}});
  }
};

// ---- advanced_graph.hpp ----

/// @brief Prints an algorithms::MinCutResult as one line, `MinCutResult(cut_size=..., side_a=..., side_b=...)`.
template <>
struct std::formatter<datamunge::algorithms::MinCutResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::MinCutResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MinCutResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MinCutResult", {
                   {"cut_size", value(v.cut_size)},
                   {"side_a", value(v.side_a)},
                   {"side_b", value(v.side_b)}});
  }
};

/// @brief Prints an algorithms::DirectedBranchingResult as one line, `DirectedBranchingResult(total_weight=..., exists=...)`.
template <>
struct std::formatter<datamunge::algorithms::DirectedBranchingResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::DirectedBranchingResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DirectedBranchingResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DirectedBranchingResult", {
                   {"total_weight", value(v.total_weight)},
                   {"exists", value(v.exists)}});
  }
};

// ---- aho_corasick.hpp ----

/// @brief Prints an algorithms::AhoCorasickMatch as one line, `AhoCorasickMatch(position=..., pattern_index=...)`.
template <>
struct std::formatter<datamunge::algorithms::AhoCorasickMatch> : datamunge::format_detail::text_formatter<datamunge::algorithms::AhoCorasickMatch> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AhoCorasickMatch& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AhoCorasickMatch", {
                   {"position", value(v.position)},
                   {"pattern_index", value(v.pattern_index)}});
  }
};

// ---- almeida_pineda.hpp ----

/// @brief Prints an algorithms::AlmeidaPinedaNetwork as one line, `AlmeidaPinedaNetwork(num_units=..., num_inputs=..., output_units=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::AlmeidaPinedaNetwork> : datamunge::format_detail::text_formatter<datamunge::algorithms::AlmeidaPinedaNetwork> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AlmeidaPinedaNetwork& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AlmeidaPinedaNetwork", {
                   {"num_units", value(v.num_units)},
                   {"num_inputs", value(v.num_inputs)},
                   {"output_units", value(v.output_units)},
                   {"W", value(v.W)}});
  }
};

/// @brief Prints an algorithms::AlmeidaPinedaResult as one line, `AlmeidaPinedaResult(net=..., error_history=...)`.
template <>
struct std::formatter<datamunge::algorithms::AlmeidaPinedaResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::AlmeidaPinedaResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AlmeidaPinedaResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AlmeidaPinedaResult", {
                   {"net", value(v.net)},
                   {"error_history", value(v.error_history)}});
  }
};

// ---- alopex.hpp ----

/// @brief Prints an algorithms::AlopexResult as one line, `AlopexResult(best_parameters=..., best_cost=..., cost_history=...)`.
template <>
struct std::formatter<datamunge::algorithms::AlopexResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::AlopexResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AlopexResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AlopexResult", {
                   {"best_parameters", value(v.best_parameters)},
                   {"best_cost", value(v.best_cost)},
                   {"cost_history", value(v.cost_history)}});
  }
};

/// @brief Prints an algorithms::AlopexParameters as one line, `AlopexParameters(step=..., temperature=..., cooling=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::AlopexParameters> : datamunge::format_detail::text_formatter<datamunge::algorithms::AlopexParameters> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AlopexParameters& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AlopexParameters", {
                   {"step", value(v.step)},
                   {"temperature", value(v.temperature)},
                   {"cooling", value(v.cooling)},
                   {"iterations", value(v.iterations)},
                   {"seed", value(v.seed)}});
  }
};

// ---- arc_consistency.hpp ----

/// @brief Prints an algorithms::Ac3Result as one line, `Ac3Result(domains=..., consistent=...)`.
template <>
struct std::formatter<datamunge::algorithms::Ac3Result> : datamunge::format_detail::text_formatter<datamunge::algorithms::Ac3Result> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::Ac3Result& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("Ac3Result", {
                   {"domains", value(v.domains)},
                   {"consistent", value(v.consistent)}});
  }
};

// ---- arnoldi_iteration.hpp ----

/// @brief Prints an algorithms::ArnoldiResult as one line, `ArnoldiResult(V=..., H=...)`.
template <>
struct std::formatter<datamunge::algorithms::ArnoldiResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::ArnoldiResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ArnoldiResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ArnoldiResult", {
                   {"V", value(v.V)},
                   {"H", value(v.H)}});
  }
};

// ---- association_rules.hpp ----

/// @brief Prints an algorithms::FrequentItemset as one line, `FrequentItemset(items=..., support=...)`.
template <>
struct std::formatter<datamunge::algorithms::FrequentItemset> : datamunge::format_detail::text_formatter<datamunge::algorithms::FrequentItemset> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::FrequentItemset& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("FrequentItemset", {
                   {"items", value(v.items)},
                   {"support", value(v.support)}});
  }
};

/// @brief Prints an algorithms::AssociationRule as one line, `AssociationRule(antecedent=..., consequent=..., support=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::AssociationRule> : datamunge::format_detail::text_formatter<datamunge::algorithms::AssociationRule> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AssociationRule& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AssociationRule", {
                   {"antecedent", value(v.antecedent)},
                   {"consequent", value(v.consequent)},
                   {"support", value(v.support)},
                   {"confidence", value(v.confidence)},
                   {"lift", value(v.lift)}});
  }
};

// ---- automaton.hpp ----

/// @brief Prints an algorithms::NFA as one line, `NFA(num_states=..., num_symbols=..., delta=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::NFA> : datamunge::format_detail::text_formatter<datamunge::algorithms::NFA> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::NFA& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("NFA", {
                   {"num_states", value(v.num_states)},
                   {"num_symbols", value(v.num_symbols)},
                   {"delta", value(v.delta)},
                   {"start_states", value(v.start_states)},
                   {"accept", value(v.accept)}});
  }
};

/// @brief Prints an algorithms::DFA as one line, `DFA(num_states=..., num_symbols=..., delta=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::DFA> : datamunge::format_detail::text_formatter<datamunge::algorithms::DFA> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DFA& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DFA", {
                   {"num_states", value(v.num_states)},
                   {"num_symbols", value(v.num_symbols)},
                   {"delta", value(v.delta)},
                   {"start", value(v.start)},
                   {"accept", value(v.accept)}});
  }
};

// ---- bakery.hpp ----

/// @brief Prints an algorithms::BakeryTicket as one line, `BakeryTicket(number=..., id=...)`.
template <>
struct std::formatter<datamunge::algorithms::BakeryTicket> : datamunge::format_detail::text_formatter<datamunge::algorithms::BakeryTicket> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BakeryTicket& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BakeryTicket", {
                   {"number", value(v.number)},
                   {"id", value(v.id)}});
  }
};

// ---- banker.hpp ----

/// @brief Prints an algorithms::SafetyResult as one line, `SafetyResult(safe=..., sequence=...)`.
template <>
struct std::formatter<datamunge::algorithms::SafetyResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::SafetyResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SafetyResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SafetyResult", {
                   {"safe", value(v.safe)},
                   {"sequence", value(v.sequence)}});
  }
};

// ---- bellman_ford.hpp ----

/// @brief Prints an algorithms::BellmanFordResult as one line, `BellmanFordResult(distance=..., predecessor=..., has_negative_cycle=...)`.
template <>
struct std::formatter<datamunge::algorithms::BellmanFordResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BellmanFordResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BellmanFordResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BellmanFordResult", {
                   {"distance", value(v.distance)},
                   {"predecessor", value(v.predecessor)},
                   {"has_negative_cycle", value(v.has_negative_cycle)}});
  }
};

// ---- berlekamp_massey.hpp ----

/// @brief Prints an algorithms::BerlekampMasseyResult as one line, `BerlekampMasseyResult(length=..., polynomial=...)`.
template <>
struct std::formatter<datamunge::algorithms::BerlekampMasseyResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BerlekampMasseyResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BerlekampMasseyResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BerlekampMasseyResult", {
                   {"length", value(v.length)},
                   {"polynomial", value(v.polynomial)}});
  }
};

// ---- birkhoff_interpolation.hpp ----

/// @brief Prints an algorithms::BirkhoffConstraint as one line, `BirkhoffConstraint(node=..., order=..., value=...)`.
template <>
struct std::formatter<datamunge::algorithms::BirkhoffConstraint> : datamunge::format_detail::text_formatter<datamunge::algorithms::BirkhoffConstraint> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BirkhoffConstraint& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BirkhoffConstraint", {
                   {"node", value(v.node)},
                   {"order", value(v.order)},
                   {"value", value(v.value)}});
  }
};

// ---- blast.hpp ----

/// @brief Prints an algorithms::HSP as one line, `HSP(query_start=..., db_start=..., length=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::HSP> : datamunge::format_detail::text_formatter<datamunge::algorithms::HSP> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::HSP& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("HSP", {
                   {"query_start", value(v.query_start)},
                   {"db_start", value(v.db_start)},
                   {"length", value(v.length)},
                   {"score", value(v.score)}});
  }
};

// ---- blossom.hpp ----

/// @brief Prints an algorithms::GeneralMatchingResult as one line, `GeneralMatchingResult(mate=..., size=...)`.
template <>
struct std::formatter<datamunge::algorithms::GeneralMatchingResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GeneralMatchingResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GeneralMatchingResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GeneralMatchingResult", {
                   {"mate", value(v.mate)},
                   {"size", value(v.size)}});
  }
};

// ---- branch_and_bound.hpp ----

/// @brief Prints an algorithms::KnapsackResult as one line, `KnapsackResult(value=..., chosen=..., nodes_explored=...)`.
template <>
struct std::formatter<datamunge::algorithms::KnapsackResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::KnapsackResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::KnapsackResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("KnapsackResult", {
                   {"value", value(v.value)},
                   {"chosen", value(v.chosen)},
                   {"nodes_explored", value(v.nodes_explored)}});
  }
};

// ---- bully_election.hpp ----

/// @brief Prints an algorithms::BullyResult as one line, `BullyResult(leader=..., election_messages=..., answer_messages=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::BullyResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BullyResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BullyResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BullyResult", {
                   {"leader", value(v.leader)},
                   {"election_messages", value(v.election_messages)},
                   {"answer_messages", value(v.answer_messages)},
                   {"coordinator_messages", value(v.coordinator_messages)}});
  }
};

// ---- burrows_wheeler.hpp ----

/// @brief Prints an algorithms::BWTResult as one line, `BWTResult(transformed=..., index=...)`.
template <>
struct std::formatter<datamunge::algorithms::BWTResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BWTResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BWTResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BWTResult", {
                   {"transformed", value(v.transformed)},
                   {"index", value(v.index)}});
  }
};

// ---- canopy_clustering.hpp ----

/// @brief Prints an algorithms::Canopy as one line, `Canopy(center=..., members=...)`.
template <>
struct std::formatter<datamunge::algorithms::Canopy> : datamunge::format_detail::text_formatter<datamunge::algorithms::Canopy> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::Canopy& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("Canopy", {
                   {"center", value(v.center)},
                   {"members", value(v.members)}});
  }
};

// ---- cheney.hpp ----

/// @brief Prints an algorithms::CheneyResult as one line, `CheneyResult(to_space_order=..., forwarding=...)`.
template <>
struct std::formatter<datamunge::algorithms::CheneyResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::CheneyResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::CheneyResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("CheneyResult", {
                   {"to_space_order", value(v.to_space_order)},
                   {"forwarding", value(v.forwarding)}});
  }
};

// ---- chinese_remainder.hpp ----

/// @brief Prints an algorithms::CrtSolution as one line, `CrtSolution(solvable=..., remainder=..., modulus=...)`.
template <>
struct std::formatter<datamunge::algorithms::CrtSolution> : datamunge::format_detail::text_formatter<datamunge::algorithms::CrtSolution> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::CrtSolution& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("CrtSolution", {
                   {"solvable", value(v.solvable)},
                   {"remainder", value(v.remainder)},
                   {"modulus", value(v.modulus)}});
  }
};

// ---- clock_synchronization.hpp ----

/// @brief Prints an algorithms::BerkeleyResult as one line, `BerkeleyResult(synchronized_time=..., adjustments=...)`.
template <>
struct std::formatter<datamunge::algorithms::BerkeleyResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BerkeleyResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BerkeleyResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BerkeleyResult", {
                   {"synchronized_time", value(v.synchronized_time)},
                   {"adjustments", value(v.adjustments)}});
  }
};

// ---- cohen_sutherland.hpp ----

/// @brief Prints an algorithms::ClippedSegment as one line, `ClippedSegment(visible=..., x0=..., y0=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::ClippedSegment> : datamunge::format_detail::text_formatter<datamunge::algorithms::ClippedSegment> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ClippedSegment& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ClippedSegment", {
                   {"visible", value(v.visible)},
                   {"x0", value(v.x0)},
                   {"y0", value(v.y0)},
                   {"x1", value(v.x1)},
                   {"y1", value(v.y1)}});
  }
};

// ---- computus.hpp ----

/// @brief Prints an algorithms::EasterDate as one line, `EasterDate(month=..., day=...)`.
template <>
struct std::formatter<datamunge::algorithms::EasterDate> : datamunge::format_detail::text_formatter<datamunge::algorithms::EasterDate> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::EasterDate& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("EasterDate", {
                   {"month", value(v.month)},
                   {"day", value(v.day)}});
  }
};

// ---- connected_components.hpp ----

/// @brief Prints an algorithms::ConnectedComponents as one line, `ConnectedComponents(labels=..., count=...)`.
template <>
struct std::formatter<datamunge::algorithms::ConnectedComponents> : datamunge::format_detail::text_formatter<datamunge::algorithms::ConnectedComponents> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ConnectedComponents& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ConnectedComponents", {
                   {"labels", value(v.labels)},
                   {"count", value(v.count)}});
  }
};

// ---- cpu_scheduling.hpp ----

/// @brief Prints an algorithms::SchedProcess as one line, `SchedProcess(id=..., arrival=..., burst=...)`.
template <>
struct std::formatter<datamunge::algorithms::SchedProcess> : datamunge::format_detail::text_formatter<datamunge::algorithms::SchedProcess> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SchedProcess& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SchedProcess", {
                   {"id", value(v.id)},
                   {"arrival", value(v.arrival)},
                   {"burst", value(v.burst)}});
  }
};

/// @brief Prints an algorithms::SchedMetrics as one line, `SchedMetrics(id=..., completion=..., turnaround=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::SchedMetrics> : datamunge::format_detail::text_formatter<datamunge::algorithms::SchedMetrics> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SchedMetrics& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SchedMetrics", {
                   {"id", value(v.id)},
                   {"completion", value(v.completion)},
                   {"turnaround", value(v.turnaround)},
                   {"waiting", value(v.waiting)}});
  }
};

/// @brief Prints an algorithms::GanttSegment as one line, `GanttSegment(id=..., start=..., end=...)`.
template <>
struct std::formatter<datamunge::algorithms::GanttSegment> : datamunge::format_detail::text_formatter<datamunge::algorithms::GanttSegment> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GanttSegment& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GanttSegment", {
                   {"id", value(v.id)},
                   {"start", value(v.start)},
                   {"end", value(v.end)}});
  }
};

/// @brief Prints an algorithms::ScheduleResult as one line, `ScheduleResult(metrics=..., gantt=..., avg_waiting=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::ScheduleResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::ScheduleResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ScheduleResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ScheduleResult", {
                   {"metrics", value(v.metrics)},
                   {"gantt", value(v.gantt)},
                   {"avg_waiting", value(v.avg_waiting)},
                   {"avg_turnaround", value(v.avg_turnaround)}});
  }
};

// ---- cycle_detection.hpp ----

/// @brief Prints an algorithms::CycleResult as one line, `CycleResult(lambda=..., mu=...)`.
template <>
struct std::formatter<datamunge::algorithms::CycleResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::CycleResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::CycleResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("CycleResult", {
                   {"lambda", value(v.lambda)},
                   {"mu", value(v.mu)}});
  }
};

// ---- cyk_parser.hpp ----

/// @brief Prints an algorithms::CNFGrammar as one line, `CNFGrammar(start=..., unary=..., binary=...)`.
template <>
struct std::formatter<datamunge::algorithms::CNFGrammar> : datamunge::format_detail::text_formatter<datamunge::algorithms::CNFGrammar> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::CNFGrammar& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("CNFGrammar", {
                   {"start", value(v.start)},
                   {"unary", value(v.unary)},
                   {"binary", value(v.binary)}});
  }
};

// ---- decision_tree_induction.hpp ----

/// @brief Prints an algorithms::DecisionTreeNode as one line, `DecisionTreeNode(leaf=..., prediction=..., feature=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::DecisionTreeNode> : datamunge::format_detail::text_formatter<datamunge::algorithms::DecisionTreeNode> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DecisionTreeNode& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DecisionTreeNode", {
                   {"leaf", value(v.leaf)},
                   {"prediction", value(v.prediction)},
                   {"feature", value(v.feature)},
                   {"threshold", value(v.threshold)},
                   {"children", value(v.children)},
                   {"left", value(v.left)},
                   {"right", value(v.right)}});
  }
};

/// @brief Prints an algorithms::DecisionTree as one line, `DecisionTree(nodes=...)`.
template <>
struct std::formatter<datamunge::algorithms::DecisionTree> : datamunge::format_detail::text_formatter<datamunge::algorithms::DecisionTree> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DecisionTree& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DecisionTree", {
                   {"nodes", value(v.nodes)}});
  }
};

// ---- deutsch_jozsa.hpp ----

/// @brief Prints an algorithms::DeutschJozsaResult as one line, `DeutschJozsaResult(constant=..., prob_all_zero=...)`.
template <>
struct std::formatter<datamunge::algorithms::DeutschJozsaResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::DeutschJozsaResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DeutschJozsaResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DeutschJozsaResult", {
                   {"constant", value(v.constant)},
                   {"prob_all_zero", value(v.prob_all_zero)}});
  }
};

// ---- difference_map.hpp ----

/// @brief Prints an algorithms::DifferenceMapResult as one line, `DifferenceMapResult(solution=..., iterations=..., residual=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::DifferenceMapResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::DifferenceMapResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DifferenceMapResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DifferenceMapResult", {
                   {"solution", value(v.solution)},
                   {"iterations", value(v.iterations)},
                   {"residual", value(v.residual)},
                   {"converged", value(v.converged)}});
  }
};

// ---- dijkstra.hpp ----

/// @brief Prints an algorithms::ShortestPathResult as one line, `ShortestPathResult(distance=..., predecessor=...)`.
template <>
struct std::formatter<datamunge::algorithms::ShortestPathResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::ShortestPathResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ShortestPathResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ShortestPathResult", {
                   {"distance", value(v.distance)},
                   {"predecessor", value(v.predecessor)}});
  }
};

// ---- division.hpp ----

/// @brief Prints an algorithms::DivResultEx as one line, `DivResultEx(quotient=..., remainder=..., iterations=...)`.
template <>
struct std::formatter<datamunge::algorithms::DivResultEx> : datamunge::format_detail::text_formatter<datamunge::algorithms::DivResultEx> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DivResultEx& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DivResultEx", {
                   {"quotient", value(v.quotient)},
                   {"remainder", value(v.remainder)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- dpll.hpp ----

/// @brief Prints an algorithms::SatResult as one line, `SatResult(satisfiable=..., assignment=...)`.
template <>
struct std::formatter<datamunge::algorithms::SatResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::SatResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SatResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SatResult", {
                   {"satisfiable", value(v.satisfiable)},
                   {"assignment", value(v.assignment)}});
  }
};

// ---- edmonds_karp.hpp ----

/// @brief Prints an algorithms::MaxFlowResult as one line, `MaxFlowResult(max_flow=..., edge_flow=..., min_cut_source_side=...)`.
template <>
struct std::formatter<datamunge::algorithms::MaxFlowResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::MaxFlowResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MaxFlowResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MaxFlowResult", {
                   {"max_flow", value(v.max_flow)},
                   {"edge_flow", value(v.edge_flow)},
                   {"min_cut_source_side", value(v.min_cut_source_side)}});
  }
};

// ---- elementary_more.hpp ----

/// @brief Prints an algorithms::BinSplitResult as one line, `BinSplitResult(num=..., den=..., value=...)`.
template <>
struct std::formatter<datamunge::algorithms::BinSplitResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BinSplitResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BinSplitResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BinSplitResult", {
                   {"num", value(v.num)},
                   {"den", value(v.den)},
                   {"value", value(v.value)}});
  }
};

/// @brief Prints an algorithms::DivResult as one line, `DivResult(quotient=..., remainder=..., iterations=...)`.
template <>
struct std::formatter<datamunge::algorithms::DivResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::DivResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DivResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DivResult", {
                   {"quotient", value(v.quotient)},
                   {"remainder", value(v.remainder)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- ellipsoid.hpp ----

/// @brief Prints an algorithms::EllipsoidResult as one line, `EllipsoidResult(x=..., value=..., iterations=...)`.
template <>
struct std::formatter<datamunge::algorithms::EllipsoidResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::EllipsoidResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::EllipsoidResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("EllipsoidResult", {
                   {"x", value(v.x)},
                   {"value", value(v.value)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- exact_cover.hpp ----

/// @brief Prints an algorithms::ExactCoverResult as one line, `ExactCoverResult(solved=..., rows=...)`.
template <>
struct std::formatter<datamunge::algorithms::ExactCoverResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::ExactCoverResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ExactCoverResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ExactCoverResult", {
                   {"solved", value(v.solved)},
                   {"rows", value(v.rows)}});
  }
};

// ---- exchange_sorts.hpp ----

/// @brief Prints an algorithms::PancakeResult as one line, `PancakeResult(sorted=..., flips=...)`.
template <>
struct std::formatter<datamunge::algorithms::PancakeResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::PancakeResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PancakeResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PancakeResult", {
                   {"sorted", value(v.sorted)},
                   {"flips", value(v.flips)}});
  }
};

// ---- extended_euclidean.hpp ----

/// @brief Prints an algorithms::ExtendedGcd as one line, `ExtendedGcd(gcd=..., x=..., y=...)`.
template <>
struct std::formatter<datamunge::algorithms::ExtendedGcd> : datamunge::format_detail::text_formatter<datamunge::algorithms::ExtendedGcd> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ExtendedGcd& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ExtendedGcd", {
                   {"gcd", value(v.gcd)},
                   {"x", value(v.x)},
                   {"y", value(v.y)}});
  }
};

// ---- floyd_warshall.hpp ----

/// @brief Prints an algorithms::AllPairsShortestPaths as one line, `AllPairsShortestPaths(distance=..., next=..., has_negative_cycle=...)`.
template <>
struct std::formatter<datamunge::algorithms::AllPairsShortestPaths> : datamunge::format_detail::text_formatter<datamunge::algorithms::AllPairsShortestPaths> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AllPairsShortestPaths& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AllPairsShortestPaths", {
                   {"distance", value(v.distance)},
                   {"next", value(v.next)},
                   {"has_negative_cycle", value(v.has_negative_cycle)}});
  }
};

// ---- frank_wolfe.hpp ----

/// @brief Prints an algorithms::FrankWolfeResult as one line, `FrankWolfeResult(x=..., objective=...)`.
template <>
struct std::formatter<datamunge::algorithms::FrankWolfeResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::FrankWolfeResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::FrankWolfeResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("FrankWolfeResult", {
                   {"x", value(v.x)},
                   {"objective", value(v.objective)}});
  }
};

// ---- fuzzy_c_means.hpp ----

/// @brief Prints an algorithms::FuzzyCMeansResult as one line, `FuzzyCMeansResult(centers=..., membership=..., objective=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::FuzzyCMeansResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::FuzzyCMeansResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::FuzzyCMeansResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("FuzzyCMeansResult", {
                   {"centers", value(v.centers)},
                   {"membership", value(v.membership)},
                   {"objective", value(v.objective)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- game_search.hpp ----

/// @brief Prints an algorithms::GameNode as one line, `GameNode(leaf=..., value=..., children=...)`.
template <>
struct std::formatter<datamunge::algorithms::GameNode> : datamunge::format_detail::text_formatter<datamunge::algorithms::GameNode> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GameNode& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GameNode", {
                   {"leaf", value(v.leaf)},
                   {"value", value(v.value)},
                   {"children", value(v.children)}});
  }
};

/// @brief Prints an algorithms::GameResult as one line, `GameResult(value=..., leaves_evaluated=..., nodes_visited=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::GameResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GameResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GameResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GameResult", {
                   {"value", value(v.value)},
                   {"leaves_evaluated", value(v.leaves_evaluated)},
                   {"nodes_visited", value(v.nodes_visited)},
                   {"cutoffs", value(v.cutoffs)},
                   {"best_child", value(v.best_child)}});
  }
};

// ---- gauss_newton.hpp ----

/// @brief Prints an algorithms::GaussNewtonResult as one line, `GaussNewtonResult(params=..., iterations=..., sse=...)`.
template <>
struct std::formatter<datamunge::algorithms::GaussNewtonResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GaussNewtonResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GaussNewtonResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GaussNewtonResult", {
                   {"params", value(v.params)},
                   {"iterations", value(v.iterations)},
                   {"sse", value(v.sse)}});
  }
};

// ---- generational.hpp ----

/// @brief Prints an algorithms::GenerationalResult as one line, `GenerationalResult(promoted=..., young_freed=..., remembered_set=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::GenerationalResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GenerationalResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GenerationalResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GenerationalResult", {
                   {"promoted", value(v.promoted)},
                   {"young_freed", value(v.young_freed)},
                   {"remembered_set", value(v.remembered_set)},
                   {"minor_scanned", value(v.minor_scanned)},
                   {"full_scanned", value(v.full_scanned)}});
  }
};

// ---- golden_section_search.hpp ----

/// @brief Prints an algorithms::GoldenResult as one line, `GoldenResult(argmax=..., value=..., iterations=...)`.
template <>
struct std::formatter<datamunge::algorithms::GoldenResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GoldenResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GoldenResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GoldenResult", {
                   {"argmax", value(v.argmax)},
                   {"value", value(v.value)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- graph_coloring.hpp ----

/// @brief Prints an algorithms::ColoringResult as one line, `ColoringResult(color=..., num_colors=...)`.
template <>
struct std::formatter<datamunge::algorithms::ColoringResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::ColoringResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ColoringResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ColoringResult", {
                   {"color", value(v.color)},
                   {"num_colors", value(v.num_colors)}});
  }
};

// ---- graph_hard.hpp ----

/// @brief Prints an algorithms::SubgraphMatch as one line, `SubgraphMatch(found=..., mapping=...)`.
template <>
struct std::formatter<datamunge::algorithms::SubgraphMatch> : datamunge::format_detail::text_formatter<datamunge::algorithms::SubgraphMatch> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SubgraphMatch& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SubgraphMatch", {
                   {"found", value(v.found)},
                   {"mapping", value(v.mapping)}});
  }
};

// ---- graph_layout.hpp ----

/// @brief Prints an algorithms::Circle2D as one line, `Circle2D(center=..., radius=...)`.
template <>
struct std::formatter<datamunge::algorithms::Circle2D> : datamunge::format_detail::text_formatter<datamunge::algorithms::Circle2D> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::Circle2D& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("Circle2D", {
                   {"center", value(v.center)},
                   {"radius", value(v.radius)}});
  }
};

// ---- grasp.hpp ----

/// @brief Prints an algorithms::GraspMaxCutResult as one line, `GraspMaxCutResult(value=..., side=...)`.
template <>
struct std::formatter<datamunge::algorithms::GraspMaxCutResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GraspMaxCutResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GraspMaxCutResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GraspMaxCutResult", {
                   {"value", value(v.value)},
                   {"side", value(v.side)}});
  }
};

// ---- grover.hpp ----

/// @brief Prints an algorithms::GroverResult as one line, `GroverResult(measured=..., success_prob=..., iterations=...)`.
template <>
struct std::formatter<datamunge::algorithms::GroverResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GroverResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GroverResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GroverResult", {
                   {"measured", value(v.measured)},
                   {"success_prob", value(v.success_prob)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- hamming_code.hpp ----

/// @brief Prints an algorithms::Hamming74Decoded as one line, `Hamming74Decoded(data=..., error_position=..., corrected=...)`.
template <>
struct std::formatter<datamunge::algorithms::Hamming74Decoded> : datamunge::format_detail::text_formatter<datamunge::algorithms::Hamming74Decoded> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::Hamming74Decoded& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("Hamming74Decoded", {
                   {"data", value(v.data)},
                   {"error_position", value(v.error_position)},
                   {"corrected", value(v.corrected)}});
  }
};

// ---- hmm.hpp ----

/// @brief Prints an algorithms::HiddenMarkovModel as one line, `HiddenMarkovModel(initial=..., transition=..., emission=...)`.
template <>
struct std::formatter<datamunge::algorithms::HiddenMarkovModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::HiddenMarkovModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::HiddenMarkovModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("HiddenMarkovModel", {
                   {"initial", value(v.initial)},
                   {"transition", value(v.transition)},
                   {"emission", value(v.emission)}});
  }
};

/// @brief Prints an algorithms::ViterbiResult as one line, `ViterbiResult(states=..., log_probability=...)`.
template <>
struct std::formatter<datamunge::algorithms::ViterbiResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::ViterbiResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ViterbiResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ViterbiResult", {
                   {"states", value(v.states)},
                   {"log_probability", value(v.log_probability)}});
  }
};

/// @brief Prints an algorithms::ForwardBackwardResult as one line, `ForwardBackwardResult(alpha=..., beta=..., gamma=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::ForwardBackwardResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::ForwardBackwardResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ForwardBackwardResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ForwardBackwardResult", {
                   {"alpha", value(v.alpha)},
                   {"beta", value(v.beta)},
                   {"gamma", value(v.gamma)},
                   {"scale", value(v.scale)},
                   {"log_likelihood", value(v.log_likelihood)}});
  }
};

/// @brief Prints an algorithms::BaumWelchResult as one line, `BaumWelchResult(model=..., log_likelihood_history=..., iterations=...)`.
template <>
struct std::formatter<datamunge::algorithms::BaumWelchResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BaumWelchResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BaumWelchResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BaumWelchResult", {
                   {"model", value(v.model)},
                   {"log_likelihood_history", value(v.log_likelihood_history)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- hopcroft_karp.hpp ----

/// @brief Prints an algorithms::BipartiteMatchingResult as one line, `BipartiteMatchingResult(match_left=..., match_right=..., size=...)`.
template <>
struct std::formatter<datamunge::algorithms::BipartiteMatchingResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BipartiteMatchingResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BipartiteMatchingResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BipartiteMatchingResult", {
                   {"match_left", value(v.match_left)},
                   {"match_right", value(v.match_right)},
                   {"size", value(v.size)}});
  }
};

// ---- hopfield_network.hpp ----

/// @brief Prints an algorithms::HopfieldNetwork as one line, `HopfieldNetwork(weights=...)`.
template <>
struct std::formatter<datamunge::algorithms::HopfieldNetwork> : datamunge::format_detail::text_formatter<datamunge::algorithms::HopfieldNetwork> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::HopfieldNetwork& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("HopfieldNetwork", {
                   {"weights", value(v.weights)}});
  }
};

// ---- huffman_coding.hpp ----

/// @brief Prints an algorithms::HuffmanCode as one line, `HuffmanCode(codes=..., tree=..., root=...)`.
template <>
struct std::formatter<datamunge::algorithms::HuffmanCode> : datamunge::format_detail::text_formatter<datamunge::algorithms::HuffmanCode> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::HuffmanCode& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("HuffmanCode", {
                   {"codes", value(v.codes)},
                   {"tree", value(v.tree)},
                   {"root", value(v.root)}});
  }
};

/// @brief Prints an algorithms::HuffmanCode::Node as one line, `HuffmanCode::Node(symbol=..., leaf=..., left=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::HuffmanCode::Node> : datamunge::format_detail::text_formatter<datamunge::algorithms::HuffmanCode::Node> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::HuffmanCode::Node& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("HuffmanCode::Node", {
                   {"symbol", value(v.symbol)},
                   {"leaf", value(v.leaf)},
                   {"left", value(v.left)},
                   {"right", value(v.right)}});
  }
};

// ---- hungarian.hpp ----

/// @brief Prints an algorithms::AssignmentResult as one line, `AssignmentResult(assignment=..., cost=...)`.
template <>
struct std::formatter<datamunge::algorithms::AssignmentResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::AssignmentResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AssignmentResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AssignmentResult", {
                   {"assignment", value(v.assignment)},
                   {"cost", value(v.cost)}});
  }
};

// ---- index_calculus.hpp ----

/// @brief Prints an algorithms::DiscreteLogResult as one line, `DiscreteLogResult(x=..., found=...)`.
template <>
struct std::formatter<datamunge::algorithms::DiscreteLogResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::DiscreteLogResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DiscreteLogResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DiscreteLogResult", {
                   {"x", value(v.x)},
                   {"found", value(v.found)}});
  }
};

// ---- interpolation.hpp ----

/// @brief Prints an algorithms::CubicSpline as one line, `CubicSpline(xs=..., a=..., b=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::CubicSpline> : datamunge::format_detail::text_formatter<datamunge::algorithms::CubicSpline> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::CubicSpline& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("CubicSpline", {
                   {"xs", value(v.xs)},
                   {"a", value(v.a)},
                   {"b", value(v.b)},
                   {"c", value(v.c)},
                   {"d", value(v.d)}});
  }
};

// ---- k_medoids.hpp ----

/// @brief Prints an algorithms::KMedoidsResult as one line, `KMedoidsResult(medoids=..., assignment=..., total_cost=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::KMedoidsResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::KMedoidsResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::KMedoidsResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("KMedoidsResult", {
                   {"medoids", value(v.medoids)},
                   {"assignment", value(v.assignment)},
                   {"total_cost", value(v.total_cost)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- kabsch.hpp ----

/// @brief Prints an algorithms::KabschResult as one line, `KabschResult(rotation=..., translation=..., rmsd=...)`.
template <>
struct std::formatter<datamunge::algorithms::KabschResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::KabschResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::KabschResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("KabschResult", {
                   {"rotation", value(v.rotation)},
                   {"translation", value(v.translation)},
                   {"rmsd", value(v.rmsd)}});
  }
};

// ---- knuth_bendix.hpp ----

/// @brief Prints an algorithms::RewriteRule as one line, `RewriteRule(lhs=..., rhs=...)`.
template <>
struct std::formatter<datamunge::algorithms::RewriteRule> : datamunge::format_detail::text_formatter<datamunge::algorithms::RewriteRule> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RewriteRule& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RewriteRule", {
                   {"lhs", value(v.lhs)},
                   {"rhs", value(v.rhs)}});
  }
};

/// @brief Prints an algorithms::KnuthBendixResult as one line, `KnuthBendixResult(rules=..., complete=...)`.
template <>
struct std::formatter<datamunge::algorithms::KnuthBendixResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::KnuthBendixResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::KnuthBendixResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("KnuthBendixResult", {
                   {"rules", value(v.rules)},
                   {"complete", value(v.complete)}});
  }
};

// ---- lanczos_iteration.hpp ----

/// @brief Prints an algorithms::LanczosResult as one line, `LanczosResult(alpha=..., beta=..., V=...)`.
template <>
struct std::formatter<datamunge::algorithms::LanczosResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::LanczosResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LanczosResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("LanczosResult", {
                   {"alpha", value(v.alpha)},
                   {"beta", value(v.beta)},
                   {"V", value(v.V)}});
  }
};

// ---- linde_buzo_gray.hpp ----

/// @brief Prints an algorithms::LBGResult as one line, `LBGResult(codebook=..., assignment=..., distortion=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::LBGResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::LBGResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LBGResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("LBGResult", {
                   {"codebook", value(v.codebook)},
                   {"assignment", value(v.assignment)},
                   {"distortion", value(v.distortion)},
                   {"distortion_history", value(v.distortion_history)}});
  }
};

// ---- linear_algebra.hpp ----

/// @brief Prints an algorithms::EigenPair as one line, `EigenPair(value=..., vector=...)`.
template <>
struct std::formatter<datamunge::algorithms::EigenPair> : datamunge::format_detail::text_formatter<datamunge::algorithms::EigenPair> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::EigenPair& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("EigenPair", {
                   {"value", value(v.value)},
                   {"vector", value(v.vector)}});
  }
};

// ---- linear_solvers.hpp ----

/// @brief Prints an algorithms::LinearSolution as one line, `LinearSolution(solved=..., x=...)`.
template <>
struct std::formatter<datamunge::algorithms::LinearSolution> : datamunge::format_detail::text_formatter<datamunge::algorithms::LinearSolution> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LinearSolution& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("LinearSolution", {
                   {"solved", value(v.solved)},
                   {"x", value(v.x)}});
  }
};

/// @brief Prints an algorithms::IterativeSolution as one line, `IterativeSolution(converged=..., x=..., iterations=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::IterativeSolution> : datamunge::format_detail::text_formatter<datamunge::algorithms::IterativeSolution> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::IterativeSolution& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("IterativeSolution", {
                   {"converged", value(v.converged)},
                   {"x", value(v.x)},
                   {"iterations", value(v.iterations)},
                   {"residual", value(v.residual)}});
  }
};

/// @brief Prints an algorithms::GaussJordanResult as one line, `GaussJordanResult(solved=..., x=..., inverse=...)`.
template <>
struct std::formatter<datamunge::algorithms::GaussJordanResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GaussJordanResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GaussJordanResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GaussJordanResult", {
                   {"solved", value(v.solved)},
                   {"x", value(v.x)},
                   {"inverse", value(v.inverse)}});
  }
};

// ---- logitboost.hpp ----

/// @brief Prints an algorithms::RegressionStump as one line, `RegressionStump(feature=..., threshold=..., left_value=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::RegressionStump> : datamunge::format_detail::text_formatter<datamunge::algorithms::RegressionStump> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RegressionStump& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RegressionStump", {
                   {"feature", value(v.feature)},
                   {"threshold", value(v.threshold)},
                   {"left_value", value(v.left_value)},
                   {"right_value", value(v.right_value)}});
  }
};

/// @brief Prints an algorithms::LogitBoostModel as one line, `LogitBoostModel(stumps=..., log_likelihood=...)`.
template <>
struct std::formatter<datamunge::algorithms::LogitBoostModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::LogitBoostModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LogitBoostModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("LogitBoostModel", {
                   {"stumps", value(v.stumps)},
                   {"log_likelihood", value(v.log_likelihood)}});
  }
};

// ---- lz77.hpp ----

/// @brief Prints an algorithms::LZ77Token as one line, `LZ77Token(offset=..., length=..., next=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::LZ77Token> : datamunge::format_detail::text_formatter<datamunge::algorithms::LZ77Token> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LZ77Token& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("LZ77Token", {
                   {"offset", value(v.offset)},
                   {"length", value(v.length)},
                   {"next", value(v.next)},
                   {"literal", value(v.literal)}});
  }
};

// ---- manacher.hpp ----

/// @brief Prints an algorithms::PalindromeResult as one line, `PalindromeResult(start=..., length=...)`.
template <>
struct std::formatter<datamunge::algorithms::PalindromeResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::PalindromeResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PalindromeResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PalindromeResult", {
                   {"start", value(v.start)},
                   {"length", value(v.length)}});
  }
};

// ---- mark_compact.hpp ----

/// @brief Prints an algorithms::MarkCompactResult as one line, `MarkCompactResult(marked=..., new_address=..., order=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::MarkCompactResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::MarkCompactResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MarkCompactResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MarkCompactResult", {
                   {"marked", value(v.marked)},
                   {"new_address", value(v.new_address)},
                   {"order", value(v.order)},
                   {"updated_refs", value(v.updated_refs)}});
  }
};

// ---- mark_sweep.hpp ----

/// @brief Prints an algorithms::MarkSweepResult as one line, `MarkSweepResult(marked=..., live=..., freed=...)`.
template <>
struct std::formatter<datamunge::algorithms::MarkSweepResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::MarkSweepResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MarkSweepResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MarkSweepResult", {
                   {"marked", value(v.marked)},
                   {"live", value(v.live)},
                   {"freed", value(v.freed)}});
  }
};

// ---- matrix_chain.hpp ----

/// @brief Prints an algorithms::MatrixChainResult as one line, `MatrixChainResult(cost=..., parenthesization=...)`.
template <>
struct std::formatter<datamunge::algorithms::MatrixChainResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::MatrixChainResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MatrixChainResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MatrixChainResult", {
                   {"cost", value(v.cost)},
                   {"parenthesization", value(v.parenthesization)}});
  }
};

// ---- maximum_parsimony.hpp ----

/// @brief Prints an algorithms::ParsimonyTree as one line, `ParsimonyTree(root=..., children=...)`.
template <>
struct std::formatter<datamunge::algorithms::ParsimonyTree> : datamunge::format_detail::text_formatter<datamunge::algorithms::ParsimonyTree> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ParsimonyTree& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ParsimonyTree", {
                   {"root", value(v.root)},
                   {"children", value(v.children)}});
  }
};

// ---- memetic.hpp ----

/// @brief Prints an algorithms::MemeticMaxCutResult as one line, `MemeticMaxCutResult(value=..., side=...)`.
template <>
struct std::formatter<datamunge::algorithms::MemeticMaxCutResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::MemeticMaxCutResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MemeticMaxCutResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MemeticMaxCutResult", {
                   {"value", value(v.value)},
                   {"side", value(v.side)}});
  }
};

// ---- min_conflicts.hpp ----

/// @brief Prints an algorithms::NQueensResult as one line, `NQueensResult(queens=..., solved=..., steps=...)`.
template <>
struct std::formatter<datamunge::algorithms::NQueensResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::NQueensResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::NQueensResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("NQueensResult", {
                   {"queens", value(v.queens)},
                   {"solved", value(v.solved)},
                   {"steps", value(v.steps)}});
  }
};

// ---- miser.hpp ----

/// @brief Prints an algorithms::MiserResult as one line, `MiserResult(value=..., variance=...)`.
template <>
struct std::formatter<datamunge::algorithms::MiserResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::MiserResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MiserResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MiserResult", {
                   {"value", value(v.value)},
                   {"variance", value(v.variance)}});
  }
};

// ---- mst_result.hpp ----

/// @brief Prints an algorithms::MinimumSpanningTree as one line, `MinimumSpanningTree(edges=..., total_weight=..., is_connected=...)`.
template <>
struct std::formatter<datamunge::algorithms::MinimumSpanningTree> : datamunge::format_detail::text_formatter<datamunge::algorithms::MinimumSpanningTree> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MinimumSpanningTree& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MinimumSpanningTree", {
                   {"edges", value(v.edges)},
                   {"total_weight", value(v.total_weight)},
                   {"is_connected", value(v.is_connected)}});
  }
};

// ---- multiplication.hpp ----

/// @brief Prints an algorithms::BoothResult as one line, `BoothResult(product=..., additions=..., subtractions=...)`.
template <>
struct std::formatter<datamunge::algorithms::BoothResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BoothResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BoothResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BoothResult", {
                   {"product", value(v.product)},
                   {"additions", value(v.additions)},
                   {"subtractions", value(v.subtractions)}});
  }
};

// ---- nested_sampling.hpp ----

/// @brief Prints an algorithms::NestedSamplingResult as one line, `NestedSamplingResult(log_evidence=..., log_evidence_error=..., information=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::NestedSamplingResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::NestedSamplingResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::NestedSamplingResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("NestedSamplingResult", {
                   {"log_evidence", value(v.log_evidence)},
                   {"log_evidence_error", value(v.log_evidence_error)},
                   {"information", value(v.information)},
                   {"iterations", value(v.iterations)},
                   {"posterior_samples", value(v.posterior_samples)},
                   {"posterior_log_weights", value(v.posterior_log_weights)}});
  }
};

// ---- nesting.hpp ----

/// @brief Prints an algorithms::NestingResult as one line, `NestingResult(bins=..., assignment=..., loads=...)`.
template <>
struct std::formatter<datamunge::algorithms::NestingResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::NestingResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::NestingResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("NestingResult", {
                   {"bins", value(v.bins)},
                   {"assignment", value(v.assignment)},
                   {"loads", value(v.loads)}});
  }
};

// ---- network_analysis.hpp ----

/// @brief Prints an algorithms::HITSResult as one line, `HITSResult(hubs=..., authorities=..., iterations=...)`.
template <>
struct std::formatter<datamunge::algorithms::HITSResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::HITSResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::HITSResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("HITSResult", {
                   {"hubs", value(v.hubs)},
                   {"authorities", value(v.authorities)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- number_theory.hpp ----

/// @brief Prints an algorithms::ModSqrt as one line, `ModSqrt(exists=..., root=...)`.
template <>
struct std::formatter<datamunge::algorithms::ModSqrt> : datamunge::format_detail::text_formatter<datamunge::algorithms::ModSqrt> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ModSqrt& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ModSqrt", {
                   {"exists", value(v.exists)},
                   {"root", value(v.root)}});
  }
};

/// @brief Prints an algorithms::DiscreteLog as one line, `DiscreteLog(exists=..., exponent=...)`.
template <>
struct std::formatter<datamunge::algorithms::DiscreteLog> : datamunge::format_detail::text_formatter<datamunge::algorithms::DiscreteLog> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DiscreteLog& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DiscreteLog", {
                   {"exists", value(v.exists)},
                   {"exponent", value(v.exponent)}});
  }
};

// ---- number_theory_advanced.hpp ----

/// @brief Prints an algorithms::PellSolution as one line, `PellSolution(x=..., y=..., found=...)`.
template <>
struct std::formatter<datamunge::algorithms::PellSolution> : datamunge::format_detail::text_formatter<datamunge::algorithms::PellSolution> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PellSolution& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PellSolution", {
                   {"x", value(v.x)},
                   {"y", value(v.y)},
                   {"found", value(v.found)}});
  }
};

// ---- object_graph.hpp ----

/// @brief Prints an algorithms::ObjectGraph as one line, `ObjectGraph(num_objects=..., refs=..., roots=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::ObjectGraph> : datamunge::format_detail::text_formatter<datamunge::algorithms::ObjectGraph> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ObjectGraph& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ObjectGraph", {
                   {"num_objects", value(v.num_objects)},
                   {"refs", value(v.refs)},
                   {"roots", value(v.roots)},
                   {"sizes", value(v.sizes)}});
  }
};

// ---- odds_algorithm.hpp ----

/// @brief Prints an algorithms::OddsResult as one line, `OddsResult(threshold=..., win_probability=...)`.
template <>
struct std::formatter<datamunge::algorithms::OddsResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::OddsResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::OddsResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("OddsResult", {
                   {"threshold", value(v.threshold)},
                   {"win_probability", value(v.win_probability)}});
  }
};

// ---- ode.hpp ----

/// @brief Prints an algorithms::OdeSolution as one line, `OdeSolution(t=..., y=...)`.
template <>
struct std::formatter<datamunge::algorithms::OdeSolution> : datamunge::format_detail::text_formatter<datamunge::algorithms::OdeSolution> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::OdeSolution& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("OdeSolution", {
                   {"t", value(v.t)},
                   {"y", value(v.y)}});
  }
};

/// @brief Prints an algorithms::VerletSolution as one line, `VerletSolution(t=..., x=..., v=...)`.
template <>
struct std::formatter<datamunge::algorithms::VerletSolution> : datamunge::format_detail::text_formatter<datamunge::algorithms::VerletSolution> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::VerletSolution& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("VerletSolution", {
                   {"t", value(v.t)},
                   {"x", value(v.x)},
                   {"v", value(v.v)}});
  }
};

// ---- online_linear_classifiers.hpp ----

/// @brief Prints an algorithms::PerceptronModel as one line, `PerceptronModel(weights=..., bias=..., epochs_run=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::PerceptronModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::PerceptronModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PerceptronModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PerceptronModel", {
                   {"weights", value(v.weights)},
                   {"bias", value(v.bias)},
                   {"epochs_run", value(v.epochs_run)},
                   {"converged", value(v.converged)},
                   {"total_mistakes", value(v.total_mistakes)}});
  }
};

/// @brief Prints an algorithms::WinnowModel as one line, `WinnowModel(weights=..., threshold=..., alpha=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::WinnowModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::WinnowModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::WinnowModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("WinnowModel", {
                   {"weights", value(v.weights)},
                   {"threshold", value(v.threshold)},
                   {"alpha", value(v.alpha)},
                   {"epochs_run", value(v.epochs_run)},
                   {"converged", value(v.converged)},
                   {"total_mistakes", value(v.total_mistakes)}});
  }
};

// ---- optics.hpp ----

/// @brief Prints an algorithms::OpticsResult as one line, `OpticsResult(ordering=..., reachability=..., core_distance=...)`.
template <>
struct std::formatter<datamunge::algorithms::OpticsResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::OpticsResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::OpticsResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("OpticsResult", {
                   {"ordering", value(v.ordering)},
                   {"reachability", value(v.reachability)},
                   {"core_distance", value(v.core_distance)}});
  }
};

// ---- packrat_parser.hpp ----

/// @brief Prints an algorithms::PegExpr as one line, `PegExpr(op=..., text=..., children=...)`.
template <>
struct std::formatter<datamunge::algorithms::PegExpr> : datamunge::format_detail::text_formatter<datamunge::algorithms::PegExpr> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PegExpr& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PegExpr", {
                   {"op", value(v.op)},
                   {"text", value(v.text)},
                   {"children", value(v.children)}});
  }
};

// ---- painters_algorithm.hpp ----

/// @brief Prints an algorithms::DepthRect as one line, `DepthRect(x0=..., y0=..., x1=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::DepthRect> : datamunge::format_detail::text_formatter<datamunge::algorithms::DepthRect> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::DepthRect& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("DepthRect", {
                   {"x0", value(v.x0)},
                   {"y0", value(v.y0)},
                   {"x1", value(v.x1)},
                   {"y1", value(v.y1)},
                   {"depth", value(v.depth)},
                   {"color", value(v.color)}});
  }
};

// ---- pcnn.hpp ----

/// @brief Prints an algorithms::PCNNResult as one line, `PCNNResult(pulses=..., time_signature=..., first_fire=...)`.
template <>
struct std::formatter<datamunge::algorithms::PCNNResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::PCNNResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PCNNResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PCNNResult", {
                   {"pulses", value(v.pulses)},
                   {"time_signature", value(v.time_signature)},
                   {"first_fire", value(v.first_fire)}});
  }
};

/// @brief Prints an algorithms::PCNNParameters as one line, `PCNNParameters(iterations=..., beta=..., threshold_decay=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::PCNNParameters> : datamunge::format_detail::text_formatter<datamunge::algorithms::PCNNParameters> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PCNNParameters& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PCNNParameters", {
                   {"iterations", value(v.iterations)},
                   {"beta", value(v.beta)},
                   {"threshold_decay", value(v.threshold_decay)},
                   {"threshold_gain", value(v.threshold_gain)},
                   {"initial_threshold", value(v.initial_threshold)}});
  }
};

// ---- permutations.hpp ----

/// @brief Prints an algorithms::YoungTableaux as one line, `YoungTableaux(p=..., q=...)`.
template <>
struct std::formatter<datamunge::algorithms::YoungTableaux> : datamunge::format_detail::text_formatter<datamunge::algorithms::YoungTableaux> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::YoungTableaux& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("YoungTableaux", {
                   {"p", value(v.p)},
                   {"q", value(v.q)}});
  }
};

// ---- petrick.hpp ----

/// @brief Prints an algorithms::PetrickResult as one line, `PetrickResult(minimal_covers=...)`.
template <>
struct std::formatter<datamunge::algorithms::PetrickResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::PetrickResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PetrickResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PetrickResult", {
                   {"minimal_covers", value(v.minimal_covers)}});
  }
};

// ---- phase_estimation.hpp ----

/// @brief Prints an algorithms::PhaseEstimationResult as one line, `PhaseEstimationResult(estimate=..., measured=..., probability=...)`.
template <>
struct std::formatter<datamunge::algorithms::PhaseEstimationResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::PhaseEstimationResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PhaseEstimationResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PhaseEstimationResult", {
                   {"estimate", value(v.estimate)},
                   {"measured", value(v.measured)},
                   {"probability", value(v.probability)}});
  }
};

// ---- phonetic.hpp ----

/// @brief Prints an algorithms::MatchRatingResult as one line, `MatchRatingResult(comparable=..., similarity=..., min_rating=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::MatchRatingResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::MatchRatingResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MatchRatingResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MatchRatingResult", {
                   {"comparable", value(v.comparable)},
                   {"similarity", value(v.similarity)},
                   {"min_rating", value(v.min_rating)},
                   {"match", value(v.match)}});
  }
};

// ---- pi_and_cordic.hpp ----

/// @brief Prints an algorithms::SinCos as one line, `SinCos(sin=..., cos=...)`.
template <>
struct std::formatter<datamunge::algorithms::SinCos> : datamunge::format_detail::text_formatter<datamunge::algorithms::SinCos> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SinCos& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SinCos", {
                   {"sin", value(v.sin)},
                   {"cos", value(v.cos)}});
  }
};

// ---- pollard_kangaroo.hpp ----

/// @brief Prints an algorithms::KangarooResult as one line, `KangarooResult(x=..., found=..., jumps=...)`.
template <>
struct std::formatter<datamunge::algorithms::KangarooResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::KangarooResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::KangarooResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("KangarooResult", {
                   {"x", value(v.x)},
                   {"found", value(v.found)},
                   {"jumps", value(v.jumps)}});
  }
};

// ---- polynomial_gf.hpp ----

/// @brief Prints an algorithms::PolyDivMod as one line, `PolyDivMod(quotient=..., remainder=...)`.
template <>
struct std::formatter<datamunge::algorithms::PolyDivMod> : datamunge::format_detail::text_formatter<datamunge::algorithms::PolyDivMod> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PolyDivMod& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PolyDivMod", {
                   {"quotient", value(v.quotient)},
                   {"remainder", value(v.remainder)}});
  }
};

// ---- quine_mccluskey.hpp ----

/// @brief Prints an algorithms::Implicant as one line, `Implicant(bits=..., mask=...)`.
template <>
struct std::formatter<datamunge::algorithms::Implicant> : datamunge::format_detail::text_formatter<datamunge::algorithms::Implicant> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::Implicant& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("Implicant", {
                   {"bits", value(v.bits)},
                   {"mask", value(v.mask)}});
  }
};

/// @brief Prints an algorithms::QuineMcCluskeyResult as one line, `QuineMcCluskeyResult(prime_implicants=..., essential=..., minimal_cover=...)`.
template <>
struct std::formatter<datamunge::algorithms::QuineMcCluskeyResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::QuineMcCluskeyResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::QuineMcCluskeyResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("QuineMcCluskeyResult", {
                   {"prime_implicants", value(v.prime_implicants)},
                   {"essential", value(v.essential)},
                   {"minimal_cover", value(v.minimal_cover)}});
  }
};

// ---- random_restart_hill_climbing.hpp ----

/// @brief Prints an algorithms::HillClimbResult as one line, `HillClimbResult(x=..., value=...)`.
template <>
struct std::formatter<datamunge::algorithms::HillClimbResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::HillClimbResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::HillClimbResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("HillClimbResult", {
                   {"x", value(v.x)},
                   {"value", value(v.value)}});
  }
};

// ---- ransac.hpp ----

/// @brief Prints an algorithms::RansacLine as one line, `RansacLine(a=..., b=..., c=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::RansacLine> : datamunge::format_detail::text_formatter<datamunge::algorithms::RansacLine> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RansacLine& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RansacLine", {
                   {"a", value(v.a)},
                   {"b", value(v.b)},
                   {"c", value(v.c)},
                   {"inliers", value(v.inliers)},
                   {"inlier_count", value(v.inlier_count)},
                   {"trials", value(v.trials)}});
  }
};

// ---- rbf_network.hpp ----

/// @brief Prints an algorithms::RBFNetwork as one line, `RBFNetwork(centers=..., width=..., weights=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::RBFNetwork> : datamunge::format_detail::text_formatter<datamunge::algorithms::RBFNetwork> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RBFNetwork& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RBFNetwork", {
                   {"centers", value(v.centers)},
                   {"width", value(v.width)},
                   {"weights", value(v.weights)},
                   {"bias", value(v.bias)}});
  }
};

// ---- realtime_scheduling.hpp ----

/// @brief Prints an algorithms::RTTask as one line, `RTTask(id=..., period=..., wcet=...)`.
template <>
struct std::formatter<datamunge::algorithms::RTTask> : datamunge::format_detail::text_formatter<datamunge::algorithms::RTTask> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RTTask& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RTTask", {
                   {"id", value(v.id)},
                   {"period", value(v.period)},
                   {"wcet", value(v.wcet)}});
  }
};

/// @brief Prints an algorithms::RTSlice as one line, `RTSlice(id=..., start=..., end=...)`.
template <>
struct std::formatter<datamunge::algorithms::RTSlice> : datamunge::format_detail::text_formatter<datamunge::algorithms::RTSlice> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RTSlice& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RTSlice", {
                   {"id", value(v.id)},
                   {"start", value(v.start)},
                   {"end", value(v.end)}});
  }
};

/// @brief Prints an algorithms::RTScheduleResult as one line, `RTScheduleResult(timeline=..., deadline_missed=...)`.
template <>
struct std::formatter<datamunge::algorithms::RTScheduleResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::RTScheduleResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RTScheduleResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RTScheduleResult", {
                   {"timeline", value(v.timeline)},
                   {"deadline_missed", value(v.deadline_missed)}});
  }
};

// ---- reference_counting.hpp ----

/// @brief Prints an algorithms::RefCountResult as one line, `RefCountResult(refcount=..., freed=..., leaked=...)`.
template <>
struct std::formatter<datamunge::algorithms::RefCountResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::RefCountResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RefCountResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RefCountResult", {
                   {"refcount", value(v.refcount)},
                   {"freed", value(v.freed)},
                   {"leaked", value(v.leaked)}});
  }
};

// ---- root_finding.hpp ----

/// @brief Prints an algorithms::RootResult as one line, `RootResult(converged=..., root=..., iterations=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::RootResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::RootResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RootResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RootResult", {
                   {"converged", value(v.converged)},
                   {"root", value(v.root)},
                   {"iterations", value(v.iterations)},
                   {"residual", value(v.residual)}});
  }
};

// ---- routing_batch.hpp ----

/// @brief Prints an algorithms::LongestPathResult as one line, `LongestPathResult(distance=..., predecessor=...)`.
template <>
struct std::formatter<datamunge::algorithms::LongestPathResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::LongestPathResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::LongestPathResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("LongestPathResult", {
                   {"distance", value(v.distance)},
                   {"predecessor", value(v.predecessor)}});
  }
};

/// @brief Prints an algorithms::JohnsonAllPairsResult as one line, `JohnsonAllPairsResult(distance=..., has_negative_cycle=...)`.
template <>
struct std::formatter<datamunge::algorithms::JohnsonAllPairsResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::JohnsonAllPairsResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::JohnsonAllPairsResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("JohnsonAllPairsResult", {
                   {"distance", value(v.distance)},
                   {"has_negative_cycle", value(v.has_negative_cycle)}});
  }
};

// ---- rule_learners.hpp ----

/// @brief Prints an algorithms::ZeroRModel as one line, `ZeroRModel(prediction=..., majority_count=..., total=...)`.
template <>
struct std::formatter<datamunge::algorithms::ZeroRModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::ZeroRModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ZeroRModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ZeroRModel", {
                   {"prediction", value(v.prediction)},
                   {"majority_count", value(v.majority_count)},
                   {"total", value(v.total)},
                   {"accuracy", value(v.accuracy())}});
  }
};

/// @brief Prints an algorithms::OneRModel as one line, `OneRModel(feature=..., value_to_class=..., default_class=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::OneRModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::OneRModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::OneRModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("OneRModel", {
                   {"feature", value(v.feature)},
                   {"value_to_class", value(v.value_to_class)},
                   {"default_class", value(v.default_class)},
                   {"errors", value(v.errors)},
                   {"total", value(v.total)},
                   {"accuracy", value(v.accuracy())}});
  }
};

// ---- run_length_encoding.hpp ----

/// @brief Prints an algorithms::Run as one line, `Run(symbol=..., count=...)`.
template <>
struct std::formatter<datamunge::algorithms::Run> : datamunge::format_detail::text_formatter<datamunge::algorithms::Run> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::Run& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("Run", {
                   {"symbol", value(v.symbol)},
                   {"count", value(v.count)}});
  }
};

// ---- rvm.hpp ----

/// @brief Prints an algorithms::RVMModel as one line, `RVMModel(centers=..., weights=..., relevant=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::RVMModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::RVMModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RVMModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RVMModel", {
                   {"centers", value(v.centers)},
                   {"weights", value(v.weights)},
                   {"relevant", value(v.relevant)},
                   {"gamma", value(v.gamma)},
                   {"beta", value(v.beta)},
                   {"iterations", value(v.iterations)},
                   {"num_relevance_vectors", value(v.num_relevance_vectors)}});
  }
};

// ---- schreier_sims.hpp ----

/// @brief Prints an algorithms::BSGS as one line, `BSGS(n=..., base=..., transversal=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::BSGS> : datamunge::format_detail::text_formatter<datamunge::algorithms::BSGS> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BSGS& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BSGS", {
                   {"n", value(v.n)},
                   {"base", value(v.base)},
                   {"transversal", value(v.transversal)},
                   {"order", value(v.order)}});
  }
};

// ---- search_batch.hpp ----

/// @brief Prints an algorithms::UnweightedSearchResult as one line, `UnweightedSearchResult(path=..., order=..., found=...)`.
template <>
struct std::formatter<datamunge::algorithms::UnweightedSearchResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::UnweightedSearchResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::UnweightedSearchResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("UnweightedSearchResult", {
                   {"path", value(v.path)},
                   {"order", value(v.order)},
                   {"found", value(v.found)}});
  }
};

// ---- search_more.hpp ----

/// @brief Prints an algorithms::BasicPathResult as one line, `BasicPathResult(path=..., expanded=..., found=...)`.
template <>
struct std::formatter<datamunge::algorithms::BasicPathResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::BasicPathResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::BasicPathResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("BasicPathResult", {
                   {"path", value(v.path)},
                   {"expanded", value(v.expanded)},
                   {"found", value(v.found)}});
  }
};

/// @brief Prints an algorithms::GridPathResult as one line, `GridPathResult(path=..., cost=..., found=...)`.
template <>
struct std::formatter<datamunge::algorithms::GridPathResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::GridPathResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::GridPathResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("GridPathResult", {
                   {"path", value(v.path)},
                   {"cost", value(v.cost)},
                   {"found", value(v.found)}});
  }
};

/// @brief Prints an algorithms::PlanningAction as one line, `PlanningAction(name=..., require=..., add=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::PlanningAction> : datamunge::format_detail::text_formatter<datamunge::algorithms::PlanningAction> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PlanningAction& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PlanningAction", {
                   {"name", value(v.name)},
                   {"require", value(v.require)},
                   {"add", value(v.add)},
                   {"remove", value(v.remove)}});
  }
};

/// @brief Prints an algorithms::PlanningResult as one line, `PlanningResult(actions=..., final_state=..., found=...)`.
template <>
struct std::formatter<datamunge::algorithms::PlanningResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::PlanningResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::PlanningResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("PlanningResult", {
                   {"actions", value(v.actions)},
                   {"final_state", value(v.final_state)},
                   {"found", value(v.found)}});
  }
};

// ---- self_organizing_map.hpp ----

/// @brief Prints an algorithms::SelfOrganizingMap as one line, `SelfOrganizingMap(rows=..., cols=..., dim=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::SelfOrganizingMap> : datamunge::format_detail::text_formatter<datamunge::algorithms::SelfOrganizingMap> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SelfOrganizingMap& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SelfOrganizingMap", {
                   {"rows", value(v.rows)},
                   {"cols", value(v.cols)},
                   {"dim", value(v.dim)},
                   {"weights", value(v.weights)}});
  }
};

/// @brief Prints an algorithms::SOMParameters as one line, `SOMParameters(rows=..., cols=..., iterations=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::SOMParameters> : datamunge::format_detail::text_formatter<datamunge::algorithms::SOMParameters> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SOMParameters& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SOMParameters", {
                   {"rows", value(v.rows)},
                   {"cols", value(v.cols)},
                   {"iterations", value(v.iterations)},
                   {"learning_rate", value(v.learning_rate)},
                   {"radius", value(v.radius)},
                   {"seed", value(v.seed)}});
  }
};

// ---- sequence_alignment.hpp ----

/// @brief Prints an algorithms::AlignmentScoring as one line, `AlignmentScoring(match=..., mismatch=..., gap=...)`.
template <>
struct std::formatter<datamunge::algorithms::AlignmentScoring> : datamunge::format_detail::text_formatter<datamunge::algorithms::AlignmentScoring> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::AlignmentScoring& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("AlignmentScoring", {
                   {"match", value(v.match)},
                   {"mismatch", value(v.mismatch)},
                   {"gap", value(v.gap)}});
  }
};

/// @brief Prints an algorithms::Alignment as one line, `Alignment(score=..., a_aligned=..., b_aligned=...)`.
template <>
struct std::formatter<datamunge::algorithms::Alignment> : datamunge::format_detail::text_formatter<datamunge::algorithms::Alignment> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::Alignment& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("Alignment", {
                   {"score", value(v.score)},
                   {"a_aligned", value(v.a_aligned)},
                   {"b_aligned", value(v.b_aligned)}});
  }
};

// ---- shamir_secret_sharing.hpp ----

/// @brief Prints an algorithms::SecretShare as one line, `SecretShare(x=..., y=...)`.
template <>
struct std::formatter<datamunge::algorithms::SecretShare> : datamunge::format_detail::text_formatter<datamunge::algorithms::SecretShare> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SecretShare& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SecretShare", {
                   {"x", value(v.x)},
                   {"y", value(v.y)}});
  }
};

// ---- shannon_fano.hpp ----

/// @brief Prints an algorithms::ShannonFanoCode as one line, `ShannonFanoCode(codes=...)`.
template <>
struct std::formatter<datamunge::algorithms::ShannonFanoCode> : datamunge::format_detail::text_formatter<datamunge::algorithms::ShannonFanoCode> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ShannonFanoCode& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ShannonFanoCode", {
                   {"codes", value(v.codes)}});
  }
};

// ---- signed_reversals.hpp ----

/// @brief Prints an algorithms::SignedReversalResult as one line, `SignedReversalResult(distance=..., reversals=...)`.
template <>
struct std::formatter<datamunge::algorithms::SignedReversalResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::SignedReversalResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SignedReversalResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SignedReversalResult", {
                   {"distance", value(v.distance)},
                   {"reversals", value(v.reversals)}});
  }
};

// ---- simon.hpp ----

/// @brief Prints an algorithms::SimonResult as one line, `SimonResult(recovered_period=..., observed_y=...)`.
template <>
struct std::formatter<datamunge::algorithms::SimonResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::SimonResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SimonResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SimonResult", {
                   {"recovered_period", value(v.recovered_period)},
                   {"observed_y", value(v.observed_y)}});
  }
};

// ---- simplex.hpp ----

/// @brief Prints an algorithms::SimplexResult as one line, `SimplexResult(status=..., value=..., x=...)`.
template <>
struct std::formatter<datamunge::algorithms::SimplexResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::SimplexResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SimplexResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SimplexResult", {
                   {"status", value(v.status)},
                   {"value", value(v.value)},
                   {"x", value(v.x)}});
  }
};

// ---- sobel.hpp ----

/// @brief Prints an algorithms::SobelResult as one line, `SobelResult(gx=..., gy=..., magnitude=...)`.
template <>
struct std::formatter<datamunge::algorithms::SobelResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::SobelResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SobelResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SobelResult", {
                   {"gx", value(v.gx)},
                   {"gy", value(v.gy)},
                   {"magnitude", value(v.magnitude)}});
  }
};

// ---- sorted_search.hpp ----

/// @brief Prints an algorithms::SearchResult as one line, `SearchResult(found=..., index=..., probes=...)`.
template <>
struct std::formatter<datamunge::algorithms::SearchResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::SearchResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SearchResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SearchResult", {
                   {"found", value(v.found)},
                   {"index", value(v.index)},
                   {"probes", value(v.probes)}});
  }
};

// ---- stable_matching.hpp ----

/// @brief Prints an algorithms::StableMatchingResult as one line, `StableMatchingResult(proposer_match=..., reviewer_match=..., proposals=...)`.
template <>
struct std::formatter<datamunge::algorithms::StableMatchingResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::StableMatchingResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::StableMatchingResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("StableMatchingResult", {
                   {"proposer_match", value(v.proposer_match)},
                   {"reviewer_match", value(v.reviewer_match)},
                   {"proposals", value(v.proposals)}});
  }
};

// ---- stone_sip.hpp ----

/// @brief Prints an algorithms::StoneResult as one line, `StoneResult(x=..., iterations=..., residual=...)`.
template <>
struct std::formatter<datamunge::algorithms::StoneResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::StoneResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::StoneResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("StoneResult", {
                   {"x", value(v.x)},
                   {"iterations", value(v.iterations)},
                   {"residual", value(v.residual)}});
  }
};

// ---- structured_svm.hpp ----

/// @brief Prints an algorithms::StructuredSVMModel as one line, `StructuredSVMModel(weights=..., num_classes=..., dim=...)`.
template <>
struct std::formatter<datamunge::algorithms::StructuredSVMModel> : datamunge::format_detail::text_formatter<datamunge::algorithms::StructuredSVMModel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::StructuredSVMModel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("StructuredSVMModel", {
                   {"weights", value(v.weights)},
                   {"num_classes", value(v.num_classes)},
                   {"dim", value(v.dim)}});
  }
};

/// @brief Prints an algorithms::StructuredSVMParameters as one line, `StructuredSVMParameters(lambda=..., epochs=..., seed=...)`.
template <>
struct std::formatter<datamunge::algorithms::StructuredSVMParameters> : datamunge::format_detail::text_formatter<datamunge::algorithms::StructuredSVMParameters> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::StructuredSVMParameters& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("StructuredSVMParameters", {
                   {"lambda", value(v.lambda)},
                   {"epochs", value(v.epochs)},
                   {"seed", value(v.seed)}});
  }
};

// ---- subsequences.hpp ----

/// @brief Prints an algorithms::ScoringSegment as one line, `ScoringSegment(begin=..., end=..., score=...)`.
template <>
struct std::formatter<datamunge::algorithms::ScoringSegment> : datamunge::format_detail::text_formatter<datamunge::algorithms::ScoringSegment> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ScoringSegment& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ScoringSegment", {
                   {"begin", value(v.begin)},
                   {"end", value(v.end)},
                   {"score", value(v.score)}});
  }
};

// ---- subset_sum.hpp ----

/// @brief Prints an algorithms::SubsetSumResult as one line, `SubsetSumResult(found=..., indices=...)`.
template <>
struct std::formatter<datamunge::algorithms::SubsetSumResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::SubsetSumResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SubsetSumResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SubsetSumResult", {
                   {"found", value(v.found)},
                   {"indices", value(v.indices)}});
  }
};

// ---- substrings.hpp ----

/// @brief Prints an algorithms::MaxSubarray as one line, `MaxSubarray(sum=..., begin=..., end=...)`.
template <>
struct std::formatter<datamunge::algorithms::MaxSubarray> : datamunge::format_detail::text_formatter<datamunge::algorithms::MaxSubarray> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::MaxSubarray& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("MaxSubarray", {
                   {"sum", value(v.sum)},
                   {"begin", value(v.begin)},
                   {"end", value(v.end)}});
  }
};

/// @brief Prints an algorithms::SuffixTreeNode as one line, `SuffixTreeNode(start=..., end=..., suffix_link=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::SuffixTreeNode> : datamunge::format_detail::text_formatter<datamunge::algorithms::SuffixTreeNode> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SuffixTreeNode& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SuffixTreeNode", {
                   {"start", value(v.start)},
                   {"end", value(v.end)},
                   {"suffix_link", value(v.suffix_link)},
                   {"children", value(v.children)}});
  }
};

/// @brief Prints an algorithms::SuffixTree as one line, `SuffixTree(text=..., nodes=...)`.
template <>
struct std::formatter<datamunge::algorithms::SuffixTree> : datamunge::format_detail::text_formatter<datamunge::algorithms::SuffixTree> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::SuffixTree& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("SuffixTree", {
                   {"text", value(v.text)},
                   {"nodes", value(v.nodes)}});
  }
};

// ---- tabular_rl.hpp ----

/// @brief Prints an algorithms::StepResult as one line, `StepResult(next_state=..., reward=..., terminal=...)`.
template <>
struct std::formatter<datamunge::algorithms::StepResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::StepResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::StepResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("StepResult", {
                   {"next_state", value(v.next_state)},
                   {"reward", value(v.reward)},
                   {"terminal", value(v.terminal)}});
  }
};

/// @brief Prints an algorithms::TabularEnvironment as one line, `TabularEnvironment(num_states=..., num_actions=...)`.
template <>
struct std::formatter<datamunge::algorithms::TabularEnvironment> : datamunge::format_detail::text_formatter<datamunge::algorithms::TabularEnvironment> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::TabularEnvironment& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("TabularEnvironment", {
                   {"num_states", value(v.num_states)},
                   {"num_actions", value(v.num_actions)}});
  }
};

/// @brief Prints an algorithms::RLParameters as one line, `RLParameters(learning_rate=..., discount=..., epsilon=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::RLParameters> : datamunge::format_detail::text_formatter<datamunge::algorithms::RLParameters> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::RLParameters& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("RLParameters", {
                   {"learning_rate", value(v.learning_rate)},
                   {"discount", value(v.discount)},
                   {"epsilon", value(v.epsilon)},
                   {"episodes", value(v.episodes)},
                   {"max_steps", value(v.max_steps)},
                   {"seed", value(v.seed)}});
  }
};

/// @brief Prints an algorithms::QLearningResult as one line, `QLearningResult(q=..., greedy_policy=..., episode_returns=...)`.
template <>
struct std::formatter<datamunge::algorithms::QLearningResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::QLearningResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::QLearningResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("QLearningResult", {
                   {"q", value(v.q)},
                   {"greedy_policy", value(v.greedy_policy)},
                   {"episode_returns", value(v.episode_returns)}});
  }
};

/// @brief Prints an algorithms::TDPredictionResult as one line, `TDPredictionResult(value=..., episode_returns=...)`.
template <>
struct std::formatter<datamunge::algorithms::TDPredictionResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::TDPredictionResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::TDPredictionResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("TDPredictionResult", {
                   {"value", value(v.value)},
                   {"episode_returns", value(v.episode_returns)}});
  }
};

// ---- ternary_search.hpp ----

/// @brief Prints an algorithms::TernaryResult as one line, `TernaryResult(x=..., value=..., iterations=...)`.
template <>
struct std::formatter<datamunge::algorithms::TernaryResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::TernaryResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::TernaryResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("TernaryResult", {
                   {"x", value(v.x)},
                   {"value", value(v.value)},
                   {"iterations", value(v.iterations)}});
  }
};

// ---- topological_sort.hpp ----

/// @brief Prints an algorithms::TopologicalSortResult as one line, `TopologicalSortResult(order=..., is_dag=...)`.
template <>
struct std::formatter<datamunge::algorithms::TopologicalSortResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::TopologicalSortResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::TopologicalSortResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("TopologicalSortResult", {
                   {"order", value(v.order)},
                   {"is_dag", value(v.is_dag)}});
  }
};

// ---- tour_search.hpp ----

/// @brief Prints an algorithms::TourResult as one line, `TourResult(tour=..., cost=...)`.
template <>
struct std::formatter<datamunge::algorithms::TourResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::TourResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::TourResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("TourResult", {
                   {"tour", value(v.tour)},
                   {"cost", value(v.cost)}});
  }
};

/// @brief Prints an algorithms::VehicleRoutes as one line, `VehicleRoutes(routes=..., total_distance=...)`.
template <>
struct std::formatter<datamunge::algorithms::VehicleRoutes> : datamunge::format_detail::text_formatter<datamunge::algorithms::VehicleRoutes> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::VehicleRoutes& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("VehicleRoutes", {
                   {"routes", value(v.routes)},
                   {"total_distance", value(v.total_distance)}});
  }
};

/// @brief Prints an algorithms::InformedSearchResult as one line, `InformedSearchResult(cost=..., path=..., expanded=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::InformedSearchResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::InformedSearchResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::InformedSearchResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("InformedSearchResult", {
                   {"cost", value(v.cost)},
                   {"path", value(v.path)},
                   {"expanded", value(v.expanded)},
                   {"found", value(v.found)}});
  }
};

// ---- uniform_cost_search.hpp ----

/// @brief Prints an algorithms::UniformCostResult as one line, `UniformCostResult(cost=..., path=..., found=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::UniformCostResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::UniformCostResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::UniformCostResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("UniformCostResult", {
                   {"cost", value(v.cost)},
                   {"path", value(v.path)},
                   {"found", value(v.found)},
                   {"expanded", value(v.expanded)}});
  }
};

// ---- upgma.hpp ----

/// @brief Prints an algorithms::UPGMAMerge as one line, `UPGMAMerge(left=..., right=..., height=..., ...)`.
template <>
struct std::formatter<datamunge::algorithms::UPGMAMerge> : datamunge::format_detail::text_formatter<datamunge::algorithms::UPGMAMerge> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::UPGMAMerge& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("UPGMAMerge", {
                   {"left", value(v.left)},
                   {"right", value(v.right)},
                   {"height", value(v.height)},
                   {"id", value(v.id)}});
  }
};

/// @brief Prints an algorithms::UPGMAResult as one line, `UPGMAResult(merges=...)`.
template <>
struct std::formatter<datamunge::algorithms::UPGMAResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::UPGMAResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::UPGMAResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("UPGMAResult", {
                   {"merges", value(v.merges)}});
  }
};

// ---- vector_clock.hpp ----

/// @brief Prints an algorithms::VectorClock as one line, `VectorClock(v=...)`.
template <>
struct std::formatter<datamunge::algorithms::VectorClock> : datamunge::format_detail::text_formatter<datamunge::algorithms::VectorClock> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::VectorClock& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("VectorClock", {
                   {"v", value(v.v)}});
  }
};

// ---- vincenty.hpp ----

/// @brief Prints an algorithms::VincentyResult as one line, `VincentyResult(distance_m=..., iterations=..., converged=...)`.
template <>
struct std::formatter<datamunge::algorithms::VincentyResult> : datamunge::format_detail::text_formatter<datamunge::algorithms::VincentyResult> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::VincentyResult& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("VincentyResult", {
                   {"distance_m", value(v.distance_m)},
                   {"iterations", value(v.iterations)},
                   {"converged", value(v.converged)}});
  }
};

// ---- xiaolin_wu_line.hpp ----

/// @brief Prints an algorithms::WuPixel as one line, `WuPixel(x=..., y=..., brightness=...)`.
template <>
struct std::formatter<datamunge::algorithms::WuPixel> : datamunge::format_detail::text_formatter<datamunge::algorithms::WuPixel> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::WuPixel& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("WuPixel", {
                   {"x", value(v.x)},
                   {"y", value(v.y)},
                   {"brightness", value(v.brightness)}});
  }
};

// ---- zobrist_hashing.hpp ----

/// @brief Prints an algorithms::ZobristTable as one line, `ZobristTable(num_cells=..., num_pieces=..., keys=...)`.
template <>
struct std::formatter<datamunge::algorithms::ZobristTable> : datamunge::format_detail::text_formatter<datamunge::algorithms::ZobristTable> {
  /// @brief Renders the public fields.
  /// @param v The value to print.
  /// @return The one-line text.
  static std::string text(const datamunge::algorithms::ZobristTable& v) {
    using namespace datamunge::format_detail::algorithms_fmt;
    return record("ZobristTable", {
                   {"num_cells", value(v.num_cells)},
                   {"num_pieces", value(v.num_pieces)},
                   {"keys", value(v.keys)}});
  }
};
