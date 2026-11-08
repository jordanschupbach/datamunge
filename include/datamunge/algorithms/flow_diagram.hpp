#pragma once
/// @file flow_diagram.hpp
/// @brief Native directed flow/capacity diagrams with separate lanes for input edge identities.
#include <datamunge/algorithms/dinic.hpp>
#include <datamunge/plot/diagram.hpp>
namespace datamunge::algorithms {
/// @brief Draw flow/capacity labels and a supplied cut using a native editable diagram.
///
/// Positions are caller-chosen equal-scale diagram coordinates, one per vertex.
/// Separate polyline lanes preserve parallel/antiparallel edge identities; self
/// edges use loops. Labels show input index and four-significant-digit flow/capacity,
/// so displayed numbers may be rounded. White-backed labels remain legible over
/// their edge strokes. Source-side nodes are green, complement
/// nodes orange, and edges crossing the supplied cut are red. Nodes name source,
/// sink and vertex ids. Layout aesthetics and avoidance of unrelated crossings
/// remain the caller's responsibility. Work/storage O(V+E log(E+1)+text).
/// Validates complete finite graph/result shape, cut ids and finite coordinates;
/// adjacent nodes need room for their circles. It does not certify feasibility or
/// optimality of caller-edited flows/cuts. Inputs are unchanged; returned primitives
/// own all their data and independent calls share no mutable state.
/// @param network Original graph with distinct valid terminals and finite nonnegative capacities.
/// @param result One finite flow per input edge, finite total and unique in-range cut ids.
/// @param positions Exactly vertex_count finite native diagram positions.
/// @return Owned native diagram, including every input edge even if its flow is zero.
/// @throws std::invalid_argument On invalid graph/result/coordinates or insufficient endpoint spacing.
/// @throws std::length_error On container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const dm::algorithms::FlowNetwork g{2,{{0,1,3}},0,1};
/// const auto r=dm::algorithms::edmonds_karp(2,g.edges,0,1);
/// const auto d=dm::algorithms::plot_max_flow(g,r,{{0,0},{6,0}});
/// std::print("{}\n",d.to_svg().find("<svg")!=std::string::npos); // true
/// @endcode
/// @see max_flow_certificate(), edmonds_karp_trace(), graphics::Diagram
[[nodiscard]] graphics::Diagram plot_max_flow(const FlowNetwork& network,const MaxFlowResult& result,
                                             const std::vector<graphics::DiagramPoint>& positions);
} // namespace datamunge::algorithms
