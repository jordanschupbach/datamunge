#pragma once
/// @file girvan_newman.hpp
/// @brief Checked unweighted edge betweenness, batch community cuts, modularity and actual histories.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief An owned vertex count and edge-list input for native graph experiments.
///
/// This plain aggregate performs no validation. Undirected analyses canonicalize
/// directions, discard loops and deduplicate edges. Copies own their edge vectors.
///
/// @par Example
/// @code
/// const auto g=dm::algorithms::clique_bridge_graph();
/// std::cout << g.vertex_count << " " << g.edges.size(); // 6 7
/// @endcode
/// @see clique_bridge_graph(), layered_graph()
struct NetworkGraph {
    std::size_t vertex_count{0}; ///< Number of zero-based vertex ids.
    std::vector<std::pair<std::size_t,std::size_t>> edges; ///< Owned input edges; interpretation belongs to the called algorithm.
};
/// @brief Selected actual operations of a betweenness or community run.
///
/// Every call owns fresh counters. Counts exclude sorting comparisons, allocation,
/// hash internals, validation checks and arithmetic FLOPs. source_slots counts n
/// vertex slots per source, not bytes or the number of separate arrays. Component
/// passes include the terminal pass; rounds counts scored/deleted batches only.
/// These selected operations are not wall time or a complete cost model.
///
/// @par Example
/// @code
/// const dm::algorithms::GirvanNewmanWork w{};
/// std::cout << w.rounds << " " << w.bfs_pops; // 0 0
/// @endcode
/// @see GirvanNewmanRun, EdgeBetweenness
struct GirvanNewmanWork {
    std::size_t input_edges{0}; ///< Raw edge-list length before canonicalization.
    std::size_t component_passes{0}; ///< Actual connected-component computations, including terminal.
    std::size_t component_vertices{0}; ///< Vertices popped in actual component BFS passes.
    std::size_t component_neighbor_checks{0}; ///< Actual component adjacency entries examined.
    std::size_t rounds{0}; ///< Actual scored batches, each followed by deletion.
    std::size_t source_initializations{0}; ///< Actual Brandes source initializations, including isolated sources.
    std::size_t source_slots{0}; ///< Vertex slots initialized per source, summed over all sources.
    std::size_t bfs_pops{0}; ///< Vertices popped by actual Brandes BFS.
    std::size_t neighbor_checks{0}; ///< Brandes BFS adjacency entries examined.
    std::size_t discoveries{0}; ///< New non-source vertices discovered by Brandes BFS.
    std::size_t path_updates{0}; ///< Actual shortest-path count additions.
    std::size_t dependency_updates{0}; ///< Actual predecessor contributions accumulated.
    std::size_t edge_search_comparisons{0}; ///< Actual lower_bound edge comparisons during accumulation.
    std::size_t edge_scores{0}; ///< Final raw edge-score rows emitted after scored passes.
    std::size_t edge_decisions{0}; ///< Active edges tested against the batch threshold.
    std::size_t removed_edges{0}; ///< Edges actually removed over all batches.
};
/// @brief Actual raw and conventional undirected edge betweenness on canonical edges.
///
/// edges are ascending unique (u,v),u<v, matching both score vectors. raw_scores
/// retain the original ordered-double sum over all sources; scores are raw/2, the
/// conventional unordered-pair sum. No normalization by vertex-pair count is applied.
/// Disconnected pairs contribute zero. max_paths is the largest actual path count;
/// path_counts_integer_guaranteed is conservative: cleared BEFORE an integer
/// addition can exceed2^53, including2^53+1 rounded back to2^53. Larger exact powers
/// also clear it. True does not certify exact dependency fractions or rankings.
/// Nonfinite intermediates throw.
/// All vectors own their data and no caller data are mutated.
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::edge_betweenness(2,{{0,1}});
/// std::cout << r.raw_scores[0] << " " << r.scores[0]; // 2 1
/// @endcode
/// @par References
/// - M. Girvan and M. E. J. Newman (2002), Community structure in social and biological
///   networks, PNAS99:7821–7826. https://doi.org/10.1073/pnas.122653799
/// - U. Brandes (2001), A Faster Algorithm for Betweenness Centrality,
///   Journal of Mathematical Sociology25(2):163–177. https://doi.org/10.1080/0022250X.2001.9990249
/// @see edge_betweenness(), GirvanNewmanRun
struct EdgeBetweenness {
    std::vector<std::pair<std::size_t,std::size_t>> edges; ///< Ascending canonical undirected edges.
    std::vector<double> raw_scores; ///< Actual source-summed scores, twice conventional undirected values.
    std::vector<double> scores; ///< Conventional raw_scores/2, in edge order.
    GirvanNewmanWork work; ///< Selected actual operations for this call.
    double max_paths{0}; ///< Maximum actual initialized or updated shortest-path count.
    bool path_counts_integer_guaranteed{true}; ///< All path additions certified within0..2^53 before rounding; not an exact-fraction certificate.
};
/// @brief The legacy final BFS components with actual work and numeric diagnostics.
///
/// Communities retain ascending-root and sorted-adjacency BFS order, not sorted
/// member order. Stop once there are AT LEAST the requested components or no edges
/// remain. Removing every near-maximum edge in a batch can overshoot the target.
/// Only actual scored passes contribute path-count diagnostics. No modularity
/// optimization or ground-truth community certificate is implied.
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::girvan_newman_run(4,{{0,1},{1,2},{2,3},{3,0}},2);
/// std::cout << r.groups.size() << " " << r.work.removed_edges; // 4 4
/// @endcode
/// @par References
/// - M. Girvan and M. E. J. Newman (2002), Community structure in social and biological
///   networks, PNAS99:7821–7826. https://doi.org/10.1073/pnas.122653799
/// - U. Brandes (2001), A Faster Algorithm for Betweenness Centrality,
///   Journal of Mathematical Sociology25(2):163–177. https://doi.org/10.1080/0022250X.2001.9990249
/// @see girvan_newman_run(), GirvanNewmanTrace
struct GirvanNewmanRun {
    std::vector<std::vector<std::size_t>> groups; ///< Owned terminal connected components in actual BFS order.
    GirvanNewmanWork work; ///< Selected actual operations through the terminal pass.
    double max_paths{0}; ///< Largest actual shortest-path count over scored passes;0 if none.
    bool path_counts_integer_guaranteed{true}; ///< Every scored-pass path addition certified within0..2^53 before rounding; true without scoring.
};
/// @brief The operation represented by one copied primitive history record.
///
/// Canonical/round/component metadata record actual prepared inputs and returned
/// component state. BFS/path/dependency records are captured at the executing
/// operation, not replayed. No variable-length vector is stored inside a record.
///
/// @par Example
/// @code
/// const auto k=dm::algorithms::GirvanNewmanEventKind::path;
/// std::cout << (k==dm::algorithms::GirvanNewmanEventKind::path); // 1
/// @endcode
/// @see GirvanNewmanEvent, GirvanNewmanTrace
enum class GirvanNewmanEventKind {
    canonical, ///< Prepared canonical edge count.
    edge, ///< One canonical edge, before any round.
    round, ///< Start of an actual component-check round.
    components, ///< Number of actual BFS components.
    component, ///< Start of a component, with its size.
    vertex, ///< One vertex of the current component in BFS order.
    source, ///< Actual Brandes source initialization.
    pop, ///< Actual Brandes queue pop.
    scan, ///< Actual neighbour examination, with state before updates.
    discover, ///< Actual first discovery, with updated distance.
    path, ///< Actual shortest-path addition, with updated destination count.
    dependency, ///< Actual predecessor contribution, with updated dependency/edge score.
    edge_score, ///< Final raw edge score of the current scored pass.
    maximum, ///< Actual maximum raw score before deletion.
    keep, ///< Actual edge keep/remove decision.
    stop, ///< Terminal component result.
};
/// @brief One fixed-schema copied state from the actual native kernel.
///
/// round is zero based. source identifies Brandes sources, u/v the active
/// vertices, index an edge or component id and count a size; unused ids default0.
/// For pop/scan/discover, distance is the actual selected distance; scan stores
/// the neighbour state BEFORE updates. path stores paths_u and updated paths_v.
/// dependency stores both path counts, dependency_v before contribution, the actual
/// contribution, updated dependency_u and updated raw edge score. Edge scores,
/// maxima and keep decisions use score/maximum/accepted as appropriate.
/// For vertex records u is the actual BFS member, index the component id. All fields
/// are owned primitives; modifying records cannot alter the ordinary result.
///
/// @par Example
/// @code
/// const dm::algorithms::GirvanNewmanEvent e{};
/// std::cout << e.distance << " " << e.accepted; // -1 0
/// @endcode
/// @see GirvanNewmanEventKind, edge_betweenness_trace()
struct GirvanNewmanEvent {
    GirvanNewmanEventKind kind{GirvanNewmanEventKind::canonical}; ///< Operation selecting the active field meanings.
    std::size_t round{0}; ///< Zero-based actual component-check round.
    std::size_t source{0}; ///< Active Brandes source id.
    std::size_t u{0}; ///< Active predecessor, popped vertex or component member.
    std::size_t v{0}; ///< Active neighbour or successor.
    std::size_t index{0}; ///< Active canonical/current edge index or component index.
    std::size_t count{0}; ///< Canonical/active/component size or group count.
    int distance{-1}; ///< Actual selected BFS distance;minus1 before discovery.
    double paths_u{0}; ///< Actual predecessor/popped path count.
    double paths_v{0}; ///< Actual neighbour path count, updated for path events.
    double dependency_v{0}; ///< Successor dependency before the contribution.
    double contribution{0}; ///< Actual reverse predecessor contribution.
    double dependency_u{0}; ///< Predecessor dependency after accumulation.
    double score{0}; ///< Actual raw current edge score after accumulation or final row.
    double maximum{0}; ///< Actual raw maximum of the scored pass.
    bool accepted{false}; ///< For keep events, true means edge remains active.
};
/// @brief An owned ordinary result and complete online primitive event history.
///
/// A positive quota is checked before each whole-record append; default1m,0
/// disables only that optional quota. Capacity/counter/domain guards remain. No
/// partial/truncated return, numerical replay or caller mutation on rejection.
/// Input vectors, root/work/final ordinary output, working BFS/pred/queue and
/// allocator storage are excluded. Records are not numeric entries or bytes and
/// this is not a peak RAM limit. Copies own both results and event vectors.
/// Canonical/round/component metadata count too, including empty/early-stop calls.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::girvan_newman_trace(0,{},1);
/// std::cout << t.run.groups.size(); // 0
/// @endcode
/// @see girvan_newman_trace(), GirvanNewmanEvent
struct GirvanNewmanTrace {
    GirvanNewmanRun run; ///< Owned ordinary result and actual diagnostics.
    std::vector<GirvanNewmanEvent> events; ///< Complete actual primitive records, in execution order.
};
/// @brief An owned ordinary result and complete online primitive event history.
///
/// A positive quota is checked before each whole-record append; default1m,0
/// disables only that optional quota. Capacity/counter/domain guards remain. No
/// partial/truncated return, numerical replay or caller mutation on rejection.
/// Input vectors, root/work/final ordinary output, working BFS/pred/queue and
/// allocator storage are excluded. Records are not numeric entries or bytes and
/// this is not a peak RAM limit. Copies own both results and event vectors.
/// Canonical/round/component metadata count too, including empty/early-stop calls.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::edge_betweenness_trace(2,{{0,1}});
/// std::cout << t.result.scores[0]; // 1
/// @endcode
/// @see edge_betweenness_trace(), GirvanNewmanEvent
struct EdgeBetweennessTrace {
    EdgeBetweenness result; ///< Owned ordinary result and actual diagnostics.
    std::vector<GirvanNewmanEvent> events; ///< Complete actual primitive records, in execution order.
};
/// @brief Compute actual unweighted undirected edge betweenness.
///
/// All endpoints are validated before normalization and target/quota checks.
/// Directions are ignored, loops discarded and duplicate edges merged. Vertex count
/// must fit int because legacy BFS uses int distances. Input order does not affect
/// canonical adjacency order. Independent calls are thread safe unless caller data
/// are concurrently mutated. Nonfinite numeric updates throw before propagation.
/// No arbitrary positive edge-count limit is imposed, subject to capacities.
/// For n vertices,m canonical edges and r raw input edges, normalization is
/// O(r log(r+1)). One scored pass uses O(n²+n m log(m+1)) time: n-slot arrays
/// are initialized for every source and each dependency searches the edge vector.
/// Normalization can retain O(r) copied edge capacity after deduplication.
/// Scoring scratch is O(n+m); total ordinary auxiliary storage is O(r+n+m),
/// excluding caller input and final result.
/// Retains actual double source sums and returns conventional raw/2 alongside them.
/// An edgeless graph returns empty score vectors without source traversal.
///
/// @param vertex_count Number of zero-based vertices, at most INT_MAX.
/// @param undirected_edges Endpoint pairs; directions/duplicates/self-loops ignored after validation.
/// @return Owned canonical edges, actual/conventional scores, work and numeric flags.
/// @throws std::invalid_argument Endpoint/target/partition domain invalid as applicable.
/// @throws std::length_error Vertex ids, counters, containers or positive history quota cannot be represented/admitted.
/// @throws std::overflow_error A required path/dependency/score intermediate is nonfinite.
/// @par Example
/// @code
/// const auto r=dm::algorithms::edge_betweenness(3,{{0,1},{1,2}});
/// std::cout << r.scores[0] << " " << r.raw_scores[0]; // 2 4
/// @endcode
/// @par References
/// - M. Girvan and M. E. J. Newman (2002), Community structure in social and biological
///   networks, PNAS99:7821–7826. https://doi.org/10.1073/pnas.122653799
/// - U. Brandes (2001), A Faster Algorithm for Betweenness Centrality,
///   Journal of Mathematical Sociology25(2):163–177. https://doi.org/10.1080/0022250X.2001.9990249
/// @see EdgeBetweenness, girvan_newman()
EdgeBetweenness edge_betweenness(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& undirected_edges);
/// @brief Compute actual unweighted undirected edge betweenness with owned events.
///
/// All endpoints are validated before normalization and target/quota checks.
/// Directions are ignored, loops discarded and duplicate edges merged. Vertex count
/// must fit int because legacy BFS uses int distances. Input order does not affect
/// canonical adjacency order. Independent calls are thread safe unless caller data
/// are concurrently mutated. Nonfinite numeric updates throw before propagation.
/// No arbitrary positive edge-count limit is imposed, subject to capacities.
/// For n vertices,m canonical edges and r raw input edges, normalization is
/// O(r log(r+1)). One scored pass uses O(n²+n m log(m+1)) time: n-slot arrays
/// are initialized for every source and each dependency searches the edge vector.
/// Normalization can retain O(r) copied edge capacity after deduplication.
/// Scoring scratch is O(n+m); total ordinary auxiliary storage is O(r+n+m),
/// excluding caller input and final result.
/// Retains actual double source sums and returns conventional raw/2 alongside them.
/// An edgeless graph returns empty score vectors without source traversal.
/// Online complete-event admission, not a precomputed whole-work estimate; metadata
/// records count. Inputs/root/work/final ordinary output/working/allocator storage
/// excluded; not numeric entries, bytes or peak RAM. No replay/partial/truncation.
///
/// @param vertex_count Number of zero-based vertices, at most INT_MAX.
/// @param undirected_edges Endpoint pairs; directions/duplicates/self-loops ignored after validation.
/// @param max_records Positive complete-record quota, default1000000;0 disables optional quota.
/// @return Owned scores/work/numeric diagnostics and actual history.
/// @throws std::invalid_argument Endpoint/target/partition domain invalid as applicable.
/// @throws std::length_error Vertex ids, counters, containers or positive history quota cannot be represented/admitted.
/// @throws std::overflow_error A required path/dependency/score intermediate is nonfinite.
/// @par Example
/// @code
/// const auto t=dm::algorithms::edge_betweenness_trace(3,{{0,1},{1,2}});
/// std::cout << t.result.work.source_initializations; // 3
/// @endcode
/// @par References
/// - M. Girvan and M. E. J. Newman (2002), Community structure in social and biological
///   networks, PNAS99:7821–7826. https://doi.org/10.1073/pnas.122653799
/// - U. Brandes (2001), A Faster Algorithm for Betweenness Centrality,
///   Journal of Mathematical Sociology25(2):163–177. https://doi.org/10.1080/0022250X.2001.9990249
/// @see EdgeBetweenness, girvan_newman()
EdgeBetweennessTrace edge_betweenness_trace(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& undirected_edges,std::size_t max_records=1000000);
/// @brief Run the legacy near-maximum batch community cut.
///
/// All endpoints are validated before normalization and target/quota checks.
/// Directions are ignored, loops discarded and duplicate edges merged. Vertex count
/// must fit int because legacy BFS uses int distances. Input order does not affect
/// canonical adjacency order. Independent calls are thread safe unless caller data
/// are concurrently mutated. Nonfinite numeric updates throw before propagation.
/// No arbitrary positive edge-count limit is imposed, subject to capacities.
/// For n vertices,m canonical edges and r raw input edges, normalization is
/// O(r log(r+1)). One scored pass uses O(n²+n m log(m+1)) time: n-slot arrays
/// are initialized for every source and each dependency searches the edge vector.
/// Normalization can retain O(r) copied edge capacity after deduplication.
/// Scoring scratch is O(n+m); total ordinary auxiliary storage is O(r+n+m),
/// excluding caller input and final result.
/// Legacy signature/default and exact finite arithmetic/order are retained. Each
/// batch keeps only edges with raw_score<maximum−1e-12; all others are removed
/// simultaneously. Recompute after each batch, not each tied edge: this differs from
/// the original single-edge algorithm. Stop at AT LEAST target components, so ties
/// can overshoot. Initial disconnected components may already meet the target.
/// For n=0 only target1 is valid and returns empty; default2 then throws. Members
/// retain actual BFS order. At most m scored batches: total bound includes
/// O(r log(r+1)+m n²+n m² log(m+1)) and terminal component work. No modularity
/// maximization or statistical community certificate. Integer path-count guarantee
/// can be lost above2^53 even while all double updates remain finite.
///
/// @param vertex_count Number of zero-based vertices, at most INT_MAX.
/// @param undirected_edges Endpoint pairs; directions/duplicates/self-loops ignored after validation.
/// @param target_communities Desired minimum components, default2;1..n, except n0 permits only1.
/// @return Owned final connected components in actual BFS order.
/// @throws std::invalid_argument Endpoint/target/partition domain invalid as applicable.
/// @throws std::length_error Vertex ids, counters, containers or positive history quota cannot be represented/admitted.
/// @throws std::overflow_error A required path/dependency/score intermediate is nonfinite.
/// @par Example
/// @code
/// const auto g=dm::algorithms::girvan_newman(4,{{0,1},{1,2},{2,3},{3,0}},2);
/// std::cout << g.size(); // 4
/// @endcode
/// @par References
/// - M. Girvan and M. E. J. Newman (2002), Community structure in social and biological
///   networks, PNAS99:7821–7826. https://doi.org/10.1073/pnas.122653799
/// - U. Brandes (2001), A Faster Algorithm for Betweenness Centrality,
///   Journal of Mathematical Sociology25(2):163–177. https://doi.org/10.1080/0022250X.2001.9990249
/// @see GirvanNewmanRun, graph_modularity(), edge_betweenness()
std::vector<std::vector<std::size_t>> girvan_newman(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& undirected_edges,std::size_t target_communities=2);
/// @brief Run the legacy near-maximum batch community cut with actual work.
///
/// All endpoints are validated before normalization and target/quota checks.
/// Directions are ignored, loops discarded and duplicate edges merged. Vertex count
/// must fit int because legacy BFS uses int distances. Input order does not affect
/// canonical adjacency order. Independent calls are thread safe unless caller data
/// are concurrently mutated. Nonfinite numeric updates throw before propagation.
/// No arbitrary positive edge-count limit is imposed, subject to capacities.
/// For n vertices,m canonical edges and r raw input edges, normalization is
/// O(r log(r+1)). One scored pass uses O(n²+n m log(m+1)) time: n-slot arrays
/// are initialized for every source and each dependency searches the edge vector.
/// Normalization can retain O(r) copied edge capacity after deduplication.
/// Scoring scratch is O(n+m); total ordinary auxiliary storage is O(r+n+m),
/// excluding caller input and final result.
/// Legacy signature/default and exact finite arithmetic/order are retained. Each
/// batch keeps only edges with raw_score<maximum−1e-12; all others are removed
/// simultaneously. Recompute after each batch, not each tied edge: this differs from
/// the original single-edge algorithm. Stop at AT LEAST target components, so ties
/// can overshoot. Initial disconnected components may already meet the target.
/// For n=0 only target1 is valid and returns empty; default2 then throws. Members
/// retain actual BFS order. At most m scored batches: total bound includes
/// O(r log(r+1)+m n²+n m² log(m+1)) and terminal component work. No modularity
/// maximization or statistical community certificate. Integer path-count guarantee
/// can be lost above2^53 even while all double updates remain finite.
///
/// @param vertex_count Number of zero-based vertices, at most INT_MAX.
/// @param undirected_edges Endpoint pairs; directions/duplicates/self-loops ignored after validation.
/// @param target_communities Desired minimum components, default2;1..n, except n0 permits only1.
/// @return Owned final components and actual diagnostics.
/// @throws std::invalid_argument Endpoint/target/partition domain invalid as applicable.
/// @throws std::length_error Vertex ids, counters, containers or positive history quota cannot be represented/admitted.
/// @throws std::overflow_error A required path/dependency/score intermediate is nonfinite.
/// @par Example
/// @code
/// const auto r=dm::algorithms::girvan_newman_run(2,{{0,1}},2);
/// std::cout << r.groups.size() << " " << r.work.rounds; // 2 1
/// @endcode
/// @par References
/// - M. Girvan and M. E. J. Newman (2002), Community structure in social and biological
///   networks, PNAS99:7821–7826. https://doi.org/10.1073/pnas.122653799
/// - U. Brandes (2001), A Faster Algorithm for Betweenness Centrality,
///   Journal of Mathematical Sociology25(2):163–177. https://doi.org/10.1080/0022250X.2001.9990249
/// @see GirvanNewmanRun, graph_modularity(), edge_betweenness()
GirvanNewmanRun girvan_newman_run(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& undirected_edges,std::size_t target_communities=2);
/// @brief Run the legacy near-maximum batch community cut with owned actual events.
///
/// All endpoints are validated before normalization and target/quota checks.
/// Directions are ignored, loops discarded and duplicate edges merged. Vertex count
/// must fit int because legacy BFS uses int distances. Input order does not affect
/// canonical adjacency order. Independent calls are thread safe unless caller data
/// are concurrently mutated. Nonfinite numeric updates throw before propagation.
/// No arbitrary positive edge-count limit is imposed, subject to capacities.
/// For n vertices,m canonical edges and r raw input edges, normalization is
/// O(r log(r+1)). One scored pass uses O(n²+n m log(m+1)) time: n-slot arrays
/// are initialized for every source and each dependency searches the edge vector.
/// Normalization can retain O(r) copied edge capacity after deduplication.
/// Scoring scratch is O(n+m); total ordinary auxiliary storage is O(r+n+m),
/// excluding caller input and final result.
/// Legacy signature/default and exact finite arithmetic/order are retained. Each
/// batch keeps only edges with raw_score<maximum−1e-12; all others are removed
/// simultaneously. Recompute after each batch, not each tied edge: this differs from
/// the original single-edge algorithm. Stop at AT LEAST target components, so ties
/// can overshoot. Initial disconnected components may already meet the target.
/// For n=0 only target1 is valid and returns empty; default2 then throws. Members
/// retain actual BFS order. At most m scored batches: total bound includes
/// O(r log(r+1)+m n²+n m² log(m+1)) and terminal component work. No modularity
/// maximization or statistical community certificate. Integer path-count guarantee
/// can be lost above2^53 even while all double updates remain finite.
/// Online whole-record admission, default1m/0 optional-off. All canonical, round,
/// component and actual numeric events count. No synthetic uncounted snapshots,
/// replay, partial/truncated return or caller mutation. Inputs/root/work/final
/// ordinary result/working/allocator storage excluded; not bytes or peak RAM.
///
/// @param vertex_count Number of zero-based vertices, at most INT_MAX.
/// @param undirected_edges Endpoint pairs; directions/duplicates/self-loops ignored after validation.
/// @param target_communities Desired minimum components, default2;1..n, except n0 permits only1.
/// @param max_records Positive complete-record quota, default1000000;0 disables optional quota.
/// @return Owned final components/work/numeric diagnostics and complete history.
/// @throws std::invalid_argument Endpoint/target/partition domain invalid as applicable.
/// @throws std::length_error Vertex ids, counters, containers or positive history quota cannot be represented/admitted.
/// @throws std::overflow_error A required path/dependency/score intermediate is nonfinite.
/// @par Example
/// @code
/// const auto t=dm::algorithms::girvan_newman_trace(2,{{0,1}},2);
/// std::cout << t.run.work.removed_edges; // 1
/// @endcode
/// @par References
/// - M. Girvan and M. E. J. Newman (2002), Community structure in social and biological
///   networks, PNAS99:7821–7826. https://doi.org/10.1073/pnas.122653799
/// - U. Brandes (2001), A Faster Algorithm for Betweenness Centrality,
///   Journal of Mathematical Sociology25(2):163–177. https://doi.org/10.1080/0022250X.2001.9990249
/// @see GirvanNewmanRun, graph_modularity(), edge_betweenness()
GirvanNewmanTrace girvan_newman_trace(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& undirected_edges,std::size_t target_communities=2,std::size_t max_records=1000000);
/// @brief Evaluate unweighted original-graph modularity for a complete partition.
///
/// After canonicalizing the ORIGINAL edges, compute sum_c[L_c/m−(K_c/(2m))²],
/// where L_c is internal edges and K_c is original degree sum. Deleted GN edges must
/// still be supplied. Partition must contain every vertex exactly once and no empty
/// groups; n0 requires an empty partition. Edgeless graphs return0 by explicit
/// convention after validation. No mutation; O(r log(r+1)+n+m) work and
/// O(r+n+m) owned working storage, including raw-record normalization capacity.
/// A high value is neither global optimality nor ground-truth community evidence.
///
/// @param vertex_count Number of zero-based vertices, at most INT_MAX.
/// @param undirected_edges Endpoint pairs; directions/duplicates/self-loops ignored after validation.
/// @param groups Complete disjoint nonempty groups in arbitrary order.
/// @return Finite modularity, or0 for an edgeless valid graph.
/// @throws std::invalid_argument Invalid endpoint or incomplete/overlapping/empty/invalid-id partition.
/// @throws std::length_error Vertex ids, degree counters or container capacities cannot fit.
/// @par Example
/// @code
/// std::cout << dm::algorithms::graph_modularity(2,{{0,1}},{{0},{1}}); // -0.5
/// @endcode
/// @par References
/// - M. E. J. Newman and M. Girvan (2004), Finding and evaluating community
///   structure in networks, Physical Review E69:026113, Eq5.
///   https://doi.org/10.1103/PhysRevE.69.026113
/// @see girvan_newman_trace()
double graph_modularity(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& undirected_edges,const std::vector<std::vector<std::size_t>>& groups);
/// @brief Build two complete subgraphs with distinct cross bridges.
///
/// Left ids precede right ids. Bridge k joins left_count−1−k to left_count+k.
/// All edges are unique, ascending endpoints; output order is deterministic.
/// Counts/capacities and positive whole-vertex quota are checked before allocation.
/// 0 quota disables only that optional limit; int-id capacity remains. O(n²) worst
/// work/output. No caller mutation or external data.
///
/// @param left_count Positive left clique size, default3.
/// @param right_count Positive right clique size, default3.
/// @param bridges Number of distinct cross bridges, default1; at most min(left,right).
/// @param max_vertices Positive whole-vertex quota, default1000000;0 disables optional quota.
/// @return Owned graph with checked zero-based vertex ids.
/// @throws std::invalid_argument Zero dimensions or excessive bridge count.
/// @throws std::length_error Whole vertex quota, representable counts or capacities exceeded.
/// @par Example
/// @code
/// const auto g=dm::algorithms::clique_bridge_graph();
/// std::cout << g.vertex_count << " " << g.edges.size(); // 6 7
/// @endcode
/// @see NetworkGraph, edge_betweenness()
NetworkGraph clique_bridge_graph(std::size_t left_count=3,std::size_t right_count=3,std::size_t bridges=1,std::size_t max_vertices=1000000);
/// @brief Build consecutive complete-bipartite layers for path-count experiments.
///
/// Vertices are consecutive within layers. Every vertex of one layer connects to
/// every vertex of the next, with no other edges. Counts/capacities and positive
/// whole-vertex quota are checked before allocation;0 quota optional-off, int ids
/// still bounded. Storage/work O(layers*width²); duplicate or loop edges absent.
///
/// @param layers Positive number of layers.
/// @param width Positive vertices per layer, default2.
/// @param max_vertices Positive whole-vertex quota, default1000000;0 disables optional quota.
/// @return Owned graph with checked zero-based vertex ids.
/// @throws std::invalid_argument Zero dimensions or excessive bridge count.
/// @throws std::length_error Whole vertex quota, representable counts or capacities exceeded.
/// @par Example
/// @code
/// const auto g=dm::algorithms::layered_graph(3,2);
/// std::cout << g.vertex_count << " " << g.edges.size(); // 6 8
/// @endcode
/// @see NetworkGraph, edge_betweenness()
NetworkGraph layered_graph(std::size_t layers,std::size_t width=2,std::size_t max_vertices=1000000);
} // namespace datamunge::algorithms
