#pragma once
/// @file bron_kerbosch.hpp
/// @brief Deterministic maximal-clique enumeration, actual pivot frames and multipartite fixtures.
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
/// @brief One owned actual recursion-entry frame of pivoted Bron-Kerbosch.
/// r is the current clique in search insertion order. Disjoint p/x contain
/// vertices adjacent to every member of r: p may still branch, x was processed
/// earlier at this frame's ancestors/siblings. They are not lists of already
/// reported vertices. Each frame is captured before its own branch loop mutates
/// p/x. Returned clique members are sorted separately, without changing records.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bron_kerbosch_trace(3,{{1,2},{0,2},{0,1}});
/// std::print("{} {}\n",trace.nodes[0].pivot,trace.nodes[0].branch_vertices); // 0 [0]
/// @endcode
/// @see BronKerboschTrace, bron_kerbosch_trace()
struct BronKerboschNode {
    std::size_t id{0}; ///< Preorder frame index, matching its index in the owning trace.
    std::size_t parent{0}; ///< Parent frame index; the root (id zero) uses parent zero.
    int via_vertex{-1}; ///< Candidate selected by the parent, or -1 for the root.
    std::vector<int> r; ///< Current clique in actual insertion order.
    std::vector<int> p; ///< Candidate set at frame entry, in deterministic inherited order.
    std::vector<int> x; ///< Excluded set at frame entry, in actual inherited/appended order.
    int pivot{-1}; ///< Actual pivot from p then x, or -1 when both sets are empty.
    int pivot_degree{0}; ///< Number of the pivot's neighbours in this frame's p.
    std::vector<int> branch_vertices; ///< Actual snapshot p minus N(pivot), in p order.
    bool reported{false}; ///< Both p and x are empty; this frame emits a maximal clique.
};

/// @brief Owned maximal cliques, actual frames, histogram and logical search counters.
/// Clique members are sorted; outer order is deterministic depth-first discovery
/// order, preserving the historical implementation. The empty zero-vertex graph
/// emits one empty clique. Counts describe this library's vector/matrix search,
/// not wall time or a general optimal-time guarantee. Nodes add O((n+1)*C) retained
/// vertex storage for C calls; ordinary enumeration allocates no diagnostic records.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bron_kerbosch_trace(3,{{1},{0,2},{1}});
/// std::print("{} {}\n",trace.cliques.size(),trace.clique_size_counts[2]); // 2 2
/// @endcode
/// @see bron_kerbosch_trace(), BronKerboschNode
struct BronKerboschTrace {
    std::vector<std::vector<int>> cliques; ///< Owned maximal cliques with sorted members and discovery outer order.
    std::vector<BronKerboschNode> nodes; ///< Actual preorder frames, copied before their branch loops.
    std::vector<std::size_t> clique_size_counts; ///< Counts indexed by size 0..n, including the empty-graph size-zero clique.
    int vertex_count{0}; ///< Validated n; only the first n adjacency rows are used.
    std::size_t edge_count{0}; ///< Unique undirected non-loop edges after union symmetrisation.
    std::size_t input_adjacency_entries{0}; ///< Original entries in the first n rows, including duplicates and loops.
    std::size_t recursive_calls{0}; ///< Actual frame entries, including reported and non-reporting leaves.
    std::size_t branches{0}; ///< Actual parent-to-child calls, equal to recursive_calls-1 on success.
    std::size_t pivot_candidates{0}; ///< Vertices examined in all p-union-x pivot selections.
    std::size_t pivot_tests{0}; ///< Matrix adjacency probes during pivot scoring.
    std::size_t branch_tests{0}; ///< Matrix probes forming p minus N(pivot) branch snapshots.
    std::size_t intersection_tests{0}; ///< Matrix probes while forming child p/x intersections.
    std::size_t max_depth{0}; ///< Greatest actual r size, with root depth zero.
};

/// @brief Private validated graph normalisation and one shared deterministic search.
namespace bron_kerbosch_detail {
/// @brief Increment an actual logical counter without wrapping size_t.
/// @param value Counter to update.
/// @throws std::overflow_error If another increment is unrepresentable.
inline void increment(std::size_t& value) {
    if (value==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("Bron-Kerbosch counter exceeds size_t");
    ++value;
}

/// @brief Execute one actual pivoted frame, preserving original scan and branch order.
/// @tparam Record Whether to copy diagnostic frames and update the size histogram.
/// @param r Owned current clique in search order.
/// @param p Owned current candidate set.
/// @param x Owned current excluded set.
/// @param matrix Borrowed normalised symmetric non-loop adjacency matrix.
/// @param trace Owned result and logical counters being populated.
/// @param parent Parent id, zero for the root.
/// @param via_vertex Parent's selected vertex, -1 for the root.
/// @param max_calls Optional maximum complete search calls, zero meaning unlimited.
/// @throws std::runtime_error Before exceeding a positive call budget.
/// @throws std::overflow_error Before counter wrap; allocation failures propagate.
template<bool Record>
inline void visit(std::vector<int> r,std::vector<int> p,std::vector<int> x,
                  const std::vector<std::vector<unsigned char>>& matrix,
                  BronKerboschTrace& trace,std::size_t parent,int via_vertex,std::size_t max_calls) {
    if (max_calls && trace.recursive_calls>=max_calls) throw std::runtime_error("Bron-Kerbosch call budget exhausted");
    const auto id=trace.recursive_calls;
    increment(trace.recursive_calls);
    trace.max_depth=std::max(trace.max_depth,r.size());
    if (p.empty() && x.empty()) {
        if constexpr (Record) {
            trace.nodes.push_back({id,parent,via_vertex,r,p,x,-1,0,{},true});
            increment(trace.clique_size_counts[r.size()]);
        }
        trace.cliques.push_back(std::move(r));
        return;
    }
    int pivot=-1,best=-1;
    for (int u:p) {
        increment(trace.pivot_candidates);
        int degree=0;
        for (int v:p) {increment(trace.pivot_tests);if (matrix[u][v]) ++degree;}
        if (degree>best) {best=degree;pivot=u;}
    }
    for (int u:x) {
        increment(trace.pivot_candidates);
        int degree=0;
        for (int v:p) {increment(trace.pivot_tests);if (matrix[u][v]) ++degree;}
        if (degree>best) {best=degree;pivot=u;}
    }
    std::vector<int> candidates;
    for (int v:p) {increment(trace.branch_tests);if (!matrix[pivot][v]) candidates.push_back(v);}
    if constexpr (Record) trace.nodes.push_back({id,parent,via_vertex,r,p,x,pivot,best,candidates,false});
    for (int v:candidates) {
        auto next_r=r;next_r.push_back(v);
        std::vector<int> next_p,next_x;
        for (int w:p) {increment(trace.intersection_tests);if (matrix[v][w]) next_p.push_back(w);}
        for (int w:x) {increment(trace.intersection_tests);if (matrix[v][w]) next_x.push_back(w);}
        increment(trace.branches);
        visit<Record>(std::move(next_r),std::move(next_p),std::move(next_x),matrix,trace,id,v,max_calls);
        p.erase(std::remove(p.begin(),p.end(),v),p.end());
        x.push_back(v);
    }
}

/// @brief Validate all used endpoints before allocation, symmetrise by union and enumerate.
/// @tparam Record Whether to retain actual frames and size counts.
/// @param n Nonnegative vertex count.
/// @param adjacency At least n rows; extra rows ignored as historically.
/// @param max_calls Optional complete-call budget, zero meaning unlimited.
/// @return Owned output and actual counters, with records only when requested.
/// @throws std::invalid_argument If n is negative or fewer than n rows are supplied.
/// @throws std::out_of_range If a used row contains an endpoint outside [0,n).
/// @throws std::runtime_error If the positive call budget is exhausted.
/// @throws std::overflow_error Before counter wrap; allocation failures propagate.
template<bool Record>
inline BronKerboschTrace enumerate(int n,const std::vector<std::vector<int>>& adjacency,std::size_t max_calls) {
    if (n<0 || adjacency.size()<static_cast<std::size_t>(n))
        throw std::invalid_argument("Bron-Kerbosch requires nonnegative n and at least n adjacency rows");
    for (int u=0;u<n;++u) for (int v:adjacency[u])
        if (v<0 || v>=n) throw std::out_of_range("Bron-Kerbosch endpoint outside [0,n)");
    BronKerboschTrace trace;trace.vertex_count=n;
    std::vector<std::vector<unsigned char>> matrix(n,std::vector<unsigned char>(n,0));
    for (int u=0;u<n;++u) for (int v:adjacency[u]) {
        increment(trace.input_adjacency_entries);
        if (u!=v && !matrix[u][v]) {matrix[u][v]=matrix[v][u]=1;increment(trace.edge_count);}
    }
    if constexpr (Record) trace.clique_size_counts.assign(static_cast<std::size_t>(n)+1,0);
    std::vector<int> p(n);
    for (int v=0;v<n;++v) p[v]=v;
    visit<Record>({},std::move(p),{},matrix,trace,0,-1,max_calls);
    for (auto& clique:trace.cliques) std::sort(clique.begin(),clique.end());
    return trace;
}
} // namespace bron_kerbosch_detail

/// @brief Enumerate every inclusion-maximal clique of a union-symmetrised integer graph.
/// Preserves the original inline signature, sorted clique members and discovery
/// outer order. This is not just one largest clique. Uses vertices 0..n-1;
/// each used adjacency entry adds an undirected edge, duplicates collapse and
/// valid self-loops are ignored. Extra rows are ignored, preserving legacy input
/// semantics. The empty graph returns one empty clique. Inputs are borrowed only
/// during the call and never changed; outputs own their storage. Pivot scoring
/// scans p then x and keeps the first maximum; branches follow p order.
/// With A used input entries, C actual calls and S returned vertex ids plus
/// clique headers, conservative work is O(A+(n+1)^2*(C+1)+S*log(n+2)); storage is
/// O(n^2+S), including the matrix and value-copied recursion sets. Ordinary calls
/// retain no diagnostic records. Output and search can grow exponentially.
/// @param n Nonnegative number of vertices.
/// @param adjacency At least n rows with endpoints in [0,n); later rows are ignored.
/// @return Owned maximal cliques, including singleton isolates and the n=0 empty clique.
/// @throws std::invalid_argument If n is negative or fewer than n rows are supplied.
/// @throws std::out_of_range If an endpoint in a used row is invalid.
/// @throws std::overflow_error Before logical counter wrap; allocation failures propagate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bron_kerbosch(3,{{1},{0,2},{1}})); // [[0, 1], [1, 2]]
/// @endcode
/// @par References
/// - C. Bron and J. Kerbosch (1973), Algorithm 457: finding all cliques of an
///   undirected graph, CACM 16(9), 575-577, doi:10.1145/362342.362367.
///   This vector/matrix implementation uses maximum-neighbours-in-p pivoting.
/// @see bron_kerbosch_trace(), complete_multipartite_graph()
inline std::vector<std::vector<int>> bron_kerbosch(int n,const std::vector<std::vector<int>>& adjacency) {
    return bron_kerbosch_detail::enumerate<false>(n,adjacency,0).cliques;
}

/// @brief Record actual recursion-entry sets, pivot branches and maximal-clique counts.
/// Uses exactly the same kernel and output ordering as bron_kerbosch(). Snapshots
/// precede each frame's branch loop; parent p/x must be updated by preceding sibling
/// choices before deriving a later child. Records keep r in search insertion order.
/// Adds O((n+1)*C) retained set/branch storage and O(n) histogram storage. A positive
/// max_calls throws before entering an excess frame, returning no partial result;
/// zero (default) is unlimited. This limit does not avoid adjacency-matrix allocation.
/// Input validation, symmetrisation and ownership rules match ordinary enumeration.
/// @param n Nonnegative vertex count.
/// @param adjacency At least n rows with valid used endpoints; extra rows ignored.
/// @param max_calls Maximum complete-search frame entries, or zero for unlimited.
/// @return Owned cliques, actual preorder frames, histogram and logical counts.
/// @throws std::invalid_argument If n or the row count is invalid.
/// @throws std::out_of_range If a used endpoint is invalid.
/// @throws std::runtime_error If a positive call limit is exhausted; no partial result returned.
/// @throws std::overflow_error Before counter wrap; allocation failures propagate.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bron_kerbosch_trace(3,{{1,2},{0,2},{0,1}});
/// std::print("{} {}\n",trace.recursive_calls,trace.max_depth); // 4 3
/// @endcode
/// @par References
/// - C. Bron and J. Kerbosch (1973), doi:10.1145/362342.362367; actual shared
///   pivoted search records, not a separately reconstructed recursion tree.
/// @see bron_kerbosch(), BronKerboschNode, BronKerboschTrace
inline BronKerboschTrace bron_kerbosch_trace(int n,const std::vector<std::vector<int>>& adjacency,std::size_t max_calls=0) {
    return bron_kerbosch_detail::enumerate<true>(n,adjacency,max_calls);
}

/// @brief Build a complete multipartite undirected graph with consecutive vertex groups.
/// Every pair in different parts is adjacent; vertices in one part are not.
/// Positive part sizes produce symmetric, sorted, duplicate-free rows. An empty
/// list returns zero rows. For n total vertices, O(n^2) work and O(n+E) owned
/// output/temporary storage, without randomness, borrowed results or shared mutation.
/// @param part_sizes Positive sizes of consecutive parts; an empty list is allowed.
/// @return Owned adjacency lists, with vertices labelled consecutively from zero.
/// @throws std::invalid_argument If any part is nonpositive.
/// @throws std::overflow_error If the total vertex count exceeds int, before allocation.
/// Allocation/capacity failures propagate for large valid output.
/// @par Example
/// @code
/// const auto adjacency=dm::algorithms::complete_multipartite_graph({2,2});
/// std::print("{}\n",adjacency[0]); // [2, 3]
/// @endcode
/// @see bron_kerbosch_trace()
inline std::vector<std::vector<int>> complete_multipartite_graph(const std::vector<int>& part_sizes) {
    int n=0;
    for (int size:part_sizes) {
        if (size<=0) throw std::invalid_argument("multipartite part sizes must be positive");
        if (size>std::numeric_limits<int>::max()-n) throw std::overflow_error("multipartite vertex count exceeds int");
        n+=size;
    }
    std::vector<int> group(n);
    int vertex=0,id=0;
    for (int size:part_sizes) {for (int k=0;k<size;++k) group[vertex++]=id;++id;}
    std::vector<std::vector<int>> adjacency(n);
    for (int u=0;u<n;++u) for (int v=u+1;v<n;++v) if (group[u]!=group[v]) {
        adjacency[u].push_back(v);adjacency[v].push_back(u);
    }
    return adjacency;
}
} // namespace datamunge::algorithms
