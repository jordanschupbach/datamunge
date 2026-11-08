/// @file brute_force_search.cpp
/// @brief Shared explicit-stack simple-prefix search preserving historical incumbent-bound order.
#include <datamunge/algorithms/brute_force_search.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private graph validation and actual explicit DFS state transitions.
namespace {
/// @brief Owned per-source directed adjacency in input order.
using Adjacency=std::vector<std::vector<std::size_t>>;
/// @brief One active explicit DFS frame, with its next adjacency entry.
struct PathFrame {
    std::size_t vertex{0}; ///< Last live-path vertex represented by this frame.
    std::size_t next_successor{0}; ///< Next adjacency entry to inspect.
    std::size_t id{0}; ///< Actual preorder entry id, whether recorded or not.
    bool terminal{false}; ///< Goal or bound-pruned frame; pop without scanning edges.
};
/// @brief Increment a logical search counter without wrap.
/// @param value Mutable size counter.
/// @throws std::overflow_error Before an unrepresentable increment.
void increment(std::size_t& value) {
    if (value==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("brute-force search counter exceeds size_t");
    ++value;
}
/// @brief Validate all indices before allocating directed adjacency.
/// @param n Number of vertices.
/// @param edges Directed arcs in input order.
/// @param start Initial terminal.
/// @param goal Final terminal.
/// @return Owned per-source adjacency preserving duplicates and loops.
/// @throws std::invalid_argument If any endpoint or terminal is outside [0,n).
Adjacency adjacency(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                    std::size_t start,std::size_t goal) {
    if (start>=n || goal>=n) throw std::invalid_argument("search: terminal");
    for (const auto& [u,v]:edges) if (u>=n || v>=n) throw std::invalid_argument("search: endpoint");
    Adjacency result(n);
    for (const auto& [u,v]:edges) result[u].push_back(v);
    return result;
}
/// @brief Enter one actual simple prefix and apply its goal/incumbent decision.
/// @tparam Record Whether to retain the entered-prefix record.
/// @param vertex Vertex being appended to the live path.
/// @param parent Actual parent visit id; zero for the root.
/// @param goal Requested goal.
/// @param path Mutable live simple path.
/// @param on_path Mutable membership flags for that path only.
/// @param trace Mutable owned result and actual counters.
/// @param options Complete-search budget and retained-prefix choice.
/// @return Active frame, already classified as goal, pruned or expandable.
/// @throws std::runtime_error Before exceeding a positive visit budget.
/// @throws std::overflow_error Before counter wrap; allocation errors propagate.
template<bool Record>
PathFrame enter(std::size_t vertex,std::size_t parent,std::size_t goal,std::vector<std::size_t>& path,
                std::vector<unsigned char>& on_path,BruteForceTrace& trace,BruteForceTraceOptions options) {
    if (options.max_visits && trace.result.expanded>=options.max_visits)
        throw std::runtime_error("brute-force search visit budget exhausted");
    const auto id=trace.result.expanded;
    increment(trace.result.expanded);
    path.push_back(vertex);on_path[vertex]=1;
    trace.max_depth=std::max(trace.max_depth,path.size()-1);
    const auto before=trace.result.found?trace.result.path.size():0;
    const bool is_goal=vertex==goal;
    bool improved=false;
    if (is_goal) {
        increment(trace.goal_visits);
        if (!trace.result.found || path.size()<trace.result.path.size()) {
            trace.result.path=path;trace.result.found=true;improved=true;
            increment(trace.improvements);
        }
    }
    const bool pruned=!is_goal && trace.result.found && path.size()>=trace.result.path.size();
    if (pruned) increment(trace.bound_prunes);
    const auto after=trace.result.found?trace.result.path.size():0;
    if constexpr (Record)
        trace.steps.push_back({id,parent,vertex,path.size()-1,options.record_paths?path:std::vector<std::size_t>{},
                               before,after,0,0,is_goal,improved,pruned});
    return {vertex,0,id,is_goal || pruned};
}
/// @brief Canonical explicit-stack DFS; ordinary calls compile out record allocation.
/// @tparam Record Whether to retain actual visit records.
/// @param n Vertex count.
/// @param edges Directed input arcs.
/// @param start Start vertex.
/// @param goal Goal vertex.
/// @param options Optional complete-search limit and path-copy choice.
/// @return Owned result and actual counters, with records only when requested.
/// @throws std::invalid_argument For invalid terminals/endpoints.
/// @throws std::runtime_error If a positive visit budget is exhausted.
/// @throws std::overflow_error Before counter wrap; allocation errors propagate.
template<bool Record>
BruteForceTrace search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                        std::size_t start,std::size_t goal,BruteForceTraceOptions options) {
    const auto next=adjacency(n,edges,start,goal);
    BruteForceTrace trace;trace.paths_recorded=options.record_paths;
    std::vector<unsigned char> on_path(n,0);
    std::vector<std::size_t> path;
    std::vector<PathFrame> active;
    active.push_back(enter<Record>(start,0,goal,path,on_path,trace,options));
    while (!active.empty()) {
        auto& frame=active.back();
        if (frame.terminal || frame.next_successor==next[frame.vertex].size()) {
            on_path[frame.vertex]=0;path.pop_back();active.pop_back();continue;
        }
        const auto vertex=next[frame.vertex][frame.next_successor++];
        increment(trace.edges_examined);
        if constexpr (Record) increment(trace.steps[frame.id].edges_examined);
        if (on_path[vertex]) {
            increment(trace.cycle_rejections);
            if constexpr (Record) increment(trace.steps[frame.id].cycle_rejections);
            continue;
        }
        const auto parent=frame.id;
        active.push_back(enter<Record>(vertex,parent,goal,path,on_path,trace,options));
    }
    return trace;
}
} // namespace
BasicPathResult brute_force_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                                   std::size_t start,std::size_t goal) {
    return search<false>(n,edges,start,goal,{0,false}).result;
}
BruteForceTrace brute_force_search_trace(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                                         std::size_t start,std::size_t goal,BruteForceTraceOptions options) {
    return search<true>(n,edges,start,goal,options);
}
std::vector<std::pair<std::size_t,std::size_t>> complete_directed_graph_edges(std::size_t n) {
    std::vector<std::pair<std::size_t,std::size_t>> edges;
    if (n<2) return edges;
    if (n>std::numeric_limits<std::size_t>::max()/(n-1))
        throw std::overflow_error("complete directed graph arc count exceeds size_t");
    const auto count=n*(n-1);
    if (count>edges.max_size()) throw std::length_error("complete directed graph exceeds vector capacity");
    edges.reserve(count);
    for (std::size_t u=0;u<n;++u) for (std::size_t v=0;v<n;++v) if (u!=v) edges.emplace_back(u,v);
    return edges;
}
} // namespace datamunge::algorithms
