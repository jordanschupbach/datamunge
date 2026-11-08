/// @file breadth_first_search.cpp
/// @brief Shared FIFO shortest-hop traversal with optional actual queue/discovery recording.
#include <datamunge/algorithms/breadth_first_search.hpp>
#include <algorithm>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private validated adjacency and common traced/untraced traversal.
namespace {
/// @brief Owned adjacency lists in supplied edge order.
using Adjacency=std::vector<std::vector<std::size_t>>;
/// @brief Validate every endpoint/terminal before constructing ordered adjacency lists.
/// @param n Vertex count.
/// @param edges Borrowed ordered endpoint pairs.
/// @param start Required valid start.
/// @param goal Required valid goal when complete=false; ignored otherwise.
/// @param directed Whether each input edge is inserted once rather than both ways.
/// @param complete Whether there is no goal to validate or stop at.
/// @return Owned validated adjacency.
/// @throws std::invalid_argument For any out-of-range terminal or endpoint.
Adjacency adjacency(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
    std::size_t start,std::size_t goal,bool directed,bool complete) {
    if (start>=n || (!complete && goal>=n)) throw std::invalid_argument("breadth_first_search: terminal out of range");
    for (const auto& [u,v]:edges) if (u>=n || v>=n)
        throw std::invalid_argument("breadth_first_search: endpoint out of range");
    Adjacency out(n);
    for (const auto& [u,v]:edges) {out[u].push_back(v);if (!directed) out[v].push_back(u);}
    return out;
}
/// @brief Execute the canonical queue loop, with allocation of event records compiled out for ordinary BFS.
/// @tparam Record Whether dequeue/discovery records and layers are retained.
/// @tparam Complete Whether to exhaust the component without a target stop.
/// @param n Vertex count, also the undiscovered/predecessor sentinel.
/// @param edges Ordered endpoint pairs.
/// @param start Starting vertex.
/// @param goal Target vertex for single-goal mode; ignored in complete traversal.
/// @param directed Whether input pairs are arcs.
/// @param capture_queue Whether recorded steps also copy the pre-removal FIFO.
/// @return Owned BFS state and optionally recorded events.
/// @throws std::invalid_argument If terminals or endpoints are invalid.
template<bool Record,bool Complete=false>
BreadthFirstTrace run(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
    std::size_t start,std::size_t goal,bool directed,bool capture_queue) {
    const auto graph=adjacency(n,edges,start,goal,directed,Complete);
    BreadthFirstTrace out;out.directed=directed;out.queues_recorded=Record&&capture_queue;
    out.goal_requested=!Complete;
    out.adjacency_entries=edges.size()*(directed?1:2);out.distance.assign(n,n);out.predecessor.assign(n,n);
    std::vector<std::size_t> queue;queue.reserve(n);queue.push_back(start);std::size_t head=0;
    out.distance[start]=0;out.vertices_enqueued=1;out.max_queue_size=1;
    if constexpr (Record) out.layer_counts.push_back(1);
    while (head<queue.size()) {
        const auto vertex=queue[head];BreadthFirstStep step;
        if constexpr (Record) {
            step.vertex=vertex;step.depth=out.distance[vertex];step.queue_size_before=queue.size()-head;
            if (capture_queue) step.queue_before.assign(queue.begin()+static_cast<std::ptrdiff_t>(head),queue.end());
        }
        ++head;out.result.order.push_back(vertex);
        if (!Complete && vertex==goal) {
            out.result.found=true;
            if constexpr (Record) {step.reached_goal=true;step.queue_size_after=queue.size()-head;out.steps.push_back(std::move(step));}
            break;
        }
        for (auto next:graph[vertex]) {
            ++out.edges_examined;
            if constexpr (Record) ++step.edges_examined;
            if (out.distance[next]!=n) continue;
            out.distance[next]=out.distance[vertex]+1;out.predecessor[next]=vertex;queue.push_back(next);
            ++out.vertices_enqueued;out.max_queue_size=std::max(out.max_queue_size,queue.size()-head);
            if constexpr (Record) {
                step.discovered.push_back(next);
                const auto depth=out.distance[next];if (out.layer_counts.size()<=depth) out.layer_counts.resize(depth+1);
                ++out.layer_counts[depth];
            }
        }
        if constexpr (Record) {step.queue_size_after=queue.size()-head;out.steps.push_back(std::move(step));}
    }
    if (out.result.found) {
        for (auto vertex=goal;;vertex=out.predecessor[vertex]) {
            out.result.path.push_back(vertex);if (vertex==start) break;
        }
        std::reverse(out.result.path.begin(),out.result.path.end());
    }
    return out;
}
} // namespace
UnweightedSearchResult breadth_first_search(std::size_t n,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,bool directed) {
    return run<false>(n,edges,start,goal,directed,false).result;
}
BreadthFirstTrace breadth_first_search_trace(std::size_t n,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,bool directed,bool capture_queue) {
    return run<true>(n,edges,start,goal,directed,capture_queue);
}
BreadthFirstTrace breadth_first_traversal(std::size_t n,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,bool directed,bool capture_queue) {
    return run<true,true>(n,edges,start,n,directed,capture_queue);
}
} // namespace datamunge::algorithms
