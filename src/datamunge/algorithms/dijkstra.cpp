/// @file dijkstra.cpp
/// @brief Shared checked lazy-heap traversal and bounded actual-event capture.
#include <datamunge/algorithms/dijkstra.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Stateless traversal implementation details.
namespace dijkstra_detail {
/// @brief Weighted input tuple, preserving original public argument type.
using Edge=std::tuple<std::size_t,std::size_t,double>;
/// @brief Lazy heap item, ordered by distance then vertex id.
using Entry=std::pair<double,std::size_t>;
/// @brief Checks the complete borrowed graph before allocating adjacency.
/// @param n Vertex count.
/// @param edges Input arcs.
/// @param source Requested source.
/// @throws std::invalid_argument Invalid id or nonfinite/negative weight.
void validate(std::size_t n,const std::vector<Edge>& edges,std::size_t source){
    if(source>=n)throw std::invalid_argument("dijkstra: source vertex out of range");
    for(const auto& [u,v,w]:edges){
        if(u>=n || v>=n)throw std::invalid_argument("dijkstra: edge endpoint out of range");
        if(!std::isfinite(w) || w<0)throw std::invalid_argument("dijkstra: finite nonnegative weights required");
    }
}
/// @brief Increments a logical counter without wrapping.
/// @param count Counter to change.
/// @throws std::length_error Counter is already SIZE_MAX.
void tick(std::size_t& count){
    if(count==std::numeric_limits<std::size_t>::max())throw std::length_error("dijkstra: counter overflow");
    ++count;
}
/// @brief Reserves the next history event before the corresponding traversal operation.
/// @param trace Optional capture destination.
/// @param used Prior captured-event count, changed only when capture is enabled.
/// @param cap Optional nonzero event budget.
/// @throws std::length_error Budget or counter exhausted.
void event(DijkstraTrace* trace,std::size_t& used,std::size_t cap){
    if(!trace)return;
    if(cap && used>=cap)throw std::length_error("dijkstra: actual-event cap exceeded");
    tick(used);
}
/// @brief Runs the same strict-improvement traversal for plain and traced APIs.
/// @param n Valid vertex count.
/// @param edges Valid input arcs.
/// @param source Valid source.
/// @param trace Optional owned-history destination; null avoids event allocation.
/// @param cap Optional event cap, used only with trace.
/// @return Owned shortest-path result.
/// @throws std::overflow_error An examined candidate is nonfinite.
/// @throws std::length_error History budget, counter or container capacity exceeded.
ShortestPathResult run(std::size_t n,const std::vector<Edge>& edges,std::size_t source,DijkstraTrace* trace,std::size_t cap){
    std::vector<std::vector<std::size_t>> adjacency(n);
    for(std::size_t e=0;e<edges.size();++e)adjacency[std::get<0>(edges[e])].push_back(e);
    ShortestPathResult r;r.distance.assign(n,std::numeric_limits<double>::infinity());
    r.predecessor.assign(n,kNoPredecessor);r.distance[source]=0;
    DijkstraWork work;work.input_edges=edges.size();work.heap_pushes=1;work.max_heap_size=1;
    std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> heap;heap.emplace(0,source);
    std::size_t events=0;
    if(trace){trace->source=source;trace->predecessor_edge.assign(n,kNoPredecessor);}
    while(!heap.empty()){
        event(trace,events,cap);
        const auto before=heap.size();const auto [d,u]=heap.top();heap.pop();tick(work.heap_pops);
        const bool stale=d>r.distance[u];
        DijkstraPop pop{u,d,r.distance[u],stale,before,0,trace?trace->relaxations.size():0,0};
        if(stale)tick(work.stale_pops);
        else {
            tick(work.settled_vertices);if(trace)trace->settled_order.push_back(u);
            for(auto e:adjacency[u]){
                event(trace,events,cap);tick(work.edges_examined);
                const auto [from,v,w]=edges[e];const double previous=r.distance[v],candidate=d+w;
                if(!std::isfinite(candidate))throw std::overflow_error("dijkstra: examined route sum overflow");
                const auto parent=r.predecessor[v];const bool improved=candidate<previous;
                if(improved){
                    r.distance[v]=candidate;r.predecessor[v]=u;heap.emplace(candidate,v);
                    tick(work.successful_relaxations);tick(work.heap_pushes);
                    work.max_heap_size=std::max(work.max_heap_size,heap.size());
                    if(trace)trace->predecessor_edge[v]=e;
                }
                if(trace)trace->relaxations.push_back({e,from,v,w,d,previous,candidate,parent,improved,heap.size()});
            }
        }
        if(trace){pop.heap_after=heap.size();pop.relaxation_count=trace->relaxations.size()-pop.relaxation_begin;trace->pops.push_back(pop);}
    }
    if(trace)trace->work=work;
    return r;
}
} // namespace dijkstra_detail
ShortestPathResult dijkstra(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,std::size_t source){
    dijkstra_detail::validate(n,edges,source);return dijkstra_detail::run(n,edges,source,nullptr,0);
}
DijkstraTrace dijkstra_trace(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,std::size_t source,std::size_t max_events){
    dijkstra_detail::validate(n,edges,source);DijkstraTrace t;t.result=dijkstra_detail::run(n,edges,source,&t,max_events);return t;
}
std::vector<std::size_t> reconstruct_path(const ShortestPathResult& result,std::size_t target){
    const auto n=result.distance.size();
    if(target>=n || result.distance[target]==std::numeric_limits<double>::infinity())return {};
    if(result.predecessor.size()!=n)throw std::invalid_argument("dijkstra path: mismatched vector sizes");
    std::vector<std::size_t> path;
    for(std::size_t v=target;v!=kNoPredecessor;v=result.predecessor[v]){
        if(v>=n || path.size()>=n)throw std::invalid_argument("dijkstra path: invalid parent or cycle");
        if(!std::isfinite(result.distance[v]) || result.distance[v]<0)throw std::invalid_argument("dijkstra path: invalid visited distance");
        path.push_back(v);
    }
    std::reverse(path.begin(),path.end());return path;
}
std::vector<std::tuple<std::size_t,std::size_t,double>> dijkstra_parallel_workload(std::size_t count,std::size_t max_edges){
    using dijkstra_detail::Edge;
    if(count>9007199254740992ULL || (max_edges && count>max_edges) || count>std::vector<Edge>().max_size())
        throw std::length_error("dijkstra workload: complete edge/capacity limit");
    std::vector<Edge> result;result.reserve(count);
    for(std::size_t i=0;i<count;++i)result.emplace_back(0,1,static_cast<double>(count-i));
    return result;
}
} // namespace datamunge::algorithms
