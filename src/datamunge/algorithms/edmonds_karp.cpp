/// @file edmonds_karp.cpp
/// @brief Complete shortest-path BFS, checked residual pairs and actual Edmonds-Karp evidence.
#include <datamunge/algorithms/edmonds_karp.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Stateless graph validation, geometry and one shared residual traversal.
namespace edmonds_karp_detail {
/// @brief Oriented residual arc; paired arc identity is index XOR one.
struct Arc {
    std::size_t from; ///< Residual tail.
    std::size_t to; ///< Residual head.
    double residual; ///< Finite nonnegative pushable capacity.
};
/// @brief Missing first-discovery parent sentinel, distinct from every valid pool index.
constexpr auto no_parent=std::numeric_limits<std::size_t>::max();
/// @brief Add logical counts without wraparound.
/// @param a First count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error On overflow.
std::size_t add(std::size_t a,std::size_t b){
    if(b>no_parent-a)throw std::length_error("Edmonds-Karp: count overflow");return a+b;
}
/// @brief Check complete domain before graph allocation, including otherwise unreachable edges.
/// @param n Vertex count.
/// @param edges Original records.
/// @param source Source id.
/// @param sink Sink id.
/// @throws std::invalid_argument On invalid terminals, endpoints or nonfinite/negative capacities.
void validate(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
              std::size_t source,std::size_t sink){
    if(source>=n||sink>=n||source==sink)throw std::invalid_argument("Edmonds-Karp: distinct valid terminals required");
    for(const auto& [u,v,c]:edges)if(u>=n||v>=n||!std::isfinite(c)||c<0)
        throw std::invalid_argument("Edmonds-Karp: invalid finite capacity edge");
}
/// @brief Check residual-pool, vertex-array and minimum history geometry without allocation.
/// @param n Vertex count admitting terminals.
/// @param e Original edge count.
/// @return E+2+2V minimum trace entries.
/// @throws std::invalid_argument On fewer than two vertices.
/// @throws std::length_error On count or vector limits.
std::size_t geometry(std::size_t n,std::size_t e){
    if(n<2)throw std::invalid_argument("Edmonds-Karp: at least two vertices required");
    const auto arcs=add(e,e);
    if(arcs>std::vector<Arc>{}.max_size()||n>std::vector<std::vector<std::size_t>>{}.max_size()
       ||n>std::vector<std::size_t>{}.max_size()||e>std::vector<double>{}.max_size())
        throw std::length_error("Edmonds-Karp: graph exceeds container geometry");
    return add(add(e,2),add(n,n));
}
/// @brief Reserve a complete retained record before its allocation or traversal.
/// @param trace Optional owned capture.
/// @param entries Additional logical entries.
/// @param cap Optional nonzero quota.
/// @throws std::length_error On counter overflow or quota exhaustion.
void capture(EdmondsKarpTrace* trace,std::size_t entries,std::size_t cap){
    if(!trace)return;const auto total=add(trace->captured_entries,entries);
    if(cap&&total>cap)throw std::length_error("Edmonds-Karp: capture limit exceeded");trace->captured_entries=total;
}
/// @brief Check finite positive addition progress for totals and paired residuals.
/// @param before Existing finite nonnegative value.
/// @param amount Positive finite push.
/// @return Finite strictly larger result.
/// @throws std::overflow_error If result becomes nonfinite.
/// @throws std::runtime_error If a positive increment is lost to rounding.
double increase(double before,double amount){
    const double after=before+amount;
    if(!std::isfinite(after))throw std::overflow_error("Edmonds-Karp: positive sum overflow");
    if(after<=before)throw std::runtime_error("Edmonds-Karp: lost positive addition progress");return after;
}
/// @brief Check finite strictly decreasing nonnegative residual progress.
/// @param before Used positive residual.
/// @param amount Positive bottleneck no larger than before.
/// @return Updated residual.
/// @throws std::runtime_error If subtraction is negative, nonfinite or loses progress.
double decrease(double before,double amount){
    const double after=before-amount;
    if(!std::isfinite(after)||after<0||after>=before)throw std::runtime_error("Edmonds-Karp: lost residual subtraction progress");
    return after;
}
/// @brief Execute one checked miner with optional actual capture; no replay engine.
/// @param n Valid vertex count.
/// @param edges Valid borrowed input.
/// @param source Valid source.
/// @param sink Valid distinct sink.
/// @param trace Optional owned evidence initialized with its input cost.
/// @param cap Optional retained-entry cap, zero disables quota.
/// @return Owned ordinary result and checked actual work.
/// @throws std::length_error On counters, containers or trace quota.
/// @throws std::overflow_error On nonfinite sums.
/// @throws std::runtime_error On lost positive representable progress.
EdmondsKarpResult core(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
                       std::size_t source,std::size_t sink,EdmondsKarpTrace* trace,std::size_t cap){
    std::vector<Arc> arcs;arcs.reserve(add(edges.size(),edges.size()));
    std::vector<std::vector<std::size_t>> head(n);
    for(const auto& [u,v,c]:edges){const auto id=arcs.size();arcs.push_back({u,v,c});arcs.push_back({v,u,0});head[u].push_back(id);head[v].push_back(id+1);}
    std::vector<std::size_t> levels(n),parents(n),queue;queue.reserve(n);
    EdmondsKarpResult out;
    while(true){
        capture(trace,add(1,add(n,n)),cap);
        out.work.bfs_calls=add(out.work.bfs_calls,1);
        std::fill(levels.begin(),levels.end(),n);std::fill(parents.begin(),parents.end(),no_parent);
        levels[source]=0;queue.clear();queue.push_back(source);std::size_t inspections=0;
        for(std::size_t q=0;q<queue.size();++q){const auto u=queue[q];
            for(auto id:head[u]){inspections=add(inspections,1);const auto& a=arcs[id];
                if(a.residual>0&&levels[a.to]==n){levels[a.to]=levels[u]+1;parents[a.to]=id;queue.push_back(a.to);}}}
        out.work.bfs_arc_inspections=add(out.work.bfs_arc_inspections,inspections);
        const auto search=out.work.bfs_calls-1;
        if(trace)trace->searches.push_back({levels,parents,out.result.max_flow,inspections});
        if(levels[sink]==n){for(std::size_t v=0;v<n;++v)if(levels[v]!=n)out.result.min_cut_source_side.push_back(v);break;}
        std::vector<std::size_t> path;
        for(auto v=sink;v!=source;v=arcs[parents[v]].from)path.push_back(parents[v]);
        std::reverse(path.begin(),path.end());
        capture(trace,add(1,path.size()),cap);
        double amount=std::numeric_limits<double>::infinity();for(auto id:path)amount=std::min(amount,arcs[id].residual);
        const double next_flow=increase(out.result.max_flow,amount);
        EdmondsKarpAugmentation augmentation;
        if(trace){augmentation.search=search;augmentation.amount=amount;augmentation.flow_before=out.result.max_flow;augmentation.flow_after=next_flow;augmentation.path.reserve(path.size());}
        for(auto id:path){auto& a=arcs[id];auto& pair=arcs[id^1];const double after=decrease(a.residual,amount),paired=increase(pair.residual,amount);
            if(trace)augmentation.path.push_back({id/2,a.from,a.to,bool(id&1),a.residual,after,pair.residual,paired});
            a.residual=after;pair.residual=paired;
        }
        out.result.max_flow=next_flow;out.work.augmentations=add(out.work.augmentations,1);
        out.work.path_arc_updates=add(out.work.path_arc_updates,path.size());out.work.max_path_edges=std::max(out.work.max_path_edges,path.size());
        if(trace)trace->augmentations.push_back(std::move(augmentation));
    }
    out.result.edge_flow.reserve(edges.size());for(std::size_t i=0;i<edges.size();++i)out.result.edge_flow.push_back(arcs[2*i+1].residual);
    return out;
}
} // namespace edmonds_karp_detail

std::size_t edmonds_karp_trace_base_entries(std::size_t n,std::size_t edge_count){
    return edmonds_karp_detail::geometry(n,edge_count);
}
EdmondsKarpResult edmonds_karp_work(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,std::size_t source,std::size_t sink){
    edmonds_karp_detail::validate(n,edges,source,sink);(void)edmonds_karp_detail::geometry(n,edges.size());
    return edmonds_karp_detail::core(n,edges,source,sink,nullptr,0);
}
MaxFlowResult edmonds_karp(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,std::size_t source,std::size_t sink){
    return edmonds_karp_work(n,edges,source,sink).result;
}
EdmondsKarpTrace edmonds_karp_trace(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,std::size_t source,std::size_t sink,std::size_t max_entries){
    edmonds_karp_detail::validate(n,edges,source,sink);const auto minimum=edmonds_karp_detail::geometry(n,edges.size());
    if(max_entries&&minimum>max_entries)throw std::length_error("Edmonds-Karp: initial capture limit exceeded");
    EdmondsKarpTrace trace;trace.vertex_count=n;trace.edges=edges;trace.source=source;trace.sink=sink;
    trace.captured_entries=edmonds_karp_detail::add(edges.size(),1);
    auto result=edmonds_karp_detail::core(n,edges,source,sink,&trace,max_entries);trace.result=std::move(result.result);trace.work=result.work;return trace;
}
} // namespace datamunge::algorithms
