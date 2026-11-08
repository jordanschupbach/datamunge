/// @file dinic.cpp
/// @brief Iterative current-arc Dinic, checked residual pairs and independent flow/cut diagnostics.
#include <datamunge/algorithms/dinic.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Stateless residual traversal, arithmetic and fixture helpers.
namespace dinic_detail {
/// @brief One oriented residual arc in a flat pool; its opposite index is index XOR one.
struct Arc {
    std::size_t from; ///< Residual tail.
    std::size_t to; ///< Residual head.
    double residual; ///< Remaining pushable capacity.
};
/// @brief Checks the complete graph before allocating adjacency.
/// @param n Vertex count.
/// @param edges Original input.
/// @param source Source id.
/// @param sink Sink id.
/// @throws std::invalid_argument Invalid terminal, endpoint or capacity.
void validate(std::size_t n,const std::vector<CapacityEdge>& edges,std::size_t source,std::size_t sink){
    if(source>=n || sink>=n || source==sink)throw std::invalid_argument("Dinic: distinct valid terminals required");
    for(const auto& [u,v,c]:edges)if(u>=n || v>=n || !std::isfinite(c) || c<0)throw std::invalid_argument("Dinic: invalid finite capacity edge");
}
/// @brief Checked size addition.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error Sum overflows size_t.
std::size_t add(std::size_t a,std::size_t b){
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("Dinic: count overflow");return a+b;
}
/// @brief Checked size multiplication.
/// @param a First factor.
/// @param b Second factor.
/// @return Exact product.
/// @throws std::length_error Product overflows size_t.
std::size_t multiply(std::size_t a,std::size_t b){
    if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("Dinic: count overflow");return a*b;
}
/// @brief Reserves history entries before a search or path update.
/// @param trace Optional destination.
/// @param used Prior count, updated if capture is enabled.
/// @param increment Next complete record cost.
/// @param cap Optional nonzero history cap.
/// @throws std::length_error Count or optional budget exceeded.
void budget(DinicTrace* trace,std::size_t& used,std::size_t increment,std::size_t cap){
    if(!trace)return;const auto next=add(used,increment);
    if(cap && next>cap)throw std::length_error("Dinic: actual history entry cap exceeded");used=next;
}
/// @brief Checks finite representable positive addition progress.
/// @param before Nonnegative value before a positive push.
/// @param amount Positive bottleneck.
/// @return Finite strictly larger sum.
/// @throws std::overflow_error Sum is nonfinite.
/// @throws std::runtime_error Positive amount is lost to rounding.
double increase(double before,double amount){
    const double after=before+amount;
    if(!std::isfinite(after))throw std::overflow_error("Dinic: flow/residual sum overflow");
    if(after<=before)throw std::runtime_error("Dinic: lost positive addition progress");return after;
}
/// @brief Executes BFS phases and iterative DFS path pushes on distinct residual pairs.
/// @param n Valid vertex count.
/// @param edges Valid capacities.
/// @param source Valid source.
/// @param sink Valid sink.
/// @param trace Optional actual-history destination.
/// @param cap Optional trace budget.
/// @return Owned legacy-layout maximum-flow result.
/// @throws std::length_error Pool, counter or capture capacity exceeded.
/// @throws std::overflow_error Nonfinite arithmetic.
/// @throws std::runtime_error Lost positive residual progress.
MaxFlowResult run(std::size_t n,const std::vector<CapacityEdge>& edges,std::size_t source,std::size_t sink,DinicTrace* trace,std::size_t cap){
    if(edges.size()>std::vector<Arc>().max_size()/2)throw std::length_error("Dinic: residual pool capacity");
    std::vector<Arc> arcs;arcs.reserve(2*edges.size());std::vector<std::vector<std::size_t>> adjacency(n);
    for(std::size_t i=0;i<edges.size();++i){const auto [u,v,c]=edges[i];
        adjacency[u].push_back(arcs.size());arcs.push_back({u,v,c});
        adjacency[v].push_back(arcs.size());arcs.push_back({v,u,0});
    }
    DinicWork work;double flow=0;std::size_t entries=0;
    std::vector<std::size_t> level(n),current(n),queue,vertices,path;
    std::vector<double> bottleneck;
    while(true){
        budget(trace,entries,trace?add(n,1):0,cap);
        const auto bfs_before=work.bfs_arc_inspections,dfs_before=work.dfs_arc_inspections,advance_before=work.current_arc_advances;
        work.bfs_calls=add(work.bfs_calls,1);std::fill(level.begin(),level.end(),n);level[source]=0;queue.clear();queue.push_back(source);
        for(std::size_t h=0;h<queue.size();++h){const auto u=queue[h];
            for(auto ai:adjacency[u]){work.bfs_arc_inspections=add(work.bfs_arc_inspections,1);const auto& a=arcs[ai];
                if(a.residual>0 && level[a.to]==n){level[a.to]=level[u]+1;queue.push_back(a.to);}}
        }
        const std::size_t phase=trace?trace->phases.size():0;
        if(trace)trace->phases.push_back({level,level[sink],trace->augmentations.size(),0,flow,flow,work.bfs_arc_inspections-bfs_before,0,0});
        if(level[sink]==n)break;
        work.blocking_phases=add(work.blocking_phases,1);std::fill(current.begin(),current.end(),0);
        while(true){
            vertices.clear();path.clear();bottleneck.clear();vertices.push_back(source);bottleneck.push_back(std::numeric_limits<double>::infinity());
            while(!vertices.empty() && vertices.back()!=sink){const auto u=vertices.back();bool descend=false;
                while(current[u]<adjacency[u].size()){
                    const auto ai=adjacency[u][current[u]];const auto& a=arcs[ai];work.dfs_arc_inspections=add(work.dfs_arc_inspections,1);
                    if(a.residual>0 && level[a.to]==level[u]+1){path.push_back(ai);vertices.push_back(a.to);bottleneck.push_back(std::min(bottleneck.back(),a.residual));descend=true;break;}
                    ++current[u];work.current_arc_advances=add(work.current_arc_advances,1);
                }
                if(!descend){vertices.pop_back();bottleneck.pop_back();if(!path.empty()){
                    path.pop_back();++current[vertices.back()];work.current_arc_advances=add(work.current_arc_advances,1);}}
            }
            if(vertices.empty())break;
            budget(trace,entries,trace?add(path.size(),1):0,cap);
            const double amount=bottleneck.back(),next_flow=increase(flow,amount);
            // Validate the complete path update before mutating any residual pair.
            for(auto ai:path){const auto& a=arcs[ai];const double next=a.residual-amount;
                if(!std::isfinite(next) || next<0)throw std::overflow_error("Dinic: invalid residual subtraction");
                if(next>=a.residual)throw std::runtime_error("Dinic: lost positive subtraction progress");
                (void)increase(arcs[ai^1].residual,amount);
            }
            DinicAugmentation event{phase,amount,flow,next_flow,{}};if(trace)event.path.reserve(path.size());
            for(auto ai:path){auto& a=arcs[ai];auto& paired=arcs[ai^1];
                if(trace)event.path.push_back({ai/2,a.from,a.to,static_cast<bool>(ai%2),a.residual,a.residual-amount,paired.residual,paired.residual+amount});
                a.residual-=amount;paired.residual+=amount;
            }
            flow=next_flow;work.augmentations=add(work.augmentations,1);work.path_arc_updates=add(work.path_arc_updates,path.size());work.max_path_edges=std::max(work.max_path_edges,path.size());
            if(trace){trace->augmentations.push_back(std::move(event));auto& p=trace->phases.back();++p.augmentation_count;p.flow_after=flow;}
        }
        if(trace){auto& p=trace->phases.back();p.dfs_arc_inspections=work.dfs_arc_inspections-dfs_before;p.current_arc_advances=work.current_arc_advances-advance_before;}
    }
    MaxFlowResult result;result.max_flow=flow;result.min_cut_source_side=std::move(queue);result.edge_flow.reserve(edges.size());
    for(std::size_t i=0;i<edges.size();++i)result.edge_flow.push_back(arcs[2*i+1].residual);
    if(trace)trace->work=work;return result;
}
/// @brief Converts a wide finite numerical diagnostic to representable double.
/// @param value Wide signed summary.
/// @return Finite double.
/// @throws std::overflow_error Nonfinite or outside double range.
double narrow(long double value){
    if(!std::isfinite(value) || std::abs(value)>std::numeric_limits<double>::max())throw std::overflow_error("flow certificate: summary overflow");return static_cast<double>(value);
}
/// @brief Checks complete generated-network counts and capacity before allocation.
/// @param n Vertex count.
/// @param m Edge count.
/// @param capacity Common capacity.
/// @param cap Optional edge cap.
/// @throws std::invalid_argument Nonfinite/negative capacity.
/// @throws std::length_error Container or optional cap exceeded.
void generated_counts(std::size_t n,std::size_t m,double capacity,std::size_t cap){
    if(!std::isfinite(capacity) || capacity<0)throw std::invalid_argument("flow paths: finite nonnegative capacity required");
    if((cap && m>cap) || m>std::vector<CapacityEdge>().max_size() || n>std::vector<std::size_t>().max_size())throw std::length_error("flow paths: complete network count limit");
}
/// @brief Emits independent paths after checked output sizing.
/// @param paths Number of paths.
/// @param length Common length, ignored for staircase.
/// @param capacity Common capacity.
/// @param staircase Whether lengths are one through paths.
/// @param n Prechecked vertex count.
/// @param m Prechecked edge count.
/// @return Owned ordered network.
FlowNetwork generate(std::size_t paths,std::size_t length,double capacity,bool staircase,std::size_t n,std::size_t m){
    FlowNetwork g{n,{},0,1};g.edges.reserve(m);std::size_t next=2;
    for(std::size_t p=0;p<paths;++p){const auto len=staircase?p+1:length;std::size_t from=0;
        for(std::size_t e=0;e<len;++e){const auto to=e+1==len?1:next++;g.edges.emplace_back(from,to,capacity);from=to;}}
    return g;
}
} // namespace dinic_detail
MaxFlowResult dinic_max_flow(std::size_t vertex_count,const std::vector<CapacityEdge>& edges,std::size_t source,std::size_t sink){
    dinic_detail::validate(vertex_count,edges,source,sink);return dinic_detail::run(vertex_count,edges,source,sink,nullptr,0);
}
DinicTrace dinic_max_flow_trace(std::size_t vertex_count,const std::vector<CapacityEdge>& edges,std::size_t source,std::size_t sink,std::size_t max_trace_entries){
    dinic_detail::validate(vertex_count,edges,source,sink);DinicTrace trace;trace.network={vertex_count,edges,source,sink};
    trace.result=dinic_detail::run(vertex_count,edges,source,sink,&trace,max_trace_entries);return trace;
}
FlowCertificate max_flow_certificate(std::size_t vertex_count,const std::vector<CapacityEdge>& edges,std::size_t source,std::size_t sink,const MaxFlowResult& result,double tolerance){
    dinic_detail::validate(vertex_count,edges,source,sink);
    if(!std::isfinite(tolerance) || tolerance<0 || !std::isfinite(result.max_flow) || result.edge_flow.size()!=edges.size())throw std::invalid_argument("flow certificate: invalid controls/result");
    std::vector<bool> side(vertex_count,false);for(auto v:result.min_cut_source_side){
        if(v>=vertex_count || side[v])throw std::invalid_argument("flow certificate: invalid/duplicate cut vertex");side[v]=true;}
    std::vector<long double> net(vertex_count,0);long double capacity_error=0,cut=0,conservation=0;
    FlowCertificate c{};
    for(std::size_t e=0;e<edges.size();++e){const auto [u,v,capacity]=edges[e];const long double f=result.edge_flow[e];
        if(!std::isfinite(f))throw std::invalid_argument("flow certificate: nonfinite edge flow");
        capacity_error=std::max({capacity_error,-f,f-capacity});if(u!=v){net[u]+=f;net[v]-=f;}
        if(side[u] && !side[v]){cut+=capacity;c.crossing_edges.push_back(e);}
    }
    for(std::size_t v=0;v<vertex_count;++v)if(v!=source && v!=sink)conservation=std::max(conservation,std::abs(net[v]));
    c.cut_capacity=dinic_detail::narrow(cut);c.max_capacity_violation=dinic_detail::narrow(capacity_error);c.max_conservation_error=dinic_detail::narrow(conservation);
    c.source_net_flow=dinic_detail::narrow(net[source]);c.sink_net_flow=dinic_detail::narrow(-net[sink]);
    c.value_balance_error=dinic_detail::narrow(std::max(std::abs(net[source]-result.max_flow),std::abs(-net[sink]-result.max_flow)));
    c.duality_gap=dinic_detail::narrow(cut-result.max_flow);c.cut_separates=side[source] && !side[sink];
    c.feasible=c.max_capacity_violation<=tolerance && c.max_conservation_error<=tolerance && c.value_balance_error<=tolerance && result.max_flow>=-tolerance;
    c.optimal_within_tolerance=c.feasible && c.cut_separates && std::abs(c.duality_gap)<=tolerance;return c;
}
FlowNetwork dinic_disjoint_paths(std::size_t paths,std::size_t length,double capacity,std::size_t max_edges){
    if(!length || !std::isfinite(capacity) || capacity<0)throw std::invalid_argument("flow paths: invalid length/capacity");
    const auto m=dinic_detail::multiply(paths,length),n=dinic_detail::add(2,dinic_detail::multiply(paths,length-1));
    dinic_detail::generated_counts(n,m,capacity,max_edges);return dinic_detail::generate(paths,length,capacity,false,n,m);
}
FlowNetwork dinic_staircase_paths(std::size_t paths,std::size_t max_edges){
    auto a=paths,b=dinic_detail::add(paths,1);if(a%2==0)a/=2;else b/=2;
    const auto m=dinic_detail::multiply(a,b),n=dinic_detail::add(2,m-paths);
    dinic_detail::generated_counts(n,m,1,max_edges);return dinic_detail::generate(paths,0,1,true,n,m);
}
} // namespace datamunge::algorithms
