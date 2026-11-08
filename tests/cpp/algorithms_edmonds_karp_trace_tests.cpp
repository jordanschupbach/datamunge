#include <datamunge/algorithms/flow_diagram.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Edge=std::tuple<std::size_t,std::size_t,double>;
const std::vector<Edge> reroute{{0,1,1},{0,2,1},{1,3,1},{1,4,1},{2,3,1},{3,5,1},{4,5,1}};
const std::vector<Edge> classic{{0,1,16},{0,2,13},{1,2,10},{2,1,4},{1,3,12},{3,2,9},{2,4,14},{4,3,7},{3,5,20},{4,5,4}};
double min_cut(std::size_t n,const std::vector<Edge>& edges,std::size_t source,std::size_t sink){
    double best=std::numeric_limits<double>::infinity();
    for(std::size_t mask=0;mask<(std::size_t{1}<<n);++mask)if((mask>>source&1)&&!(mask>>sink&1)){
        double capacity=0;for(const auto& [u,v,c]:edges)if((mask>>u&1)&&!(mask>>v&1))capacity+=c;best=std::min(best,capacity);}
    return best;
}
void certificate(const EdmondsKarpTrace& t){
    const auto n=t.vertex_count,e=t.edges.size(),none=std::numeric_limits<std::size_t>::max();
    ASSERT_EQ(t.searches.size(),t.augmentations.size()+1);std::vector<double> residual(2*e);
    const auto endpoints=[&](std::size_t id){const auto [u,v,c]=t.edges[id/2];return id&1?std::pair{v,u}:std::pair{u,v};};
    for(std::size_t i=0;i<e;++i)residual[2*i]=std::get<2>(t.edges[i]);
    double flow=0;std::size_t inspections=0,updates=0,max_path=0;std::vector<std::size_t> previous(n,0);
    std::size_t cost=e+1;
    for(std::size_t k=0;k<t.searches.size();++k){const auto& search=t.searches[k];ASSERT_EQ(search.levels.size(),n);ASSERT_EQ(search.parents.size(),n);
        // Independent all-pairs unit distances, rather than the miner's queue traversal.
        std::vector<std::vector<std::size_t>> distance(n,std::vector<std::size_t>(n,n));for(std::size_t v=0;v<n;++v)distance[v][v]=0;
        for(std::size_t id=0;id<residual.size();++id)if(residual[id]>0){const auto [u,v]=endpoints(id);distance[u][v]=std::min(distance[u][v],std::size_t{1});}
        for(std::size_t mid=0;mid<n;++mid)for(std::size_t u=0;u<n;++u)for(std::size_t v=0;v<n;++v)distance[u][v]=std::min(distance[u][v],distance[u][mid]+distance[mid][v]);
        const auto& levels=distance[t.source];EXPECT_EQ(search.levels,levels);EXPECT_DOUBLE_EQ(search.flow_before,flow);EXPECT_EQ(search.parents[t.source],none);
        // First BFS discovery is the lexicographically first shortest arc-id path.
        std::vector<std::vector<std::size_t>> best(n);
        for(std::size_t depth=1;depth<n;++depth)for(std::size_t v=0;v<n;++v)if(levels[v]==depth){
            bool found=false;std::vector<std::size_t> chosen;
            for(std::size_t id=0;id<residual.size();++id){const auto [u,to]=endpoints(id);if(to!=v||residual[id]<=0||levels[u]+1!=depth)continue;
                auto candidate=best[u];candidate.push_back(id);if(!found||candidate<chosen){chosen=std::move(candidate);found=true;}}
            ASSERT_TRUE(found);EXPECT_EQ(search.parents[v],chosen.back());best[v]=std::move(chosen);
        }
        std::size_t visits=0;for(std::size_t id=0;id<residual.size();++id){const auto [u,v]=endpoints(id);if(levels[u]!=n)++visits;}
        EXPECT_EQ(search.bfs_arc_inspections,visits);inspections+=visits;cost+=1+2*n;
        for(std::size_t v=0;v<n;++v){EXPECT_GE(levels[v],previous[v]);if(levels[v]==n)EXPECT_EQ(search.parents[v],none);}previous=levels;
        if(k==t.augmentations.size()){EXPECT_EQ(levels[t.sink],n);std::vector<std::size_t> cut;for(std::size_t v=0;v<n;++v)if(levels[v]!=n)cut.push_back(v);EXPECT_EQ(t.result.min_cut_source_side,cut);continue;}
        const auto& a=t.augmentations[k];EXPECT_EQ(a.search,k);EXPECT_DOUBLE_EQ(a.flow_before,flow);EXPECT_EQ(a.path.size(),levels[t.sink]);ASSERT_FALSE(a.path.empty());
        std::size_t vertex=t.source;double bottleneck=std::numeric_limits<double>::infinity();
        for(const auto& arc:a.path){const auto id=2*arc.input_edge+arc.reverse;ASSERT_LT(id,residual.size());const auto [u,v]=endpoints(id);EXPECT_EQ(u,vertex);EXPECT_EQ(arc.from,u);EXPECT_EQ(arc.to,v);EXPECT_EQ(search.parents[v],id);EXPECT_EQ(levels[v],levels[u]+1);vertex=v;bottleneck=std::min(bottleneck,residual[id]);}
        EXPECT_EQ(vertex,t.sink);EXPECT_DOUBLE_EQ(a.amount,bottleneck);EXPECT_GT(a.amount,0);
        for(const auto& arc:a.path){const auto id=2*arc.input_edge+arc.reverse;EXPECT_DOUBLE_EQ(arc.residual_before,residual[id]);EXPECT_DOUBLE_EQ(arc.paired_before,residual[id^1]);
            residual[id]-=a.amount;residual[id^1]+=a.amount;EXPECT_DOUBLE_EQ(arc.residual_after,residual[id]);EXPECT_DOUBLE_EQ(arc.paired_after,residual[id^1]);}
        flow+=a.amount;EXPECT_DOUBLE_EQ(a.flow_after,flow);updates+=a.path.size();max_path=std::max(max_path,a.path.size());cost+=1+a.path.size();
    }
    EXPECT_EQ(t.work.bfs_calls,t.searches.size());EXPECT_EQ(t.work.bfs_arc_inspections,inspections);EXPECT_EQ(t.work.augmentations,t.augmentations.size());EXPECT_EQ(t.work.path_arc_updates,updates);EXPECT_EQ(t.work.max_path_edges,max_path);EXPECT_EQ(t.captured_entries,cost);EXPECT_DOUBLE_EQ(t.result.max_flow,flow);
    ASSERT_EQ(t.result.edge_flow.size(),e);for(std::size_t i=0;i<e;++i)EXPECT_DOUBLE_EQ(t.result.edge_flow[i],residual[2*i+1]);
    const auto c=max_flow_certificate(n,t.edges,t.source,t.sink,t.result);EXPECT_TRUE(c.feasible);EXPECT_TRUE(c.optimal_within_tolerance);
    EXPECT_DOUBLE_EQ(flow,min_cut(n,t.edges,t.source,t.sink));const auto plain=edmonds_karp(n,t.edges,t.source,t.sink);EXPECT_EQ(plain.edge_flow,t.result.edge_flow);EXPECT_EQ(plain.min_cut_source_side,t.result.min_cut_source_side);EXPECT_DOUBLE_EQ(plain.max_flow,flow);
}
}
TEST(EdmondsKarpTrace, ExhaustiveGraphsAllTerminalsMatchIndependentCutsAndDistances){
    std::size_t cases=0;
    for(std::size_t n:{3u,4u}){const unsigned radix=n==3?3:2,total=n==3?729:4096;
        for(unsigned code=0;code<total;++code){unsigned digits=code;std::vector<Edge> edges;
            for(std::size_t u=0;u<n;++u)for(std::size_t v=0;v<n;++v)if(u!=v){edges.emplace_back(u,v,digits%radix);digits/=radix;}
            for(std::size_t s=0;s<n;++s)for(std::size_t sink=0;sink<n;++sink)if(s!=sink){certificate(edmonds_karp_trace(n,edges,s,sink));++cases;}}}
    EXPECT_EQ(cases,53526u);
}
TEST(EdmondsKarpTrace, SeededPairsArbitraryTerminalsAndActualCancellation){
    std::mt19937 rng(20261107);
    for(int trial=0;trial<600;++trial){const std::size_t n=2+rng()%7,e=rng()%35,source=rng()%n;const auto sink=(source+1+rng()%(n-1))%n;std::vector<Edge> edges;
        for(std::size_t i=0;i<e;++i)edges.emplace_back(rng()%n,rng()%n,(rng()%33)/4.);
        const auto t=edmonds_karp_trace(n,edges,source,sink);certificate(t);EXPECT_DOUBLE_EQ(t.result.max_flow,dinic_max_flow(n,edges,source,sink).max_flow);}
    const auto t=edmonds_karp_trace(6,reroute,0,5);certificate(t);ASSERT_EQ(t.augmentations.size(),2u);ASSERT_EQ(t.augmentations[1].path.size(),5u);
    const auto& arc=t.augmentations[1].path[2];EXPECT_TRUE(arc.reverse);EXPECT_EQ(arc.input_edge,2u);EXPECT_EQ(arc.from,3u);EXPECT_EQ(arc.to,1u);EXPECT_DOUBLE_EQ(arc.residual_before,1);EXPECT_DOUBLE_EQ(arc.residual_after,0);EXPECT_DOUBLE_EQ(t.result.edge_flow[2],0);EXPECT_EQ(t.work.bfs_arc_inspections,30u);EXPECT_EQ(t.captured_entries,57u);
    const auto c=edmonds_karp_trace(6,classic,0,5);certificate(c);EXPECT_DOUBLE_EQ(c.result.max_flow,23);EXPECT_EQ(c.work.bfs_arc_inspections,74u);EXPECT_EQ(c.captured_entries,76u);
}
TEST(EdmondsKarpTrace, ExactQuotasEmptyGraphsAndIndependentOwnership){
    std::vector<Edge> edges{{0,1,3}};EXPECT_EQ(edmonds_karp_trace_base_entries(2,1),7u);const auto t=edmonds_karp_trace(2,edges,0,1,14);EXPECT_EQ(t.captured_entries,14u);
    for(std::size_t cap=1;cap<14;++cap){EXPECT_THROW((void)edmonds_karp_trace(2,edges,0,1,cap),std::length_error);}
    EXPECT_EQ(edmonds_karp_trace(2,edges,0,1,0).captured_entries,14u);
    auto copy=t;edges.clear();copy.edges.clear();copy.searches[0].parents[1]=99;copy.augmentations[0].path[0].paired_after=99;EXPECT_EQ(t.edges.size(),1u);EXPECT_EQ(t.searches[0].parents[1],0u);EXPECT_DOUBLE_EQ(t.augmentations[0].path[0].paired_after,3);
    const auto empty=edmonds_karp_trace(3,{},0,2,8);certificate(empty);EXPECT_EQ(empty.captured_entries,8u);EXPECT_EQ(empty.searches.size(),1u);EXPECT_EQ(empty.work.bfs_arc_inspections,0u);
    const auto maximum=std::numeric_limits<std::size_t>::max();
    EXPECT_THROW((void)edmonds_karp_trace(maximum,{},0,1,1),std::length_error);
    EXPECT_THROW((void)edmonds_karp_trace(100000000,{},0,1,1),std::length_error);
    EXPECT_THROW((void)edmonds_karp_trace_base_entries(2,maximum),std::length_error);
    EXPECT_THROW((void)edmonds_karp_trace_base_entries(1,0),std::invalid_argument);
}
TEST(EdmondsKarpTrace, FullDomainAndFloatingProgressAreChecked){
    const auto maximum=std::numeric_limits<std::size_t>::max();const double inf=std::numeric_limits<double>::infinity();
    for(double bad:{-1.,inf,-inf,std::numeric_limits<double>::quiet_NaN()}){
        EXPECT_THROW((void)edmonds_karp(4,{{2,3,bad}},0,1),std::invalid_argument);
        EXPECT_THROW((void)edmonds_karp_trace(4,{{2,3,bad}},0,1,1),std::invalid_argument);
    }
    EXPECT_THROW((void)edmonds_karp(maximum,{{0,1,-1}},0,1),std::invalid_argument);
    EXPECT_THROW((void)edmonds_karp(0,{},0,1),std::invalid_argument);
    EXPECT_THROW((void)edmonds_karp(2,{},0,0),std::invalid_argument);
    EXPECT_THROW((void)edmonds_karp(2,{{0,2,1}},0,1),std::invalid_argument);
    EXPECT_THROW((void)edmonds_karp(2,{{0,1,1e308},{0,1,1e308}},0,1),std::overflow_error);
    EXPECT_THROW((void)edmonds_karp(2,{{0,1,1e300},{0,1,1}},0,1),std::runtime_error);
    EXPECT_THROW((void)edmonds_karp(4,{{0,1,1e300},{1,3,1},{0,2,1},{2,3,1e300}},0,3),std::runtime_error);
    const double tiny=std::numeric_limits<double>::denorm_min(),big=std::numeric_limits<double>::max();
    EXPECT_DOUBLE_EQ(edmonds_karp(2,{{0,1,tiny}},0,1).max_flow,tiny);EXPECT_DOUBLE_EQ(edmonds_karp(2,{{0,1,big}},0,1).max_flow,big);
    const auto zero=edmonds_karp_trace(2,{{0,0,4},{0,1,-0.},{1,0,7},{1,1,5}},0,1);certificate(zero);EXPECT_DOUBLE_EQ(zero.result.max_flow,0);
}
TEST(EdmondsKarpTrace, ExactParallelWorkAndLongPathsStayIterative){
    for(std::size_t k:{1,2,4,8,16,32,64}){const auto g=dinic_disjoint_paths(k,1);const auto t=edmonds_karp_trace(g.vertex_count,g.edges,g.source,g.sink);
        EXPECT_DOUBLE_EQ(t.result.max_flow,k);EXPECT_EQ(t.work.bfs_calls,k+1);EXPECT_EQ(t.work.augmentations,k);EXPECT_EQ(t.work.bfs_arc_inspections,2*k*k+k);EXPECT_EQ(t.work.path_arc_updates,k);EXPECT_EQ(t.work.max_path_edges,1u);EXPECT_EQ(t.captured_entries,8*k+6);certificate(t);}
    const auto g=dinic_disjoint_paths(1,20000);const auto t=edmonds_karp_trace(g.vertex_count,g.edges,g.source,g.sink);
    EXPECT_DOUBLE_EQ(t.result.max_flow,1);EXPECT_EQ(t.work.bfs_calls,2u);EXPECT_EQ(t.work.max_path_edges,20000u);EXPECT_EQ(t.work.path_arc_updates,20000u);EXPECT_EQ(t.work.bfs_arc_inspections,40001u);EXPECT_EQ(t.captured_entries,120008u);
}
TEST(FlowDiagram, DistinctEdgeLanesLoopsLocalValidationAndOwnedPrimitives){
    FlowNetwork g{2,{{0,0,7},{0,1,3},{0,1,2},{1,0,9}},0,1};const auto r=edmonds_karp(2,g.edges,0,1);const std::vector<datamunge::graphics::DiagramPoint> positions{{0,0},{7,0}};
    const auto d=plot_max_flow(g,r,positions);const auto svg=d.to_svg();for(std::size_t e=0;e<g.edges.size();++e)EXPECT_NE(svg.find("e"+std::to_string(e)+":"),std::string::npos);EXPECT_NE(svg.find("s=0"),std::string::npos);EXPECT_NE(svg.find("t=1"),std::string::npos);
    const auto plain=datamunge::graphics::Diagram().path({{0,0},{4,0}}).text({2,0},"edge").to_svg();
    const auto backed=datamunge::graphics::Diagram().path({{0,0},{4,0}}).label({2,0},"edge").to_svg();
    const auto white_rects=[](const std::string& xml){std::size_t count=0,pos=0;while((pos=xml.find("fill=\"white\"",pos))!=std::string::npos){++count;pos+=12;}return count;};
    EXPECT_EQ(white_rects(backed),white_rects(plain)+1);
    EXPECT_NE(backed.find("aria-label=\"edge\""),std::string::npos);
    EXPECT_THROW((void)datamunge::graphics::Diagram().label({0,0},"edge",0),std::invalid_argument);
    g.edges.clear();EXPECT_EQ(d.to_svg(),svg);
    const FlowNetwork one{2,{{0,1,3}},0,1};const auto result=edmonds_karp(2,one.edges,0,1);
    EXPECT_THROW((void)plot_max_flow(one,result,{}),std::invalid_argument);
    EXPECT_THROW((void)plot_max_flow(one,result,{{0,0},{0,0}}),std::invalid_argument);
    EXPECT_THROW((void)plot_max_flow(one,result,{{0,0},{.1,0}}),std::invalid_argument);
    EXPECT_THROW((void)plot_max_flow(one,result,{{0,0},{std::numeric_limits<double>::infinity(),0}}),std::invalid_argument);
    EXPECT_THROW((void)plot_max_flow(one,MaxFlowResult{3,{},{0}},positions),std::invalid_argument);
    EXPECT_THROW((void)plot_max_flow(one,MaxFlowResult{3,{3},{0,0}},positions),std::invalid_argument);
    EXPECT_THROW((void)plot_max_flow(one,MaxFlowResult{3,{3},{2}},positions),std::invalid_argument);
    EXPECT_THROW((void)plot_max_flow(FlowNetwork{2,{{0,1,-1}},0,1},result,positions),std::invalid_argument);
    // Structurally valid edited numbers are rendered, not silently certified as feasible.
    EXPECT_NO_THROW((void)plot_max_flow(one,MaxFlowResult{4,{4},{0}},positions));
}
