/// @file kruskal.cpp
/// @brief One stable Kruskal execution for compact results, counters and actual owned decisions.
#include <datamunge/algorithms/kruskal.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Checked geometry, actual union-find operations and optional capture.
namespace kruskal_detail {
/// @brief Add logical counters without wrapping.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a,std::size_t b){
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("kruskal count overflow");
    return a+b;
}
/// @brief Validate all graph geometry and every finite edge before sorting/indexing.
/// @param n Vertex count.
/// @param edges Borrowed input records.
/// @throws std::invalid_argument On invalid endpoints or nonfinite weights.
/// @throws std::length_error On container/count limits.
void validate(std::size_t n,const std::vector<WeightedEdge>& edges){
    if(n>std::vector<std::size_t>{}.max_size()||edges.size()>std::vector<std::size_t>{}.max_size())
        throw std::length_error("kruskal container geometry exceeded");
    (void)add(edges.size(),edges.size());
    for(const auto& [u,v,w]:edges)
        if(u>=n||v>=n||!std::isfinite(w))throw std::invalid_argument("kruskal requires valid endpoints and finite weights");
}
/// @brief Actual path-halving disjoint-set state with deterministic rank ties.
class DisjointSet {
public:
    /// @brief Start with one component per vertex.
    /// @param n Validated vertex count.
    /// @throws std::bad_alloc If storage is unavailable.
    explicit DisjointSet(std::size_t n):parent_(n),rank_(n,0){std::iota(parent_.begin(),parent_.end(),std::size_t{0});}
    /// @brief Find and halve an actual parent path.
    /// @param x Valid vertex.
    /// @return Current root.
    std::size_t find(std::size_t x){while(parent_[x]!=x){parent_[x]=parent_[parent_[x]];x=parent_[x];}return x;}
    /// @brief Merge distinct roots using the original rank/tie rule.
    /// @param u Actual first root.
    /// @param v Actual different second root.
    void merge(std::size_t u,std::size_t v){if(rank_[u]<rank_[v])std::swap(u,v);parent_[v]=u;if(rank_[u]==rank_[v])++rank_[u];}
private:
    std::vector<std::size_t> parent_; ///< Owned actual parent links.
    std::vector<std::size_t> rank_; ///< Union ranks, bounded by the number of vertices.
};
/// @brief Reserve retained records before their capture or scan.
/// @param trace Optional actual trace.
/// @param count Additional input/step records.
/// @param cap Optional quota, zero disables.
/// @throws std::length_error On checked count/quota limits.
void capture(KruskalTrace* trace,std::size_t count,std::size_t cap){
    if(!trace)return;const auto total=add(trace->captured_entries,count);
    if(cap&&total>cap)throw std::length_error("kruskal capture limit exceeded");
    trace->captured_entries=total;
}
/// @brief Execute one validated graph with optional live capture and no replay.
/// @param n Validated vertex count.
/// @param edges Validated borrowed finite graph.
/// @param trace Optional trace with input records already reserved/copied.
/// @param cap Optional retained-record quota.
/// @return Owned forest and actual logical counts.
/// @throws std::length_error On capture/container limits.
/// @throws std::overflow_error On nonfinite intermediate selected-edge total.
/// @throws std::bad_alloc If required storage is unavailable.
KruskalOutcome core(std::size_t n,const std::vector<WeightedEdge>& edges,KruskalTrace* trace,std::size_t cap){
    std::vector<std::size_t> order(edges.size());std::iota(order.begin(),order.end(),std::size_t{0});
    /// @brief Compare finite weights, preserving input order for equality via stable_sort.
    /// @param a First original edge index.
    /// @param b Second original edge index.
    /// @return Whether the first weight is strictly smaller.
    const auto lighter=[&edges](std::size_t a,std::size_t b){return std::get<2>(edges[a])<std::get<2>(edges[b]);};
    std::stable_sort(order.begin(),order.end(),lighter);
    DisjointSet dsu(n);std::size_t components=n;KruskalOutcome out;out.work.input_edges=edges.size();
    out.result.edges.reserve(std::min(n?n-1:0,edges.size()));
    for(auto index:order){
        capture(trace,1,cap);
        if(trace&&trace->steps.size()==trace->steps.max_size())throw std::length_error("kruskal step container exceeded");
        const auto& [u,v,w]=edges[index];const auto ru=dsu.find(u),rv=dsu.find(v);
        ++out.work.scanned_edges;out.work.find_calls+=2;const bool accepted=ru!=rv;
        if(accepted){
            const double total=out.result.total_weight+w;
            if(!std::isfinite(total))throw std::overflow_error("kruskal selected-edge total overflow");
            dsu.merge(ru,rv);--components;++out.work.accepted_edges;
            out.result.edges.push_back(edges[index]);out.result.total_weight=total;
        }else ++out.work.cycle_rejections;
        if(trace)trace->steps.push_back({index,ru,rv,accepted,components});
        if(accepted&&n&&out.result.edges.size()==n-1)break;
    }
    out.result.is_connected=n==0||out.result.edges.size()==n-1;return out;
}
} // namespace kruskal_detail
KruskalOutcome kruskal_work(std::size_t n,const std::vector<WeightedEdge>& edges){
    kruskal_detail::validate(n,edges);return kruskal_detail::core(n,edges,nullptr,0);
}
MinimumSpanningTree kruskal(std::size_t n,const std::vector<WeightedEdge>& edges){return kruskal_work(n,edges).result;}
KruskalTrace kruskal_trace(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t max_entries){
    kruskal_detail::validate(n,edges);KruskalTrace trace;kruskal_detail::capture(&trace,edges.size(),max_entries);
    trace.input_edges=edges;trace.outcome=kruskal_detail::core(n,edges,&trace,max_entries);return trace;
}
} // namespace datamunge::algorithms
