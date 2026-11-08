/// @file euclidean_mst.cpp
/// @brief Checked complete planar pair generation feeding the canonical stable Kruskal kernel.
#include <datamunge/algorithms/euclidean_mst.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Complete-pair input validation, arithmetic and graph preparation.
namespace euclidean_mst_detail {
/// @brief Add logical record counts without wrapping.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked total.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a,std::size_t b){
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("Euclidean MST capture geometry overflow");
    return a+b;
}
/// @brief Validate all coordinates even when no pair will be generated.
/// @param points Borrowed input vertices.
/// @throws std::invalid_argument On any nonfinite coordinate.
void validate(const std::vector<Point2D>& points){
    for(const auto& p:points)if(!std::isfinite(p.x)||!std::isfinite(p.y))throw std::invalid_argument("Euclidean MST requires finite coordinates");
}
/// @brief Build every actual hypot-weighted pair in original lexicographic order.
/// @param points Validated finite vertices.
/// @param pairs Checked complete edge count.
/// @return Owned complete graph.
/// @throws std::overflow_error On a nonfinite difference/distance.
/// @throws std::bad_alloc If required graph storage is unavailable.
std::vector<WeightedEdge> graph(const std::vector<Point2D>& points,std::size_t pairs){
    std::vector<WeightedEdge> edges;edges.reserve(pairs);
    for(std::size_t i=0;i<points.size();++i)for(std::size_t j=i+1;j<points.size();++j){
        const double dx=points[i].x-points[j].x,dy=points[i].y-points[j].y;
        if(!std::isfinite(dx)||!std::isfinite(dy))throw std::overflow_error("Euclidean MST coordinate difference overflow");
        const double distance=std::hypot(dx,dy);
        if(!std::isfinite(distance))throw std::overflow_error("Euclidean MST distance overflow");
        edges.emplace_back(i,j,distance);
    }
    return edges;
}
} // namespace euclidean_mst_detail
std::size_t euclidean_mst_pair_count(std::size_t point_count){
    if(point_count>std::vector<Point2D>{}.max_size()||point_count>std::vector<std::size_t>{}.max_size())
        throw std::length_error("Euclidean MST vertex container geometry exceeded");
    if(point_count<2)return 0;
    std::size_t a=point_count,b=point_count-1;if(a%2==0)a/=2;else b/=2;
    if(a&&b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("Euclidean MST pair count overflow");
    const auto pairs=a*b;
    if(pairs>std::vector<WeightedEdge>{}.max_size()||pairs>std::vector<std::size_t>{}.max_size())
        throw std::length_error("Euclidean MST edge container geometry exceeded");
    return pairs;
}
EuclideanMstOutcome euclidean_minimum_spanning_tree_work(const std::vector<Point2D>& points){
    const auto pairs=euclidean_mst_pair_count(points.size());euclidean_mst_detail::validate(points);
    const auto edges=euclidean_mst_detail::graph(points,pairs);auto tree=kruskal_work(points.size(),edges);
    return {std::move(tree.result),{pairs,tree.work}};
}
MinimumSpanningTree euclidean_minimum_spanning_tree(const std::vector<Point2D>& points){return euclidean_minimum_spanning_tree_work(points).result;}
EuclideanMstTrace euclidean_minimum_spanning_tree_trace(const std::vector<Point2D>& points,std::size_t max_entries){
    const auto pairs=euclidean_mst_pair_count(points.size());euclidean_mst_detail::validate(points);
    const auto base=euclidean_mst_detail::add(points.size(),pairs);
    if(max_entries&&base>max_entries)throw std::length_error("Euclidean MST initial capture limit exceeded");
    const auto edges=euclidean_mst_detail::graph(points,pairs);EuclideanMstTrace trace;
    trace.points=points;trace.distance_evaluations=pairs;
    const auto nested_cap=max_entries?max_entries-points.size():0;
    trace.kruskal=kruskal_trace(points.size(),edges,nested_cap);
    trace.captured_entries=euclidean_mst_detail::add(points.size(),trace.kruskal.captured_entries);return trace;
}
} // namespace datamunge::algorithms
