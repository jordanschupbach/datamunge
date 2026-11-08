/// @file convex_hull.cpp
/// @brief Single exact-sign Andrew kernel, owned actual stack edits and unsnapped original sampling.
#include <datamunge/geometry/convex_hull.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
namespace datamunge::geometry {
/// @brief Checked internal counters and the shared hull execution kernel.
namespace {
/// @brief Increment an actual operation counter without wrapping.
/// @param value Counter to increment.
/// @throws std::length_error Counter exhaustion.
void tick(std::size_t& value) {
    if (value==std::numeric_limits<std::size_t>::max()) throw std::length_error("convex hull: counter overflow");
    ++value;
}
/// @brief Execute sorting/deduplication and two actual exact-sign chain scans.
/// @tparam Record Keep actual scan edits/completed chains when true.
/// @param points Original records, validated completely before indexed copying/sorting.
/// @param options Complete original count and actual event caps.
/// @return Owned actual result/work and optional sorted records/scan history.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Point/event cap or counter/container exhaustion.
template<bool Record>
ConvexHullTrace run(const std::vector<Point2D>& points,ConvexHullOptions options) {
    for (const auto& p:points)
        if (!std::isfinite(p.x)||!std::isfinite(p.y)) throw std::invalid_argument("convex hull: nonfinite coordinate");
    if (options.max_points && points.size()>options.max_points) throw std::length_error("convex hull: complete point cap");
    ConvexHullTrace out;auto& w=out.analysis.work;w.input_points=points.size();
    out.sorted.reserve(points.size());
    for (std::size_t i=0;i<points.size();++i) out.sorted.push_back({points[i],i});
    /// @brief Count actual strict lexicographic comparisons, breaking equal coordinates by identity.
    /// @param a First indexed record.
    /// @param b Second indexed record.
    /// @return Whether a precedes b by (x,y,input index).
    /// @throws std::length_error Comparison counter exhaustion.
    const auto less=[&](const HullIndexedPoint& a,const HullIndexedPoint& b) {
        tick(w.sort_comparisons);
        if (a.point.x!=b.point.x) return a.point.x<b.point.x;
        if (a.point.y!=b.point.y) return a.point.y<b.point.y;
        return a.input_index<b.input_index;
    };
    std::sort(out.sorted.begin(),out.sorted.end(),less);
    /// @brief Count exact adjacent-coordinate equality tests, retaining the first identity.
    /// @param a First sorted record.
    /// @param b Second sorted record.
    /// @return Exact coordinate equality, including signed-zero equality.
    /// @throws std::length_error Equality counter exhaustion.
    const auto equal=[&](const HullIndexedPoint& a,const HullIndexedPoint& b) {
        tick(w.duplicate_comparisons);return a.point==b.point;
    };
    out.sorted.erase(std::unique(out.sorted.begin(),out.sorted.end(),equal),out.sorted.end());
    const auto n=out.sorted.size();w.distinct_points=n;
    /// @brief Retain one actual constant-size stack edit/test when capture is enabled.
    /// @param event Completed actual event.
    /// @throws std::length_error Actual event cap exceeded.
    const auto emit=[&](HullEvent event) {
        if constexpr (Record) {
            if (options.max_events && out.events.size()>=options.max_events) throw std::length_error("convex hull: actual event cap");
            out.events.push_back(event);
        }
    };
    if (n<2) {
        if (n) {out.analysis.hull.push_back(out.sorted[0].point);out.analysis.input_indices.push_back(out.sorted[0].input_index);}
        return out;
    }
    /// @brief Build one actual chain; every candidate is pushed once in its pass.
    /// @param pass Forward lower or reverse upper scan.
    /// @return Actual final stack of sorted unique indices.
    /// @throws std::length_error Event/work/container exhaustion.
    const auto scan=[&](HullPass pass) {
        std::vector<std::size_t> stack;stack.reserve(n);
        for (std::size_t step=0;step<n;++step) {
            const auto candidate=pass==HullPass::lower?step:n-1-step;
            while (stack.size()>=2) {
                const auto a=stack[stack.size()-2],b=stack.back();tick(w.orientation_tests);
                const int sign=orientation_sign(out.sorted[a].point,out.sorted[b].point,out.sorted[candidate].point);
                emit({pass,HullAction::test,candidate,candidate,a,b,sign,stack.size()});
                if (sign>0) break;
                stack.pop_back();tick(w.pops);
                emit({pass,HullAction::pop,candidate,b,0,0,0,stack.size()});
            }
            stack.push_back(candidate);tick(w.pushes);w.max_stack=std::max(w.max_stack,stack.size());
            emit({pass,HullAction::push,candidate,candidate,0,0,0,stack.size()});
        }
        return stack;
    };
    auto lower=scan(HullPass::lower),upper=scan(HullPass::upper);
    for (const auto* chain:{&lower,&upper})
        for (std::size_t k=0;k+1<chain->size();++k) {
            const auto& p=out.sorted[(*chain)[k]];
            out.analysis.hull.push_back(p.point);out.analysis.input_indices.push_back(p.input_index);
        }
    if constexpr (Record) {out.lower=std::move(lower);out.upper=std::move(upper);}
    return out;
}
} // namespace
std::vector<Point2D> convex_hull(std::vector<Point2D> points) {
    return run<false>(points,{0,0}).analysis.hull;
}
ConvexHullAnalysis convex_hull_analyze(const std::vector<Point2D>& points,ConvexHullOptions options) {
    return run<false>(points,options).analysis;
}
ConvexHullTrace convex_hull_trace(const std::vector<Point2D>& points,ConvexHullOptions options) {
    return run<true>(points,options);
}
std::vector<Point2D> sample_gaussian_points(std::size_t count,std::uint32_t seed,std::size_t max_points) {
    if (max_points && count>max_points) throw std::length_error("normal points: complete count cap");
    std::mt19937 rng(seed);std::normal_distribution<double> normal(0,1);
    std::vector<Point2D> points;points.reserve(count);
    for (std::size_t i=0;i<count;++i) {
        const double x=normal(rng),y=normal(rng);
        if (!std::isfinite(x)||!std::isfinite(y)) throw std::overflow_error("normal points: nonfinite draw");
        points.push_back({x,y});
    }
    return points;
}
} // namespace datamunge::geometry
