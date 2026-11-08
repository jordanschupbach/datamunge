/// @file branch_and_bound.cpp
/// @brief Exact density ordering, rational bounds and optional records from iterative knapsack DFS.
#include <datamunge/algorithms/branch_and_bound.hpp>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private exact arithmetic and the shared iterative traversal.
namespace {
static_assert(std::numeric_limits<int>::digits<=31,"knapsack exact products require at most 32-bit int");
/// @brief Positive-profit active item with its original index.
struct Item {
    int value; ///< Strictly positive profit; nonpositive original values are excluded.
    int weight; ///< Validated nonnegative weight, possibly zero.
    std::size_t original; ///< Index in the caller's vectors.
};
/// @brief Add nonnegative exact bound contributions without signed overflow.
/// @param value Current nonnegative sum.
/// @param extra Nonnegative next contribution.
/// @return Exact int64 sum.
/// @throws std::overflow_error If the exact bound exceeds int64.
std::int64_t add_bound(std::int64_t value,std::int64_t extra) {
    if (value>std::numeric_limits<std::int64_t>::max()-extra)
        throw std::overflow_error("knapsack: bound exceeds int64");
    return value+extra;
}
/// @brief Order active items by exact density, with free items first and original-index ties.
/// @param a First positive-profit item.
/// @param b Second positive-profit item.
/// @return Whether a must precede b in the strict total tie order.
bool density_before(const Item& a,const Item& b) {
    if (a.weight==0 || b.weight==0) {
        if ((a.weight==0)!=(b.weight==0)) return a.weight==0;
        return a.original<b.original;
    }
    const auto left=std::int64_t(a.value)*b.weight,right=std::int64_t(b.value)*a.weight;
    return left!=right?left>right:a.original<b.original;
}
/// @brief Compute exact greedy fractional-knapsack components and an integer-completion bound.
/// @param items Active positive-profit items in exact density order.
/// @param level First undecided active item.
/// @param weight Feasible current weight.
/// @param value Representable feasible current value.
/// @param capacity Validated capacity.
/// @return Exact whole/fraction terms, their floor and display-only rounded LP value.
/// @throws std::overflow_error If a bound sum exceeds int64.
KnapsackBound bound(const std::vector<Item>& items,std::size_t level,int weight,int value,int capacity) {
    KnapsackBound out;out.whole_value=value;int room=capacity-weight;
    for (std::size_t i=level;i<items.size();++i) {
        const auto& item=items[i];
        if (item.weight<=room) {
            room-=item.weight;out.whole_value=add_bound(out.whole_value,item.value);
        } else {
            out.fractional_numerator=std::int64_t(item.value)*room;
            out.fractional_denominator=item.weight;out.fractional_item=item.original;break;
        }
    }
    out.integer_upper=add_bound(out.whole_value,out.fractional_numerator/out.fractional_denominator);
    out.relaxation_value=static_cast<double>(out.whole_value)+
        static_cast<double>(out.fractional_numerator)/out.fractional_denominator;
    return out;
}
/// @brief Explicit DFS activation; stage zero enters, one returns from take, two returns from skip.
struct Frame {
    std::size_t level; ///< Number of fixed active decisions.
    int weight; ///< Feasible accumulated weight.
    std::int64_t value; ///< Accumulated value, checked against int before processing the entry.
    std::size_t parent; ///< Recorded parent index, ignored by unrecorded execution.
    std::size_t item_index; ///< Original item decided by the incoming edge, or size_t maximum for root.
    bool took; ///< Incoming decision; ignored at root.
    unsigned stage{0}; ///< Resume point in take-first DFS.
    std::size_t node_id{std::numeric_limits<std::size_t>::max()}; ///< Recorded visit index after entry.
};
/// @brief Solve through one kernel, allocating node records only in the recording specialisation.
/// @tparam Record Whether actual node events and diagnostic counts are retained.
/// @param values Signed original profits.
/// @param weights Original nonnegative weights.
/// @param capacity Nonnegative capacity.
/// @return Optimal result and optional owned records.
/// @throws std::invalid_argument If lengths, weights or capacity are invalid.
/// @throws std::overflow_error If a feasible value, bound or visit count exceeds its range.
template<bool Record>
KnapsackTrace solve(const std::vector<int>& values,const std::vector<int>& weights,int capacity) {
    if (values.size()!=weights.size() || capacity<0)
        throw std::invalid_argument("knapsack: mismatched lengths or negative capacity");
    for (int weight:weights) if (weight<0) throw std::invalid_argument("knapsack: negative weight");
    KnapsackTrace out;out.original_size=values.size();out.result.chosen.assign(values.size(),0);
    if (values.empty()) return out;
    std::vector<Item> items;items.reserve(values.size());
    for (std::size_t i=0;i<values.size();++i) if (values[i]>0) items.push_back({values[i],weights[i],i});
    std::sort(items.begin(),items.end(),density_before);
    if constexpr (Record) for (const auto& item:items) out.item_order.push_back(item.original);
    std::vector<int> take(items.size()),best_take(items.size());int best=0;
    const auto no_index=std::numeric_limits<std::size_t>::max();
    std::vector<Frame> stack;stack.push_back({0,0,0,no_index,no_index,false});
    while (!stack.empty()) {
        auto& frame=stack.back();
        if (frame.stage==0) {
            if (out.result.nodes_explored==std::numeric_limits<long long>::max())
                throw std::overflow_error("knapsack: visit count exceeds long long");
            ++out.result.nodes_explored;
            if (frame.value>std::numeric_limits<int>::max())
                throw std::overflow_error("knapsack: optimum exceeds int");
            const int value=static_cast<int>(frame.value),before=best;
            if (value>best) {best=value;best_take=take;}
            const auto upper=bound(items,frame.level,frame.weight,value,capacity);
            const auto state=frame.level==items.size()?KnapsackNodeState::leaf:
                upper.integer_upper<=best?KnapsackNodeState::pruned:KnapsackNodeState::branch;
            if constexpr (Record) {
                frame.node_id=out.nodes.size();
                out.nodes.push_back({frame.parent,frame.item_index,frame.took,frame.level,frame.weight,value,
                    upper,before,best,state});
                if (state==KnapsackNodeState::pruned) ++out.bound_prunes;
                if (state==KnapsackNodeState::leaf) ++out.leaves;
            }
            if (state!=KnapsackNodeState::branch) {stack.pop_back();continue;}
            frame.stage=1;const auto& item=items[frame.level];
            if (item.weight<=capacity-frame.weight) {
                take[frame.level]=1;
                stack.push_back({frame.level+1,frame.weight+item.weight,frame.value+item.value,
                    frame.node_id,item.original,true});
                continue;
            }
            if constexpr (Record) ++out.infeasible_takes;
        }
        if (frame.stage==1) {
            take[frame.level]=0;frame.stage=2;
            stack.push_back({frame.level+1,frame.weight,frame.value,frame.node_id,items[frame.level].original,false});
            continue;
        }
        stack.pop_back();
    }
    out.result.value=best;
    for (std::size_t k=0;k<items.size();++k) if (best_take[k]) out.result.chosen[items[k].original]=1;
    return out;
}
} // namespace
KnapsackResult knapsack_branch_and_bound(const std::vector<int>& values,const std::vector<int>& weights,int capacity) {
    return solve<false>(values,weights,capacity).result;
}
KnapsackTrace knapsack_branch_and_bound_trace(const std::vector<int>& values,const std::vector<int>& weights,int capacity) {
    return solve<true>(values,weights,capacity);
}
std::vector<KnapsackInstance> knapsack_random_instances(const std::vector<std::size_t>& sizes,
    int capacity_per_item,std::uint32_t seed,int max_value,int max_weight) {
    if (capacity_per_item<0 || max_value<=0 || max_weight<=0)
        throw std::invalid_argument("knapsack_random_instances: invalid ranges or capacity multiplier");
    for (auto n:sizes) if (capacity_per_item>0 && n>static_cast<std::size_t>(std::numeric_limits<int>::max()/capacity_per_item))
        throw std::overflow_error("knapsack_random_instances: capacity exceeds int");
    std::mt19937 rng(seed);std::uniform_int_distribution<int> value(1,max_value),weight(1,max_weight);
    std::vector<KnapsackInstance> out;out.reserve(sizes.size());
    for (auto n:sizes) {
        KnapsackInstance instance;instance.values.resize(n);instance.weights.resize(n);
        instance.capacity=capacity_per_item==0?0:static_cast<int>(n)*capacity_per_item;
        for (auto& v:instance.values) v=value(rng);
        for (auto& w:instance.weights) w=weight(rng);
        out.push_back(std::move(instance));
    }
    return out;
}
} // namespace datamunge::algorithms
