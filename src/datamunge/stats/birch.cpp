/// @file birch.cpp
/// @brief Path-copying cluster-feature insertion and weighted Ward clustering.

#include <datamunge/stats/birch.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace datamunge::stats::birch_detail {

/// @brief Count, mean and RMS radius in working precision.
struct Feature {
    std::size_t count{}; ///< Number of observations summarized.
    std::vector<long double> center; ///< Feature centroids.
    long double radius{}; ///< RMS radius about the centroid.
};

/// @brief Shared immutable node used to commit a whole batch at once.
using NodePtr = std::shared_ptr<const Node>;

/// @brief A leaf feature or the cached aggregate of an internal child.
struct Entry {
    Feature feature; ///< Sufficient statistics of the represented observations.
    NodePtr child; ///< Child subtree, absent in a leaf node.
};

/// @brief Balanced CF-tree node with its cached aggregate.
struct Node {
    std::vector<Entry> entries; ///< Features in fixed traversal order.
    Feature aggregate; ///< Statistics over all descendants.
    std::size_t height{1}; ///< Level count of the subtree.
};

/// @brief Euclidean centroid distance, accumulated without squaring the whole coordinate range.
template<class Left,class Right>
long double distance(const Left& a, const Right& b) {
    long double result = 0;
    for (std::size_t j = 0; j < a.size(); ++j)
        result = std::hypot(result,static_cast<long double>(a[j])-b[j]);
    return result;
}

/// @brief Combines summaries using their counts and the parallel variance identity.
Feature combine(const Feature& a, const Feature& b) {
    if (a.count == 0)
        return b;
    const auto count = a.count + b.count;
    const auto wa = static_cast<long double>(a.count)/count;
    const auto wb = static_cast<long double>(b.count)/count;
    Feature result{count,std::vector<long double>(a.center.size()),
        std::hypot(std::sqrt(wa)*a.radius,std::sqrt(wb)*b.radius,
            std::sqrt(wa*wb)*distance(a.center,b.center))};
    for (std::size_t j = 0; j < result.center.size(); ++j)
        result.center[j] = std::lerp(a.center[j],b.center[j],wb);
    return result;
}

/// @brief Builds a persistent node from an ordered set of features or children.
NodePtr make_node(std::vector<Entry> entries, std::size_t height) {
    Feature aggregate;
    for (const auto& entry : entries)
        aggregate = combine(aggregate,entry.feature);
    return std::make_shared<const Node>(Node{std::move(entries),std::move(aggregate),height});
}

/// @brief Replacement node and optional sibling produced by one insertion.
struct Insertion {
    NodePtr node; ///< Updated node for the existing parent slot.
    NodePtr sibling; ///< Additional node produced by a capacity split.
};

/// @brief Splits around the farthest feature pair and keeps both seeds in their own group.
Insertion split(std::vector<Entry> entries, std::size_t height) {
    std::size_t a = 0, b = 1;
    long double largest = -1;
    for (std::size_t i = 0; i < entries.size(); ++i)
        for (std::size_t j = i+1; j < entries.size(); ++j) {
            const auto current = distance(entries[i].feature.center,entries[j].feature.center);
            if (current > largest) {
                a = i;
                b = j;
                largest = current;
            }
        }
    const auto first = entries[a].feature.center;
    const auto second = entries[b].feature.center;
    std::vector<Entry> left,right;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& center = entries[i].feature.center;
        if (i == a || (i != b && distance(center,first) <= distance(center,second)))
            left.push_back(std::move(entries[i]));
        else
            right.push_back(std::move(entries[i]));
    }
    return {make_node(std::move(left),height),make_node(std::move(right),height)};
}

/// @brief Descends by nearest centroid and path-copies the affected nodes.
Insertion insert(const Node& node, const Feature& point, const BirchOptions& options) {
    std::size_t nearest = 0;
    auto best = distance(node.entries.front().feature.center,point.center);
    for (std::size_t j = 1; j < node.entries.size(); ++j) {
        const auto current = distance(node.entries[j].feature.center,point.center);
        if (current < best) {
            best = current;
            nearest = j;
        }
    }
    auto entries = node.entries;
    auto& entry = entries[nearest];
    if (node.height == 1) {
        auto merged = combine(entry.feature,point);
        if (merged.radius <= options.threshold)
            entry.feature = std::move(merged);
        else
            entries.push_back({point,{}});
    } else {
        auto [child,sibling] = insert(*entry.child,point,options);
        entry = {child->aggregate,child};
        if (sibling)
            entries.push_back({sibling->aggregate,sibling});
    }
    if (entries.size() > options.branching_factor)
        return split(std::move(entries),node.height);
    return {std::make_shared<const Node>(
        Node{std::move(entries),combine(node.aggregate,point),node.height}),{}};
}

/// @brief Traverses a snapshot and collects leaf features with their working precision.
std::vector<Feature> collect_leaves(const NodePtr& root) {
    std::vector<Feature> result;
    if (!root)
        return result;
    std::vector<const Node*> pending{root.get()};
    while (!pending.empty()) {
        const auto* node = pending.back();
        pending.pop_back();
        if (node->height == 1)
            for (const auto& entry : node->entries)
                result.push_back(entry.feature);
        else
            for (auto it = node->entries.rbegin(); it != node->entries.rend(); ++it)
                pending.push_back(it->child.get());
    }
    return result;
}

/// @brief Copies a working feature into the public representation.
BirchFeature report(const Feature& feature) {
    return {feature.count,std::vector<double>(feature.center.begin(),feature.center.end()),feature.radius};
}

/// @brief A candidate Ward merge, ordered by cost and then by component identifiers.
struct WardPair {
    long double cost{}; ///< Square root of the increase in residual sum of squares.
    std::size_t i{}; ///< First cluster identifier.
    std::size_t j{}; ///< Second cluster identifier.
};

/// @brief Comparison for a minimum-cost Ward priority queue.
struct LaterMerge {
    /// @brief Orders larger costs and later identifiers after smaller ones.
    bool operator()(const WardPair& left, const WardPair& right) const {
        return std::tie(left.cost,left.i,left.j) > std::tie(right.cost,right.i,right.j);
    }
};

/// @brief Runs count-weighted Ward merges and restores the labels of all input leaves.
BirchPartition ward_partition(std::vector<Feature> features, std::size_t clusters) {
    const auto n = features.size();
    if (clusters == 0 || clusters > n)
        throw std::invalid_argument("BIRCH: cluster count is outside the leaf count");
    BirchPartition result;
    for (const auto& feature : features)
        result.leaves.push_back(report(feature));
    if (clusters == n) {
        result.clusters = result.leaves;
        result.leaf_labels.resize(n);
        std::iota(result.leaf_labels.begin(),result.leaf_labels.end(),0);
        return result;
    }
    if (n > std::vector<Feature>{}.max_size()/2 ||
        n > std::vector<WardPair>{}.max_size()/n)
        throw std::length_error("BIRCH: Ward merge queue exceeds storage capacity");

    std::priority_queue<WardPair,std::vector<WardPair>,LaterMerge> queue;
    const auto put = [&](std::size_t i, std::size_t j) {
        const auto& a = features[i];
        const auto& b = features[j];
        const auto weight = static_cast<long double>(a.count)/(a.count+b.count)*b.count;
        queue.push({std::sqrt(weight)*distance(a.center,b.center),i,j});
    };
    for (std::size_t j = 1; j < n; ++j)
        for (std::size_t i = 0; i < j; ++i)
            put(i,j);
    const auto terminal_size = 2*n-clusters;
    std::vector<std::size_t> parent(terminal_size,terminal_size);
    while (features.size() < terminal_size) {
        const auto [cost,i,j] = queue.top();
        queue.pop();
        if (parent[i] != terminal_size || parent[j] != terminal_size)
            continue;
        const auto id = features.size();
        parent[i] = parent[j] = id;
        features.push_back(combine(features[i],features[j]));
        if (features.size() < terminal_size)
            for (std::size_t k = 0; k < id; ++k)
                if (parent[k] == terminal_size)
                    put(k,id);
    }
    // Parent identifiers are larger than those of their children.
    std::vector<std::size_t> root(terminal_size), labels(terminal_size,terminal_size);
    for (std::size_t i = terminal_size; i-- > 0;)
        root[i] = parent[i] == terminal_size ? i : root[parent[i]];
    for (std::size_t i = 0; i < n; ++i) {
        const auto r = root[i];
        if (labels[r] == terminal_size) {
            labels[r] = result.clusters.size();
            result.clusters.push_back(report(features[r]));
        }
        result.leaf_labels.push_back(labels[r]);
    }
    return result;
}

} // namespace datamunge::stats::birch_detail

namespace datamunge::stats {

Birch::Birch(std::size_t dimensions, const BirchOptions& options)
    : dimensions_(dimensions), options_(options) {
    if (dimensions == 0 || !(options.threshold >= 0) || !std::isfinite(options.threshold) ||
        options.branching_factor < 2)
        throw std::invalid_argument("BIRCH: invalid feature width, radius threshold or branching factor");
}

void Birch::partial_fit(const linalg::DenseMatrix<double>& observations) {
    if (observations.cols() != dimensions_ ||
        !std::ranges::all_of(observations.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("BIRCH: invalid input data shape or values");
    if (observations.rows() > std::numeric_limits<std::size_t>::max()-this->observations())
        throw std::length_error("BIRCH: total observation count overflows");
    auto next = root_;
    for (std::size_t i = 0; i < observations.rows(); ++i) {
        const auto row = observations.row(i);
        const birch_detail::Feature point{1,std::vector<long double>(row.begin(),row.end()),0};
        if (!next) {
            next = birch_detail::make_node({{point,{}}},1);
            continue;
        }
        auto update = birch_detail::insert(*next,point,options_);
        next = update.sibling
            ? birch_detail::make_node({{update.node->aggregate,update.node},
                {update.sibling->aggregate,update.sibling}},next->height+1)
            : std::move(update.node);
    }
    root_ = std::move(next);
}

std::size_t Birch::observations() const noexcept { return root_ ? root_->aggregate.count : 0; }

std::size_t Birch::height() const noexcept { return root_ ? root_->height : 0; }

std::vector<BirchFeature> Birch::leaf_summaries() const {
    const auto leaves = birch_detail::collect_leaves(root_);
    std::vector<BirchFeature> result;
    result.reserve(leaves.size());
    std::ranges::transform(leaves,std::back_inserter(result),birch_detail::report);
    return result;
}

BirchPartition Birch::cluster_leaves(std::size_t clusters) const {
    return birch_detail::ward_partition(birch_detail::collect_leaves(root_),clusters);
}

std::vector<std::size_t> BirchPartition::predict(const linalg::DenseMatrix<double>& observations) const {
    const auto d = observations.cols();
    if (d == 0 || leaves.empty() || leaves.size() != leaf_labels.size() ||
        !std::ranges::all_of(observations.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("BIRCH: invalid input geometry or partition");
    for (std::size_t i = 0; i < leaves.size(); ++i)
        if (leaf_labels[i] >= clusters.size() || leaves[i].centroid.size() != d ||
            !std::ranges::all_of(leaves[i].centroid,[](double x) { return std::isfinite(x); }))
            throw std::invalid_argument("BIRCH: invalid partition centroid or label");

    std::vector<std::size_t> result(observations.rows());
    for (std::size_t i = 0; i < observations.rows(); ++i) {
        const auto row = observations.row(i);
        std::size_t nearest = 0;
        auto best = birch_detail::distance(row,leaves.front().centroid);
        for (std::size_t j = 1; j < leaves.size(); ++j) {
            const auto current = birch_detail::distance(row,leaves[j].centroid);
            if (current < best) {
                best = current;
                nearest = j;
            }
        }
        result[i] = leaf_labels[nearest];
    }
    return result;
}

} // namespace datamunge::stats
