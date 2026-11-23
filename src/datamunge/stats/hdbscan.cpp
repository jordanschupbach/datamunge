/// @file hdbscan.cpp
/// @brief Exact mutual-reachability tree, minimum-size condensation and stability selection.

#include <datamunge/stats/hdbscan.hpp>

#include <datamunge/dstruct/associative.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace datamunge::stats {
/// @brief Euclidean distances, hierarchy construction and flat extraction.
namespace {

/// @brief Internal node of the single-linkage hierarchy.
struct Merge {
    std::vector<std::size_t> children; ///< Child nodes, empty for point leaves.
    std::size_t size{1}; ///< Number of point leaves.
    double distance{}; ///< Mutual-reachability distance at which the children merge.
};

/// @brief Computes a symmetric Euclidean distance using extended precision.
double distance(const linalg::DenseMatrix<double>& points, std::size_t i, std::size_t j) {
    long double norm = 0;
    for (std::size_t d = 0; d < points.cols(); ++d)
        norm = std::hypot(norm,static_cast<long double>(points(i,d))-points(j,d));
    if (norm > std::numeric_limits<double>::max())
        throw std::overflow_error("HDBSCAN: Euclidean distance exceeds finite double range");
    return static_cast<double>(norm);
}

/// @brief Selects each point's core-distance order statistic with self-distance included.
std::vector<double> core_distances(const linalg::DenseMatrix<double>& points, std::size_t rank) {
    const auto n = points.rows();
    std::vector<double> result(n), row(n);
    if (rank == 1)
        return result;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            row[j] = distance(points,i,j);
        std::nth_element(row.begin(),row.begin()+static_cast<std::ptrdiff_t>(rank-1),row.end());
        result[i] = row[rank-1];
    }
    return result;
}

/// @brief Builds the exact dense-graph Prim tree with linear auxiliary storage.
std::vector<HdbscanMstEdge> build_mst(const linalg::DenseMatrix<double>& points,
    const std::vector<double>& core) {
    const auto n = points.rows();
    std::vector<bool> used(n,false);
    std::vector<double> best(n,std::numeric_limits<double>::infinity());
    std::vector<std::size_t> from(n,n);
    std::vector<HdbscanMstEdge> edges;
    edges.reserve(n-1);
    best.front() = 0;
    for (std::size_t count = 0; count < n; ++count) {
        std::size_t v = n;
        for (std::size_t i = 0; i < n; ++i)
            if (!used[i] && (v == n || best[i] < best[v]))
                v = i;
        used[v] = true;
        if (from[v] != n) {
            const auto [first,second] = std::minmax(v,from[v]);
            edges.push_back({first,second,best[v]});
        }
        for (std::size_t i = 0; i < n; ++i)
            if (!used[i]) {
                const auto candidate = std::max({core[v],core[i],distance(points,v,i)});
                if (candidate < best[i]) {
                    best[i] = candidate;
                    from[i] = v;
                }
            }
    }
    std::ranges::sort(edges,[](const HdbscanMstEdge& a, const HdbscanMstEdge& b) {
        return std::tie(a.distance,a.first,a.second) < std::tie(b.distance,b.first,b.second);
    });
    return edges;
}

/// @brief Converts sorted MST edges to a hierarchy with one merge level per distinct distance.
std::vector<Merge> build_hierarchy(std::size_t n, const std::vector<HdbscanMstEdge>& edges) {
    dstruct::DisjointSet sets(n);
    std::vector<Merge> hierarchy(n);
    std::vector<std::size_t> active(n);
    std::iota(active.begin(),active.end(),0);
    for (std::size_t first = 0; first < edges.size();) {
        std::size_t last = first;
        const auto level = edges[first].distance;
        std::vector<std::size_t> affected;
        for (; last < edges.size() && edges[last].distance == level; ++last) {
            affected.push_back(sets.find(edges[last].first));
            affected.push_back(sets.find(edges[last].second));
        }
        std::ranges::sort(affected);
        affected.erase(std::unique(affected.begin(),affected.end()),affected.end());
        for (std::size_t i = first; i < last; ++i)
            sets.unite(edges[i].first,edges[i].second);

        std::map<std::size_t,Merge> merged;
        for (const auto old : affected) {
            auto [where,inserted] = merged.try_emplace(sets.find(old),Merge{{},0,level});
            auto& node = where->second;
            const auto child = active[old];
            node.children.push_back(child);
            node.size += hierarchy[child].size;
        }
        for (auto& [root,node] : merged) {
            active[root] = hierarchy.size();
            hierarchy.push_back(std::move(node));
        }
        first = last;
    }
    return hierarchy;
}

/// @brief Adds an outgoing condensed edge and counts the persistence carried by its members.
void add_edge(HdbscanFit& fit, std::size_t parent, std::size_t child,
    long double lambda, std::size_t size) {
    fit.condensed_tree.push_back({parent,child,lambda,size});
    auto& cluster = fit.clusters[parent-fit.labels.size()];
    if (lambda > cluster.birth_lambda)
        cluster.stability += size*(lambda-cluster.birth_lambda);
}

/// @brief Emits the point leaves of a pruned branch at the parent's split density.
void prune(const std::vector<Merge>& hierarchy, std::size_t node, HdbscanFit& fit,
    std::size_t parent, long double lambda) {
    std::vector<std::size_t> pending{node};
    while (!pending.empty()) {
        const auto next = pending.back();
        pending.pop_back();
        const auto& children = hierarchy[next].children;
        if (children.empty())
            add_edge(fit,parent,next,lambda,1);
        else
            pending.insert(pending.end(),children.rbegin(),children.rend());
    }
}

/// @brief Condenses the hierarchy, continuing cluster identities through one large child.
void condense(const std::vector<Merge>& hierarchy, std::size_t min_size, HdbscanFit& fit) {
    const auto n = fit.labels.size();
    fit.clusters.push_back({n,n,0,0,false});
    std::vector<std::pair<std::size_t,std::size_t>> pending{{hierarchy.size()-1,n}};
    while (!pending.empty()) {
        const auto [node,parent] = pending.back();
        pending.pop_back();
        const auto& merge = hierarchy[node];
        const auto lambda = merge.distance > 0 ? 1.0L/merge.distance
            : std::numeric_limits<long double>::infinity();
        const auto retained = std::ranges::count_if(merge.children,[&](std::size_t child) {
            return hierarchy[child].size >= min_size;
        });
        for (const auto child : merge.children) {
            const auto size = hierarchy[child].size;
            if (size < min_size) {
                prune(hierarchy,child,fit,parent,lambda);
            } else if (retained == 1) {
                pending.emplace_back(child,parent);
            } else {
                const auto cluster = n + fit.clusters.size();
                fit.clusters.push_back({cluster,size,lambda,0,false});
                add_edge(fit,parent,cluster,lambda,size);
                pending.emplace_back(child,cluster);
            }
        }
    }
}

/// @brief Maximizes total stability over disjoint branches and assigns each point its selected ancestor.
void select_clusters(HdbscanFit& fit, bool allow_root) {
    const auto n = fit.labels.size(), count = fit.clusters.size();
    std::vector<std::size_t> parent(count,0);
    for (const auto& edge : fit.condensed_tree)
        if (edge.child >= n)
            parent[edge.child-n] = edge.parent-n;
    std::vector<long double> descendant_stability(count,0);
    std::vector<bool> prefer_cluster(count,false);
    for (std::size_t c = count; c-- > 0;) {
        const auto own = fit.clusters[c].stability;
        prefer_cluster[c] = (c != 0 || allow_root) && own >= descendant_stability[c];
        if (c != 0)
            descendant_stability[parent[c]] += prefer_cluster[c] ? own : descendant_stability[c];
    }
    std::vector<std::int64_t> membership(count,-1);
    std::int64_t label = 0;
    for (std::size_t c = 0; c < count; ++c) {
        if (c != 0)
            membership[c] = membership[parent[c]];
        if (membership[c] < 0 && prefer_cluster[c]) {
            fit.clusters[c].selected = true;
            membership[c] = label++;
        }
    }
    for (const auto& edge : fit.condensed_tree)
        if (edge.child < n)
            fit.labels[edge.child] = membership[edge.parent-n];
}

} // namespace

HdbscanFit fit_hdbscan(const linalg::DenseMatrix<double>& observations, const HdbscanOptions& options) {
    const auto n = observations.rows();
    const auto rank = options.min_samples != 0 ? options.min_samples : options.min_cluster_size;
    if (n < 2 || observations.cols() == 0 || options.min_cluster_size < 2 ||
        options.min_cluster_size > n || rank > n ||
        !std::ranges::all_of(observations.data(),[](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("HDBSCAN: invalid dimensions, controls or observations");
    if (n > std::vector<Merge>{}.max_size()/2 ||
        n > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::length_error("HDBSCAN: hierarchy exceeds storage capacity");
    HdbscanFit fit;
    fit.labels.assign(n,-1);
    fit.core_distances = core_distances(observations,rank);
    fit.minimum_spanning_tree = build_mst(observations,fit.core_distances);
    const auto hierarchy = build_hierarchy(n,fit.minimum_spanning_tree);
    condense(hierarchy,options.min_cluster_size,fit);
    select_clusters(fit,options.allow_single_cluster);
    return fit;
}

} // namespace datamunge::stats
