/// @file blossom.cpp
/// @brief Shared checked Edmonds search, with optional recording of true contractions and lifted paths.
#include <datamunge/algorithms/blossom.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private matching conversion and canonical optional-recording search kernel.
namespace {
/// @brief Convert a complete internal mate array into an owned public matching.
/// @param match Internal partners, with -1 for unmatched vertices.
/// @param size Number of matched pairs, maintained by augmentation.
/// @return Symmetric size_t partner map with the public unmatched sentinel.
GeneralMatchingResult snapshot(const std::vector<int>& match,std::size_t size) {
  GeneralMatchingResult result;result.size=size;
  result.mate.resize(match.size(),kGeneralMatchingUnmatched);
  for (std::size_t v=0;v<match.size();++v)
    if (match[v]!=-1) result.mate[v]=static_cast<std::size_t>(match[v]);
  return result;
}

/// @brief Execute the canonical unweighted blossom algorithm, optionally collecting original-ID events.
/// @tparam Record Whether to allocate contraction/path/matching snapshots.
/// @param vertex_count Vertex count fitting int.
/// @param edges Raw undirected edges; loops ignored after validation and duplicates retained.
/// @return Final matching and, for Record=true, actual events and logical counters.
/// @throws std::length_error If vertex_count exceeds INT_MAX.
/// @throws std::invalid_argument If an endpoint is out of range.
template<bool Record>
GeneralMatchingTrace solve(std::size_t vertex_count,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges) {
  if (vertex_count>static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("blossom_maximum_matching: vertex count exceeds INT_MAX");
  for (const auto& [u,v]:edges)
    if (u>=vertex_count || v>=vertex_count)
      throw std::invalid_argument("blossom_maximum_matching: edge endpoint out of range");
  const int n=static_cast<int>(vertex_count);
  std::vector<std::vector<int>> graph(vertex_count);
  for (const auto& [u,v]:edges) {
    if (u==v) continue;
    graph[u].push_back(static_cast<int>(v));graph[v].push_back(static_cast<int>(u));
  }
  std::vector<int> match(n,-1),parent(n),base(n);
  std::vector<bool> used(n),blossom(n);
  GeneralMatchingTrace trace;std::size_t matching_size=0;

  /// @brief Find the least common outer ancestor in the current alternating tree.
  /// @param a First even-level original vertex.
  /// @param b Second even-level original vertex in the same root tree.
  /// @return Representative of their common blossom base.
  auto lca=[&](int a,int b) {
    std::vector<bool> seen(n,false);
    while (true) {
      a=base[a];seen[a]=true;
      if (match[a]==-1) break;
      a=parent[match[a]];
    }
    while (true) {
      b=base[b];if (seen[b]) return b;
      b=parent[match[b]];
    }
  };
  /// @brief Mark one alternating branch and redirect parent links for later original-vertex lifting.
  /// @param v Current outer endpoint.
  /// @param blossom_base Common representative where the branch ends.
  /// @param child Opposite endpoint of the edge closing the cycle.
  auto mark_path=[&](int v,int blossom_base,int child) {
    while (base[v]!=blossom_base) {
      blossom[base[v]]=blossom[base[match[v]]]=true;
      parent[v]=child;child=match[v];v=parent[match[v]];
    }
  };
  /// @brief BFS from one free root, contracting odd cycles and returning a free endpoint.
  /// @param root Unmatched original vertex; begins a fresh alternating tree.
  /// @return Free endpoint with liftable parent links, or -1 after exhausting this search.
  auto find_path=[&](int root) {
    std::fill(used.begin(),used.end(),false);std::fill(parent.begin(),parent.end(),-1);
    std::iota(base.begin(),base.end(),0);
    std::queue<int> queue;queue.push(root);used[root]=true;
    while (!queue.empty()) {
      const int v=queue.front();queue.pop();
      for (int u:graph[v]) {
        if constexpr (Record) ++trace.edge_scans;
        if (base[v]==base[u] || match[v]==u) continue;
        if (u==root || (match[u]!=-1 && parent[match[u]]!=-1)) {
          const int current_base=lca(v,u);
          std::fill(blossom.begin(),blossom.end(),false);
          mark_path(v,current_base,u);mark_path(u,current_base,v);
          if constexpr (Record) {
            BlossomContraction event{trace.searches-1,static_cast<std::size_t>(root),
                static_cast<std::size_t>(current_base),static_cast<std::size_t>(v),static_cast<std::size_t>(u),{}};
            for (int i=0;i<n;++i)
              if (base[i]==current_base || blossom[base[i]]) event.members.push_back(static_cast<std::size_t>(i));
            trace.contractions.push_back(std::move(event));
          }
          for (int i=0;i<n;++i) if (blossom[base[i]]) {
            base[i]=current_base;
            if (!used[i]) {used[i]=true;queue.push(i);}
          }
        } else if (parent[u]==-1) {
          parent[u]=v;if (match[u]==-1) return u;
          u=match[u];used[u]=true;queue.push(u);
        }
      }
    }
    return -1;
  };
  for (int root=0;root<n;++root) {
    if (match[root]!=-1) continue;
    if constexpr (Record) ++trace.searches;
    int v=find_path(root);
    if (v==-1) continue;
    MatchingAugmentation event;
    if constexpr (Record) event.search_index=trace.searches-1;
    while (v!=-1) {
      const int previous=parent[v],next=previous==-1?-1:match[previous];
      if constexpr (Record) {
        event.path.push_back(static_cast<std::size_t>(v));
        if (previous!=-1) event.path.push_back(static_cast<std::size_t>(previous));
      }
      match[v]=previous;if (previous!=-1) match[previous]=v;v=next;
    }
    ++matching_size;
    if constexpr (Record) {
      std::reverse(event.path.begin(),event.path.end());event.matching=snapshot(match,matching_size);
      trace.augmentations.push_back(std::move(event));
    }
  }
  trace.result=snapshot(match,matching_size);return trace;
}
} // namespace

GeneralMatchingResult blossom_maximum_matching(std::size_t vertex_count,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges) {
  return solve<false>(vertex_count,edges).result;
}
GeneralMatchingTrace blossom_maximum_matching_trace(std::size_t vertex_count,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges) {
  return solve<true>(vertex_count,edges);
}
} // namespace datamunge::algorithms
