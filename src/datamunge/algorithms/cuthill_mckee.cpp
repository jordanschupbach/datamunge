/// @file
/// @brief Canonical normalized CM queue kernel and checked symmetric-pattern utilities.
#include <datamunge/algorithms/cuthill_mckee.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Internal graph validation, normalization and queue operations.
namespace cuthill_detail {
/// @brief Add size counts without wrapping.
/// @param a Existing count.
/// @param b Nonnegative increment.
/// @return Exact representable sum.
/// @throws std::length_error size_t overflow.
std::size_t add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a) throw std::length_error("CM: size overflow");
  return a+b;
}
/// @brief Reject vertex counts that cannot be represented by public int ids.
/// @param n Count of vertices.
/// @throws std::length_error Count exceeds INT_MAX.
void vertex_count(std::size_t n) {
  if(n>static_cast<std::size_t>(INT_MAX)) throw std::length_error("CM: too many vertices");
}
/// @brief Check every raw endpoint and sum every raw row before any owning graph copy.
/// @param a Raw adjacency list.
/// @return Total raw directed entries including loops and duplicates.
/// @throws std::invalid_argument Invalid id.
/// @throws std::length_error Vertex or count representation exceeded.
std::size_t validate_raw(const std::vector<std::vector<int>>& a) {
  vertex_count(a.size());std::size_t count=0;
  for(const auto& row:a) {
    count=add(count,row.size());
    for(int v:row) if(v<0 || static_cast<std::size_t>(v)>=a.size()) throw std::invalid_argument("CM: invalid endpoint");
  }
  return count;
}
/// @brief Copy raw checked rows, remove loops/duplicates, and require reverse edges.
/// @param a Endpoint-checked raw graph.
/// @return Owned sorted simple symmetric graph.
/// @throws std::invalid_argument Missing reverse edge after normalization.
/// @throws std::bad_alloc Graph allocation failure.
std::vector<std::vector<int>> normalize(const std::vector<std::vector<int>>& a) {
  auto g=a;
  for(std::size_t i=0;i<g.size();++i) {
    auto& row=g[i];row.erase(std::remove(row.begin(),row.end(),static_cast<int>(i)),row.end());
    std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());
  }
  for(std::size_t i=0;i<g.size();++i) for(int v:g[i])
    if(!std::binary_search(g[v].begin(),g[v].end(),static_cast<int>(i))) throw std::invalid_argument("CM: asymmetric pattern");
  return g;
}
/// @brief Run one degree/id-prioritized BFS forest, optionally capturing actual work.
/// @param g Normalized simple symmetric graph.
/// @param reverse Reverse final order when true.
/// @param trace Optional destination for complete actual history; null avoids history allocation.
/// @return Owned final order.
/// @throws std::length_error Work count overflow.
/// @throws std::bad_alloc Traversal/history allocation failure.
std::vector<int> traverse(const std::vector<std::vector<int>>& g,bool reverse,CuthillMcKeeTrace* trace) {
  const std::size_t n=g.size();std::vector<int> degree(n),seeds(n),queue,order;
  for(std::size_t i=0;i<n;++i) degree[i]=static_cast<int>(g[i].size());
  std::iota(seeds.begin(),seeds.end(),0);
  /// @brief Compare unique vertices by fixed structural degree then original id.
  const auto less=[&degree](int a,int b) { return degree[a]!=degree[b]?degree[a]<degree[b]:a<b; };
  std::sort(seeds.begin(),seeds.end(),less);
  std::vector<unsigned char> seen(n,0);order.reserve(n);queue.reserve(n);
  if(trace) {trace->degree=degree;trace->parent.resize(n);trace->level.resize(n);trace->steps.reserve(n);}
  for(int root:seeds) {
    if(seen[root]) continue;
    queue.clear();queue.push_back(root);seen[root]=1;
    if(trace) {trace->roots.push_back(root);trace->parent[root]=root;trace->level[root]=0;}
    for(std::size_t head=0;head<queue.size();++head) {
      const int u=queue[head];order.push_back(u);std::vector<int> fresh;
      for(int v:g[u]) if(!seen[v]) fresh.push_back(v);
      std::sort(fresh.begin(),fresh.end(),less);
      for(int v:fresh) {
        seen[v]=1;queue.push_back(v);
        if(trace) {trace->parent[v]=u;trace->level[v]=trace->level[u]+1;}
      }
      if(trace) {
        trace->adjacency_scans=add(trace->adjacency_scans,g[u].size());
        trace->neighbor_enqueues=add(trace->neighbor_enqueues,fresh.size());
        trace->steps.push_back({u,std::move(fresh)});
      }
    }
  }
  if(trace) trace->forward_order=order;
  if(reverse) std::reverse(order.begin(),order.end());
  return order;
}
/// @brief Validate the graph/order size and invert the complete permutation.
/// @param n Graph vertex count.
/// @param order New-to-original permutation.
/// @return Owned original-to-new positions.
/// @throws std::invalid_argument Mismatched size or invalid permutation.
/// @throws std::length_error Too many vertices.
/// @throws std::bad_alloc Inverse allocation failure.
std::vector<int> positions(std::size_t n,const std::vector<int>& order) {
  if(order.size()!=n) throw std::invalid_argument("CM: order size mismatch");
  return inverse_ordering(order);
}
} // namespace cuthill_detail

std::vector<int> cuthill_mckee(const std::vector<std::vector<int>>& a,bool reverse) {
  cuthill_detail::validate_raw(a);return cuthill_detail::traverse(cuthill_detail::normalize(a),reverse,nullptr);
}
CuthillMcKeeTrace cuthill_mckee_trace(const std::vector<std::vector<int>>& a,bool reverse,std::size_t max_vertices,std::size_t max_adjacency_entries) {
  const auto count=cuthill_detail::validate_raw(a);
  if((max_vertices && a.size()>max_vertices)||(max_adjacency_entries && count>max_adjacency_entries)) throw std::length_error("CM: complete trace cap exceeded");
  CuthillMcKeeTrace t;t.order=cuthill_detail::traverse(cuthill_detail::normalize(a),reverse,&t);
  t.position=inverse_ordering(t.order);return t;
}
std::vector<int> inverse_ordering(const std::vector<int>& order) {
  cuthill_detail::vertex_count(order.size());std::vector<int> position(order.size(),-1);
  for(std::size_t k=0;k<order.size();++k) {
    const int v=order[k];
    if(v<0 || static_cast<std::size_t>(v)>=order.size() || position[v]!=-1) throw std::invalid_argument("CM: invalid permutation");
    position[v]=static_cast<int>(k);
  }
  return position;
}
std::vector<std::vector<int>> permute_symmetric_pattern(const std::vector<std::vector<int>>& a,const std::vector<int>& order) {
  cuthill_detail::validate_raw(a);const auto pos=cuthill_detail::positions(a.size(),order);
  const auto g=cuthill_detail::normalize(a);std::vector<std::vector<int>> out(g.size());
  for(std::size_t i=0;i<g.size();++i) {
    auto& row=out[pos[i]];row.reserve(g[i].size());for(int v:g[i]) row.push_back(pos[v]);std::sort(row.begin(),row.end());
  }
  return out;
}
SymmetricPatternMetrics symmetric_pattern_metrics(const std::vector<std::vector<int>>& a,const std::vector<int>& order) {
  cuthill_detail::validate_raw(a);const auto pos=cuthill_detail::positions(a.size(),order);const auto g=cuthill_detail::normalize(a);
  SymmetricPatternMetrics m;m.lower_width.resize(g.size());m.upper_width.resize(g.size());
  for(std::size_t i=0;i<g.size();++i) for(int v:g[i]) {
    const std::size_t row=static_cast<std::size_t>(pos[i]),col=static_cast<std::size_t>(pos[v]);
    if(col<row) {m.lower_width[row]=std::max(m.lower_width[row],row-col);m.edges=cuthill_detail::add(m.edges,1);}
    else m.upper_width[row]=std::max(m.upper_width[row],col-row);
  }
  for(std::size_t i=0;i<g.size();++i) {
    m.bandwidth=std::max({m.bandwidth,m.lower_width[i],m.upper_width[i]});
    m.lower_profile=cuthill_detail::add(m.lower_profile,m.lower_width[i]);m.upper_profile=cuthill_detail::add(m.upper_profile,m.upper_width[i]);
  }
  return m;
}
SymmetricPatternCoordinates symmetric_pattern_coordinates(const std::vector<std::vector<int>>& a,const std::vector<int>& order,bool include_diagonal) {
  const auto g=permute_symmetric_pattern(a,order);std::size_t count=include_diagonal?g.size():0;
  for(const auto& row:g) count=cuthill_detail::add(count,row.size());
  SymmetricPatternCoordinates p;p.row.reserve(count);p.column.reserve(count);
  for(std::size_t i=0;i<g.size();++i) {
    bool emitted=!include_diagonal;
    for(int v:g[i]) {
      if(!emitted && static_cast<std::size_t>(v)>i) {p.row.push_back(static_cast<double>(i));p.column.push_back(static_cast<double>(i));emitted=true;}
      p.row.push_back(static_cast<double>(i));p.column.push_back(v);
    }
    if(!emitted) {p.row.push_back(static_cast<double>(i));p.column.push_back(static_cast<double>(i));}
  }
  return p;
}
std::vector<std::vector<int>> symmetric_band_pattern(std::size_t n,std::size_t width,std::size_t max_entries) {
  cuthill_detail::vertex_count(n);width=std::min(width,n?n-1:0);
  // Twice the sum (n-1)+...+(n-width), with division-free checked arithmetic.
  const auto factor=n?cuthill_detail::add(n,n-width-1):0;
  if(width && factor>std::numeric_limits<std::size_t>::max()/width) throw std::length_error("CM: pattern count overflow");
  const auto entries=width*factor;
  if(max_entries && entries>max_entries) throw std::length_error("CM: complete pattern cap exceeded");
  std::vector<std::vector<int>> g(n);
  for(std::size_t i=0;i<n;++i) {
    const std::size_t lo=i>width?i-width:0,hi=i+std::min(width,n-1-i);
    g[i].reserve(hi-lo);
    for(std::size_t j=lo;j<i;++j) g[i].push_back(static_cast<int>(j));
    for(std::size_t j=i+1;j<=hi;++j) g[i].push_back(static_cast<int>(j));
  }
  return g;
}
} // namespace datamunge::algorithms
