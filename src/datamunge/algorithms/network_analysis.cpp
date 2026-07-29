#include <datamunge/algorithms/network_analysis.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {
namespace {
using Edge = std::pair<std::size_t, std::size_t>;
void validate(std::size_t n, const std::vector<Edge>& edges) {
  for (const auto& [u,v] : edges)
    if (u >= n || v >= n) throw std::invalid_argument("network analysis: endpoint out of range");
}
double normalize(std::vector<double>& values) {
  double norm = 0; for (double x : values) norm += x * x; norm = std::sqrt(norm);
  if (norm > 0) for (double& x : values) x /= norm;
  return norm;
}
std::vector<std::vector<std::size_t>> components(
    const std::vector<std::vector<std::size_t>>& adjacency) {
  std::vector<bool> seen(adjacency.size());
  std::vector<std::vector<std::size_t>> result;
  for (std::size_t root = 0; root < adjacency.size(); ++root) if (!seen[root]) {
    result.emplace_back(); std::queue<std::size_t> q; q.push(root); seen[root] = true;
    while (!q.empty()) { auto u=q.front(); q.pop(); result.back().push_back(u);
      for (auto v:adjacency[u]) if(!seen[v]) {seen[v]=true;q.push(v);} }
  }
  return result;
}
} // namespace

std::vector<double> page_rank(std::size_t n, const std::vector<Edge>& edges,
                              double damping, double tolerance, std::size_t max_iterations) {
  validate(n, edges);
  if (!(damping >= 0 && damping < 1) || tolerance <= 0)
    throw std::invalid_argument("page_rank: invalid numerical parameter");
  if (n == 0) return {};
  std::vector<std::vector<std::size_t>> out(n);
  for (auto [u,v]:edges) out[u].push_back(v);
  std::vector<double> rank(n,1.0/n), next(n);
  for (std::size_t iter=0;iter<max_iterations;++iter) {
    double dangling=0; for(std::size_t u=0;u<n;++u) if(out[u].empty()) dangling+=rank[u];
    std::fill(next.begin(),next.end(),(1-damping)/n+damping*dangling/n);
    for(std::size_t u=0;u<n;++u) if(!out[u].empty())
      for(auto v:out[u]) next[v]+=damping*rank[u]/out[u].size();
    double error=0; for(std::size_t i=0;i<n;++i) error+=std::abs(next[i]-rank[i]);
    rank.swap(next); if(error<tolerance) break;
  }
  return rank;
}

std::vector<double> trust_rank(std::size_t n, const std::vector<Edge>& edges,
                               const std::vector<std::size_t>& seeds, double damping,
                               double tolerance, std::size_t max_iterations) {
  validate(n, edges);
  if (seeds.empty() || !(damping >= 0 && damping < 1) || tolerance <= 0)
    throw std::invalid_argument("trust_rank: invalid parameter");
  std::vector<double> teleport(n);
  for (auto seed : seeds) {
    if (seed >= n) throw std::invalid_argument("trust_rank: seed out of range");
    teleport[seed] += 1.0 / seeds.size();
  }
  std::vector<std::vector<std::size_t>> out(n);
  for (auto [u,v] : edges) out[u].push_back(v);
  std::vector<double> rank = teleport, next(n);
  for (std::size_t iter=0; iter<max_iterations; ++iter) {
    double dangling=0; for(std::size_t u=0;u<n;++u) if(out[u].empty()) dangling+=rank[u];
    for(std::size_t v=0;v<n;++v) next[v]=(1-damping)*teleport[v]+damping*dangling*teleport[v];
    for(std::size_t u=0;u<n;++u) if(!out[u].empty())
      for(auto v:out[u]) next[v]+=damping*rank[u]/out[u].size();
    double error=0;for(std::size_t i=0;i<n;++i)error+=std::abs(next[i]-rank[i]);
    rank.swap(next);if(error<tolerance)break;
  }
  return rank;
}

HITSResult hits(std::size_t n, const std::vector<Edge>& edges,
                double tolerance, std::size_t max_iterations) {
  validate(n,edges); if(tolerance<=0) throw std::invalid_argument("hits: tolerance must be positive");
  HITSResult result; result.hubs.assign(n,n?1/std::sqrt(static_cast<double>(n)):0);
  result.authorities=result.hubs;
  for(;result.iterations<max_iterations;++result.iterations) {
    std::vector<double> a(n),h(n);
    for(auto [u,v]:edges) a[v]+=result.hubs[u];
    normalize(a); for(auto [u,v]:edges) h[u]+=a[v]; normalize(h);
    double error=0; for(std::size_t i=0;i<n;++i)
      error=std::max(error,std::max(std::abs(a[i]-result.authorities[i]),std::abs(h[i]-result.hubs[i])));
    result.authorities.swap(a); result.hubs.swap(h);
    if(error<tolerance) {++result.iterations;break;}
  }
  return result;
}

std::vector<std::vector<std::size_t>> girvan_newman(
    std::size_t n, const std::vector<Edge>& edges, std::size_t target) {
  validate(n,edges);
  if(target==0 || target>std::max<std::size_t>(1,n))
    throw std::invalid_argument("girvan_newman: invalid target community count");
  std::vector<Edge> active;
  for(auto [u,v]:edges) if(u!=v) active.push_back(std::minmax(u,v));
  std::sort(active.begin(),active.end()); active.erase(std::unique(active.begin(),active.end()),active.end());
  while(true) {
    std::vector<std::vector<std::size_t>> adjacency(n);
    for(auto [u,v]:active){adjacency[u].push_back(v);adjacency[v].push_back(u);}
    auto groups=components(adjacency); if(groups.size()>=target || active.empty()) return groups;
    std::vector<double> score(active.size());
    for(std::size_t source=0;source<n;++source) {
      std::vector<std::vector<std::size_t>> pred(n);
      std::vector<int> dist(n,-1); std::vector<double> paths(n), dependency(n);
      std::vector<std::size_t> order; std::queue<std::size_t> q;
      dist[source]=0;paths[source]=1;q.push(source);
      while(!q.empty()){auto u=q.front();q.pop();order.push_back(u);for(auto v:adjacency[u]){
        if(dist[v]<0){dist[v]=dist[u]+1;q.push(v);}
        if(dist[v]==dist[u]+1){paths[v]+=paths[u];pred[v].push_back(u);}
      }}
      for(auto it=order.rbegin();it!=order.rend();++it){auto v=*it;for(auto u:pred[v]){
        double contribution=(paths[u]/paths[v])*(1+dependency[v]);dependency[u]+=contribution;
        Edge e=std::minmax(u,v);auto pos=std::lower_bound(active.begin(),active.end(),e);
        score[static_cast<std::size_t>(pos-active.begin())]+=contribution;
      }}
    }
    const double maximum=*std::max_element(score.begin(),score.end());
    std::vector<Edge> kept; for(std::size_t i=0;i<active.size();++i)
      if(score[i]<maximum-1e-12) kept.push_back(active[i]);
    active.swap(kept);
  }
}

} // namespace datamunge::algorithms
