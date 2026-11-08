#include <datamunge/algorithms/advanced_graph.hpp>
#include <datamunge/algorithms/kruskal.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {
namespace {
struct Arc { std::size_t to, reverse, input; double capacity, original; bool forward; };
using Residual = std::vector<std::vector<Arc>>;

Residual make_residual(std::size_t n,const std::vector<CapacityEdge>& edges,
                       std::size_t s,std::size_t t) {
  if(s>=n||t>=n||s==t)throw std::invalid_argument("max flow: invalid terminal");
  Residual g(n);
  for(std::size_t i=0;i<edges.size();++i){auto [u,v,c]=edges[i];
    if(u>=n||v>=n||c<0||!std::isfinite(c))throw std::invalid_argument("max flow: invalid edge");
    Arc a{v,g[v].size(),i,c,c,true},b{u,g[u].size(),i,0,0,false};
    g[u].push_back(a);g[v].push_back(b);
  } return g;
}
MaxFlowResult finish(const Residual& g,const std::vector<CapacityEdge>& edges,
                     std::size_t s,double value) {
  MaxFlowResult r;r.max_flow=value;r.edge_flow.assign(edges.size(),0);
  for(const auto& row:g)for(const auto& a:row)if(a.forward)r.edge_flow[a.input]=a.original-a.capacity;
  std::vector<bool> seen(g.size());std::queue<std::size_t> q;q.push(s);seen[s]=true;
  while(!q.empty()){auto u=q.front();q.pop();r.min_cut_source_side.push_back(u);
    for(const auto&a:g[u])if(a.capacity>1e-12&&!seen[a.to]){seen[a.to]=true;q.push(a.to);}}
  return r;
}
struct DSU{std::vector<std::size_t>p;explicit DSU(std::size_t n):p(n){std::iota(p.begin(),p.end(),0);}
std::size_t f(std::size_t x){return p[x]==x?x:p[x]=f(p[x]);}void join(std::size_t a,std::size_t b){p[f(a)]=f(b);}};
}

MaxFlowResult push_relabel_max_flow(std::size_t n,const std::vector<CapacityEdge>& edges,
                                    std::size_t s,std::size_t t){
  auto g=make_residual(n,edges,s,t);std::vector<std::size_t>h(n),current(n);std::vector<double>excess(n);
  h[s]=n;for(auto&a:g[s])if(a.capacity>0){double x=a.capacity;a.capacity=0;g[a.to][a.reverse].capacity+=x;excess[a.to]+=x;excess[s]-=x;}
  std::queue<std::size_t>q;std::vector<bool>queued(n);for(std::size_t v=0;v<n;++v)if(v!=s&&v!=t&&excess[v]>1e-12){q.push(v);queued[v]=true;}
  while(!q.empty()){auto u=q.front();q.pop();queued[u]=false;while(excess[u]>1e-12){
      if(current[u]==g[u].size()){std::size_t minimum=std::numeric_limits<std::size_t>::max();
        for(auto&a:g[u])if(a.capacity>1e-12)minimum=std::min(minimum,h[a.to]);h[u]=minimum+1;current[u]=0;continue;}
      auto&a=g[u][current[u]];if(a.capacity>1e-12&&h[u]==h[a.to]+1){double x=std::min(excess[u],a.capacity);
        a.capacity-=x;g[a.to][a.reverse].capacity+=x;excess[u]-=x;excess[a.to]+=x;
        if(a.to!=s&&a.to!=t&&!queued[a.to]&&excess[a.to]>1e-12){q.push(a.to);queued[a.to]=true;}}
      else ++current[u];}
  }return finish(g,edges,s,excess[t]);
}

MinCutResult karger_min_cut(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                            std::size_t trials,std::uint64_t seed){
  for(auto[u,v]:edges)if(u>=n||v>=n)throw std::invalid_argument("karger: endpoint out of range");
  if(n<2)throw std::invalid_argument("karger: need at least two vertices");
  if(trials==0)trials=std::max<std::size_t>(1,n*n*static_cast<std::size_t>(std::ceil(std::log(n))));
  std::mt19937_64 rng(seed);MinCutResult best;best.cut_size=std::numeric_limits<std::size_t>::max();
  for(std::size_t trial=0;trial<trials;++trial){DSU d(n);std::size_t groups=n;
    while(groups>2){std::vector<std::pair<std::size_t,std::size_t>> crossing;
      for(auto e:edges)if(e.first!=e.second&&d.f(e.first)!=d.f(e.second))crossing.push_back(e);
      if(crossing.empty())break;auto e=crossing[rng()%crossing.size()];d.join(e.first,e.second);--groups;}
    std::size_t cut=0;for(auto[u,v]:edges)if(d.f(u)!=d.f(v))++cut;
    if(cut<best.cut_size){best={cut,{},{}};auto root=d.f(0);for(std::size_t v=0;v<n;++v)(d.f(v)==root?best.side_a:best.side_b).push_back(v);}}
  return best;
}



} // namespace datamunge::algorithms
