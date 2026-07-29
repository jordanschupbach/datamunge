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

MaxFlowResult dinic_max_flow(std::size_t n,const std::vector<CapacityEdge>& edges,
                             std::size_t s,std::size_t t){
  auto g=make_residual(n,edges,s,t);double flow=0;std::vector<int>level(n);std::vector<std::size_t>it(n);
  while(true){std::fill(level.begin(),level.end(),-1);std::queue<std::size_t>q;q.push(s);level[s]=0;
    while(!q.empty()){auto u=q.front();q.pop();for(auto&a:g[u])if(a.capacity>1e-12&&level[a.to]<0){level[a.to]=level[u]+1;q.push(a.to);}}
    if(level[t]<0)break;std::fill(it.begin(),it.end(),0);
    auto dfs=[&](auto&&self,std::size_t u,double pushed)->double{if(u==t)return pushed;
      for(auto&i=it[u];i<g[u].size();++i){auto&a=g[u][i];if(a.capacity>1e-12&&level[a.to]==level[u]+1){
        double x=self(self,a.to,std::min(pushed,a.capacity));if(x>1e-12){a.capacity-=x;g[a.to][a.reverse].capacity+=x;return x;}}}return 0;};
    while(double x=dfs(dfs,s,std::numeric_limits<double>::infinity()))flow+=x;
  }return finish(g,edges,s,flow);
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

DirectedBranchingResult chu_liu_edmonds(std::size_t n,
 const std::vector<std::tuple<std::size_t,std::size_t,double>>& input,std::size_t root){
  if(root>=n)throw std::invalid_argument("chu_liu_edmonds: root out of range");
  struct E{int u,v;double w;};std::vector<E> edges;for(auto[u,v,w]:input){if(u>=n||v>=n)throw std::invalid_argument("chu_liu_edmonds: endpoint out of range");edges.push_back({(int)u,(int)v,w});}
  int N=(int)n,R=(int)root;double answer=0;
  while(true){std::vector<double>in(N,std::numeric_limits<double>::infinity());std::vector<int>pre(N,-1);
    for(auto e:edges)if(e.u!=e.v&&e.w<in[e.v]){in[e.v]=e.w;pre[e.v]=e.u;}in[R]=0;
    for(int v=0;v<N;++v)if(!std::isfinite(in[v]))return {0,false};
    int cycles=0;std::vector<int>id(N,-1),seen(N,-1);
    for(int v=0;v<N;++v){answer+=in[v];int x=v;while(seen[x]!=v&&id[x]<0&&x!=R){seen[x]=v;x=pre[x];}
      if(x!=R&&id[x]<0){for(int y=pre[x];y!=x;y=pre[y])id[y]=cycles;id[x]=cycles++;}}
    if(cycles==0)break;for(int&i:id)if(i<0)i=cycles++;
    std::vector<E> next;for(auto e:edges){int u=id[e.u],v=id[e.v];double w=e.w;if(u!=v)w-=in[e.v];next.push_back({u,v,w});}
    R=id[R];N=cycles;edges.swap(next);
  }return {answer,true};
}

MinimumSpanningTree euclidean_minimum_spanning_tree(const std::vector<Point2D>& points){
  std::vector<std::tuple<std::size_t,std::size_t,double>> edges;
  for(std::size_t i=0;i<points.size();++i)for(std::size_t j=i+1;j<points.size();++j)
    edges.emplace_back(i,j,std::hypot(points[i].x-points[j].x,points[i].y-points[j].y));
  return kruskal(points.size(),edges);
}

} // namespace datamunge::algorithms
