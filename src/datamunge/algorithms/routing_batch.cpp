#include <datamunge/algorithms/routing_batch.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {
namespace {
struct DSU{std::vector<std::size_t>p,r;explicit DSU(std::size_t n):p(n),r(n){std::iota(p.begin(),p.end(),0);}
std::size_t f(std::size_t x){return p[x]==x?x:p[x]=f(p[x]);}bool join(std::size_t a,std::size_t b){a=f(a);b=f(b);if(a==b)return false;if(r[a]<r[b])std::swap(a,b);p[b]=a;if(r[a]==r[b])++r[a];return true;}};
void valid(std::size_t n,const std::vector<WeightedEdge>&e){for(auto[u,v,w]:e)if(u>=n||v>=n||!std::isfinite(w))throw std::invalid_argument("routing algorithm: invalid edge");}
}
LongestPathResult dag_longest_paths(std::size_t n,const std::vector<WeightedEdge>&edges,std::size_t source){
 valid(n,edges);if(source>=n)throw std::invalid_argument("dag_longest_paths: source out of range");
 std::vector<std::vector<std::pair<std::size_t,double>>>a(n);std::vector<std::size_t>degree(n);
 for(auto[u,v,w]:edges){a[u].push_back({v,w});++degree[v];}std::queue<std::size_t>q;for(std::size_t i=0;i<n;++i)if(!degree[i])q.push(i);
 std::vector<std::size_t>order;while(!q.empty()){auto u=q.front();q.pop();order.push_back(u);for(auto[v,w]:a[u])if(--degree[v]==0)q.push(v);}
 if(order.size()!=n)throw std::invalid_argument("dag_longest_paths: graph contains a cycle");
 LongestPathResult r;r.distance.assign(n,-std::numeric_limits<double>::infinity());r.predecessor.assign(n,n);r.distance[source]=0;
 for(auto u:order)if(std::isfinite(r.distance[u]))for(auto[v,w]:a[u])if(r.distance[u]+w>r.distance[v]){r.distance[v]=r.distance[u]+w;r.predecessor[v]=u;}return r;
}
MinimumSpanningTree boruvka(std::size_t n,const std::vector<WeightedEdge>&edges){
 valid(n,edges);MinimumSpanningTree out;DSU d(n);std::size_t groups=n;
 while(groups>1){std::vector<std::size_t>best(n,edges.size());for(std::size_t i=0;i<edges.size();++i){auto[u,v,w]=edges[i];auto a=d.f(u),b=d.f(v);if(a==b)continue;
   if(best[a]==edges.size()||w<std::get<2>(edges[best[a]]))best[a]=i;if(best[b]==edges.size()||w<std::get<2>(edges[best[b]]))best[b]=i;}
  bool changed=false;for(auto i:best)if(i<edges.size()){auto[u,v,w]=edges[i];if(d.join(u,v)){out.edges.push_back(edges[i]);out.total_weight+=w;--groups;changed=true;}}if(!changed)break;}
 out.is_connected=n<=1||groups==1;return out;
}
MinimumSpanningTree reverse_delete(std::size_t n,const std::vector<WeightedEdge>&input){
 valid(n,input);auto edges=input;std::sort(edges.begin(),edges.end(),[](auto&a,auto&b){return std::get<2>(a)>std::get<2>(b);});
 std::vector<bool>keep(edges.size(),true);auto component_count=[&](std::size_t skip){DSU d(n);for(std::size_t i=0;i<edges.size();++i)if(keep[i]&&i!=skip)d.join(std::get<0>(edges[i]),std::get<1>(edges[i]));
   std::size_t count=0;for(std::size_t v=0;v<n;++v)if(d.f(v)==v)++count;return count;};
 const auto initial_components=component_count(edges.size());for(std::size_t i=0;i<edges.size();++i)if(component_count(i)==initial_components)keep[i]=false;
 MinimumSpanningTree r;r.is_connected=n<=1||initial_components==1;for(std::size_t i=0;i<edges.size();++i)if(keep[i]){r.edges.push_back(edges[i]);r.total_weight+=std::get<2>(edges[i]);}return r;
}
std::vector<std::size_t> nonblocking_switch_routes(std::size_t ni,std::size_t no,
 const std::vector<std::pair<std::size_t,std::size_t>>&c){
 std::vector<std::vector<std::size_t>>inc(ni),out(no);for(std::size_t i=0;i<c.size();++i){auto[u,v]=c[i];if(u>=ni||v>=no)throw std::invalid_argument("switch routes: endpoint out of range");inc[u].push_back(i);out[v].push_back(i);}
 for(auto&x:inc)if(x.size()>2)throw std::invalid_argument("switch routes: input degree exceeds two");for(auto&x:out)if(x.size()>2)throw std::invalid_argument("switch routes: output degree exceeds two");
 std::vector<std::size_t>color(c.size(),2);for(std::size_t start=0;start<c.size();++start)if(color[start]==2){color[start]=0;std::queue<std::size_t>q;q.push(start);while(!q.empty()){auto e=q.front();q.pop();auto[u,v]=c[e];
   for(auto f:inc[u])if(color[f]==2){color[f]=1-color[e];q.push(f);}else if(f!=e&&color[f]==color[e])throw std::invalid_argument("switch routes: not two-routable");
   for(auto f:out[v])if(color[f]==2){color[f]=1-color[e];q.push(f);}else if(f!=e&&color[f]==color[e])throw std::invalid_argument("switch routes: not two-routable");}}return color;
}
JohnsonAllPairsResult johnson_all_pairs_shortest_paths(std::size_t n,const std::vector<WeightedEdge>&edges){
 valid(n,edges);JohnsonAllPairsResult r;r.distance.assign(n,std::vector<double>(n,std::numeric_limits<double>::infinity()));std::vector<double>h(n);
 for(std::size_t pass=0;pass<n;++pass){bool change=false;for(auto[u,v,w]:edges)if(h[u]+w<h[v]){h[v]=h[u]+w;change=true;}if(!change)break;if(pass+1==n){r.has_negative_cycle=true;return r;}}
 std::vector<std::vector<std::pair<std::size_t,double>>>a(n);for(auto[u,v,w]:edges)a[u].push_back({v,w+h[u]-h[v]});
 for(std::size_t s=0;s<n;++s){using P=std::pair<double,std::size_t>;std::priority_queue<P,std::vector<P>,std::greater<P>>q;r.distance[s][s]=0;q.push({0,s});
  while(!q.empty()){auto[d,u]=q.top();q.pop();if(d!=r.distance[s][u])continue;for(auto[v,w]:a[u])if(d+w<r.distance[s][v]){r.distance[s][v]=d+w;q.push({d+w,v});}}
  for(std::size_t v=0;v<n;++v)if(std::isfinite(r.distance[s][v]))r.distance[s][v]+=-h[s]+h[v];}return r;
}
std::vector<std::vector<bool>> transitive_closure(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>&edges,bool reflexive){
 std::vector<std::vector<bool>>r(n,std::vector<bool>(n));for(auto[u,v]:edges){if(u>=n||v>=n)throw std::invalid_argument("transitive_closure: endpoint out of range");r[u][v]=true;}if(reflexive)for(std::size_t i=0;i<n;++i)r[i][i]=true;
 for(std::size_t k=0;k<n;++k)for(std::size_t i=0;i<n;++i)if(r[i][k])for(std::size_t j=0;j<n;++j)r[i][j]=r[i][j]||r[k][j];return r;
}
}
