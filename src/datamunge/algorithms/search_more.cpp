#include <datamunge/algorithms/search_more.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
namespace datamunge::algorithms {
namespace {
using Adj=std::vector<std::vector<std::size_t>>;
Adj adj(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>&e){Adj a(n);for(auto[u,v]:e){if(u>=n||v>=n)throw std::invalid_argument("search: endpoint");a[u].push_back(v);}return a;}
void term(std::size_t n,std::size_t s,std::size_t g){if(s>=n||g>=n)throw std::invalid_argument("search: terminal");}
GridPathResult grid(const std::vector<std::vector<bool>>&b,std::pair<int,int>s,std::pair<int,int>g,bool jump){if(b.empty()||b[0].empty())throw std::invalid_argument("grid: empty");int R=b.size(),C=b[0].size();for(auto&r:b)if((int)r.size()!=C)throw std::invalid_argument("grid: ragged");auto ok=[&](int r,int c){return r>=0&&r<R&&c>=0&&c<C&&!b[r][c];};if(!ok(s.first,s.second)||!ok(g.first,g.second))return{};
 int N=R*C,src=s.first*C+s.second,dst=g.first*C+g.second;std::vector<double>d(N,std::numeric_limits<double>::infinity());std::vector<int>p(N,-1);using Q=std::pair<double,int>;std::priority_queue<Q,std::vector<Q>,std::greater<Q>>q;d[src]=0;q.push({static_cast<double>(std::abs(s.first-g.first)+std::abs(s.second-g.second)),src});int dr[]={1,-1,0,0},dc[]={0,0,1,-1};while(!q.empty()){auto[f,u]=q.top();q.pop();int r=u/C,c=u%C;if(f!=d[u]+std::abs(r-g.first)+std::abs(c-g.second))continue;if(u==dst)break;for(int k=0;k<4;++k){int nr=r+dr[k],nc=c+dc[k],steps=1;if(jump)while(ok(nr+dr[k],nc+dc[k])&&(nr!=g.first||nc!=g.second)){nr+=dr[k];nc+=dc[k];++steps;}if(!ok(nr,nc))continue;int v=nr*C+nc;if(d[u]+steps<d[v]){d[v]=d[u]+steps;p[v]=u;q.push({d[v]+std::abs(nr-g.first)+std::abs(nc-g.second),v});}}}
 if(!std::isfinite(d[dst]))return{};GridPathResult out;out.found=true;out.cost=d[dst];std::vector<int>nodes;for(int v=dst;v!=-1;v=p[v])nodes.push_back(v);std::reverse(nodes.begin(),nodes.end());for(std::size_t i=0;i<nodes.size();++i){int r=nodes[i]/C,c=nodes[i]%C;if(i){int pr=nodes[i-1]/C,pc=nodes[i-1]%C,rr=pr,cc=pc;while(rr!=r||cc!=c){rr+=(r>rr)-(r<rr);cc+=(c>cc)-(c<cc);out.path.push_back({rr,cc});}}else out.path.push_back({r,c});}return out;}
}
BasicPathResult brute_force_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>&e,std::size_t s,std::size_t g){term(n,s,g);auto a=adj(n,e);BasicPathResult best;std::vector<bool>on(n);std::vector<std::size_t>cur;std::function<void(std::size_t)>go=[&](auto u){++best.expanded;cur.push_back(u);on[u]=true;if(u==g&&(!best.found||cur.size()<best.path.size())){best.path=cur;best.found=true;}else if(!best.found||cur.size()<best.path.size())for(auto v:a[u])if(!on[v])go(v);on[u]=false;cur.pop_back();};go(s);return best;}
BasicPathResult depth_first_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>&e,std::size_t s,std::size_t g){term(n,s,g);auto a=adj(n,e);BasicPathResult r;std::vector<bool>seen(n);std::function<bool(std::size_t)>go=[&](auto u){++r.expanded;seen[u]=true;r.path.push_back(u);if(u==g)return true;for(auto v:a[u])if(!seen[v]&&go(v))return true;r.path.pop_back();return false;};r.found=go(s);return r;}
BasicPathResult iterative_deepening_dfs(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>&e,std::size_t s,std::size_t g,std::size_t maxd){term(n,s,g);auto a=adj(n,e);BasicPathResult out;for(std::size_t limit=0;limit<=maxd;++limit){std::vector<bool>on(n);std::vector<std::size_t>cur;std::function<bool(std::size_t,std::size_t)>go=[&](auto u,auto depth){++out.expanded;cur.push_back(u);if(u==g)return true;if(depth<limit){on[u]=true;for(auto v:a[u])if(!on[v]&&go(v,depth+1))return true;on[u]=false;}cur.pop_back();return false;};if(go(s,0)){out.path=cur;out.found=true;break;}}return out;}
GridPathResult d_star_replan(const std::vector<std::vector<bool>>&b,std::pair<int,int>s,std::pair<int,int>g){return grid(b,s,g,false);}
GridPathResult jump_point_search(const std::vector<std::vector<bool>>&b,std::pair<int,int>s,std::pair<int,int>g){return grid(b,s,g,true);}
PlanningResult general_problem_solver(std::uint64_t initial,std::uint64_t goals,const std::vector<PlanningAction>&actions,std::size_t max){PlanningResult out;std::unordered_set<std::uint64_t>seen;std::function<bool(std::uint64_t,std::size_t)>go=[&](auto state,auto depth){if((state&goals)==goals){out.final_state=state;return true;}if(depth==max||!seen.insert(state).second)return false;for(auto&a:actions)if((state&a.require)==a.require){auto next=(state&~a.remove)|a.add;out.actions.push_back(a.name);if(go(next,depth+1))return true;out.actions.pop_back();}return false;};out.found=go(initial,0);return out;}
}
