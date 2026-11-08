#include <datamunge/algorithms/euclidean_mst.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Matrix=std::vector<std::vector<long double>>;
Matrix distances(const std::vector<Point2D>& p){Matrix d(p.size(),std::vector<long double>(p.size()));for(std::size_t i=0;i<p.size();++i)for(std::size_t j=0;j<i;++j)d[i][j]=d[j][i]=std::hypot(static_cast<long double>(p[i].x)-p[j].x,static_cast<long double>(p[i].y)-p[j].y);return d;}
long double prim(const Matrix& d){if(d.empty())return 0;std::vector<long double> key(d.size(),std::numeric_limits<long double>::infinity());std::vector<bool> seen(d.size());key[0]=0;long double total=0;for(std::size_t k=0;k<d.size();++k){std::size_t u=d.size();for(std::size_t i=0;i<d.size();++i)if(!seen[i]&&(u==d.size()||key[i]<key[u]))u=i;seen[u]=true;total+=key[u];for(std::size_t v=0;v<d.size();++v)if(!seen[v])key[v]=std::min(key[v],d[u][v]);}return total;}
bool connected(const std::vector<std::vector<bool>>& a,std::size_t u,std::size_t v){std::vector<bool> seen(a.size());std::queue<std::size_t> q;q.push(u);seen[u]=true;while(!q.empty()){const auto x=q.front();q.pop();if(x==v)return true;for(std::size_t y=0;y<a.size();++y)if(a[x][y]&&!seen[y]){seen[y]=true;q.push(y);}}return false;}
void certificate(std::size_t n,const KruskalTrace& t){
 std::vector<std::size_t> order(t.input_edges.size());std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](auto i,auto j){const double a=std::get<2>(t.input_edges[i]),b=std::get<2>(t.input_edges[j]);return a<b||(a==b&&i<j);});
 std::vector<std::vector<bool>> adjacency(n,std::vector<bool>(n));std::size_t components=n,accepted=0;double total=0;std::vector<WeightedEdge> result;
 for(std::size_t k=0;k<t.steps.size();++k){const auto& s=t.steps[k];ASSERT_LT(k,order.size());ASSERT_EQ(s.input_index,order[k]);auto [u,v,w]=t.input_edges[s.input_index];ASSERT_LT(s.root_u,n);ASSERT_LT(s.root_v,n);EXPECT_TRUE(connected(adjacency,u,s.root_u));EXPECT_TRUE(connected(adjacency,v,s.root_v));const bool joins=!connected(adjacency,u,v);EXPECT_EQ(s.accepted,joins);EXPECT_EQ(s.root_u==s.root_v,!joins);
  if(joins){--components;++accepted;total+=w;adjacency[u][v]=adjacency[v][u]=true;result.push_back(t.input_edges[s.input_index]);}EXPECT_EQ(s.components_after,components);if(components==1)EXPECT_EQ(k+1,t.steps.size());}
 const auto& o=t.outcome;EXPECT_EQ(o.result.edges,result);EXPECT_EQ(o.result.total_weight,total);EXPECT_EQ(o.result.is_connected,n==0||components==1);EXPECT_EQ(o.work.input_edges,t.input_edges.size());EXPECT_EQ(o.work.scanned_edges,t.steps.size());EXPECT_EQ(o.work.accepted_edges,accepted);EXPECT_EQ(o.work.cycle_rejections,t.steps.size()-accepted);EXPECT_EQ(o.work.find_calls,2*t.steps.size());EXPECT_EQ(t.captured_entries,t.input_edges.size()+t.steps.size());if(n>1&&components>1)EXPECT_EQ(t.steps.size(),t.input_edges.size());
}
void geometric_certificate(const EuclideanMstTrace& t){const auto n=t.points.size();const auto& k=t.kruskal;certificate(n,k);EXPECT_EQ(t.distance_evaluations,n*(n-1)/2);EXPECT_EQ(t.captured_entries,n+k.captured_entries);const auto d=distances(t.points);std::size_t index=0;for(std::size_t i=0;i<n;++i)for(std::size_t j=i+1;j<n;++j){ASSERT_LT(index,k.input_edges.size());const auto [u,v,w]=k.input_edges[index++];EXPECT_EQ(u,i);EXPECT_EQ(v,j);EXPECT_NEAR(w,double(d[i][j]),1e-13*(1+double(d[i][j])));}EXPECT_EQ(index,k.input_edges.size());EXPECT_TRUE(k.outcome.result.is_connected);EXPECT_NEAR(k.outcome.result.total_weight,double(prim(d)),1e-11*(1+double(prim(d))));
 // Independent tree-path bottleneck certificate checks all complete graph pairs.
 std::vector<std::vector<std::pair<std::size_t,double>>> adj(n);for(auto [u,v,w]:k.outcome.result.edges){adj[u].push_back({v,w});adj[v].push_back({u,w});}
 for(std::size_t start=0;start<n;++start){std::vector<bool> seen(n);std::queue<std::pair<std::size_t,double>> q;q.push({start,0});seen[start]=true;while(!q.empty()){auto [u,maxedge]=q.front();q.pop();EXPECT_LE(maxedge,double(d[start][u])+1e-12*(1+double(d[start][u])));for(auto [v,w]:adj[u])if(!seen[v]){seen[v]=true;q.push({v,std::max(maxedge,w)});}}EXPECT_TRUE(std::all_of(seen.begin(),seen.end(),[](bool b){return b;}));}
}
long double exhaustive_trees(const Matrix& d){const auto n=d.size();if(n<2)return 0;std::size_t count=1;for(std::size_t i=0;i<n-2;++i)count*=n;long double best=std::numeric_limits<long double>::infinity();for(std::size_t code=0;code<count;++code){std::vector<std::size_t> sequence(n-2),degree(n,1);auto value=code;for(auto& v:sequence){v=value%n;value/=n;++degree[v];}long double sum=0;for(auto v:sequence){std::size_t leaf=0;while(degree[leaf]!=1)++leaf;sum+=d[leaf][v];--degree[leaf];--degree[v];}std::size_t a=n,b=n;for(std::size_t v=0;v<n;++v)if(degree[v]==1){if(a==n)a=v;else b=v;}sum+=d[a][b];best=std::min(best,sum);}return best;}
}
TEST(EuclideanMstTrace, OriginalSquareStableTiesAndActualCycleDecisions){
 const std::vector<Point2D> p{{0,0},{1,0},{1,1},{0,1}};const auto t=euclidean_minimum_spanning_tree_trace(p);geometric_certificate(t);EXPECT_EQ(t.kruskal.outcome.result.edges,(std::vector<WeightedEdge>{{0,1,1},{0,3,1},{1,2,1}}));EXPECT_EQ(t.distance_evaluations,6u);EXPECT_EQ(t.captured_entries,13u);
 // Three short cluster edges are scanned before the distant fourth point joins.
 const auto c=euclidean_minimum_spanning_tree_trace({{0,0},{1,0},{0,1},{10,0}});geometric_certificate(c);EXPECT_EQ(c.kruskal.outcome.work.cycle_rejections,1u);ASSERT_EQ(c.kruskal.steps.size(),4u);EXPECT_FALSE(c.kruskal.steps[2].accepted);EXPECT_TRUE(c.kruskal.steps[3].accepted);
 const auto flat=euclidean_minimum_spanning_tree_trace({{0,0},{0,0},{0,0},{0,0}});geometric_certificate(flat);EXPECT_EQ(flat.kruskal.outcome.result.total_weight,0);EXPECT_EQ(flat.kruskal.outcome.result.edges,(std::vector<WeightedEdge>{{0,1,0},{0,2,0},{0,3,0}}));
}
TEST(EuclideanMstTrace, RandomGeometryMatchesIndependentPrimAndTreePathCertificates){
 std::mt19937 rng(1956);for(int fixture=0;fixture<350;++fixture){std::vector<Point2D> p(rng()%30);for(auto& q:p){q.x=double(int(rng()%81)-40)/4;q.y=double(int(rng()%81)-40)/4;}const auto t=euclidean_minimum_spanning_tree_trace(p);geometric_certificate(t);const auto work=euclidean_minimum_spanning_tree_work(p);const auto ordinary=euclidean_minimum_spanning_tree(p);EXPECT_EQ(work.result.edges,t.kruskal.outcome.result.edges);EXPECT_EQ(ordinary.edges,work.result.edges);EXPECT_EQ(work.result.total_weight,ordinary.total_weight);EXPECT_EQ(work.work.distance_evaluations,t.distance_evaluations);EXPECT_EQ(work.work.kruskal.scanned_edges,t.kruskal.steps.size());}
 for(int fixture=0;fixture<30;++fixture){std::vector<Point2D> p(2+rng()%5);for(auto& q:p){q.x=int(rng()%9)-4;q.y=int(rng()%9)-4;}const auto r=euclidean_minimum_spanning_tree(p);EXPECT_NEAR(r.total_weight,double(exhaustive_trees(distances(p))),1e-11);}
}
TEST(EuclideanMstTrace, GenericForestsSignedWeightsParallelEdgesAndSelfLoopsAreActual){
 for(const auto& graph:std::vector<std::pair<std::size_t,std::vector<WeightedEdge>>>{{0,{}},{1,{}},{1,{{0,0,-10}}},{4,{{0,0,-10},{0,1,-2},{1,0,-2},{2,3,3}}},{4,{{0,1,1},{1,2,1},{0,2,1},{2,3,9}}}}){const auto t=kruskal_trace(graph.first,graph.second);certificate(graph.first,t);const auto r=kruskal(graph.first,graph.second);EXPECT_EQ(r.edges,t.outcome.result.edges);EXPECT_EQ(r.total_weight,t.outcome.result.total_weight);}
 std::mt19937 rng(1981);for(int fixture=0;fixture<400;++fixture){const std::size_t n=1+rng()%10;std::vector<WeightedEdge> edges;for(std::size_t u=0;u<n;++u)for(std::size_t v=u;v<n;++v)if(rng()%3)edges.emplace_back(u,v,int(rng()%11)-5);const auto t=kruskal_trace(n,edges);certificate(n,t);}
}
TEST(EuclideanMstTrace, ExactQuotasEmptyGraphsAndCopiesOwnAllRecords){
 const std::vector<Point2D> p{{0,0},{1,0},{0,1}};const auto t=euclidean_minimum_spanning_tree_trace(p,8);EXPECT_EQ(t.captured_entries,8u);
 for(std::size_t cap=1;cap<8;++cap){EXPECT_THROW((void)euclidean_minimum_spanning_tree_trace(p,cap),std::length_error);}
 EXPECT_EQ(euclidean_minimum_spanning_tree_trace(p,0).captured_entries,8u);EXPECT_EQ(euclidean_minimum_spanning_tree_trace({},1).captured_entries,0u);EXPECT_EQ(euclidean_minimum_spanning_tree_trace({{2,3}},1).captured_entries,1u);
 const std::vector<WeightedEdge> edges{{0,1,1},{1,2,2},{0,2,3}};EXPECT_EQ(kruskal_trace(3,edges,5).captured_entries,5u);
 for(std::size_t cap=1;cap<5;++cap){EXPECT_THROW((void)kruskal_trace(3,edges,cap),std::length_error);}
 EXPECT_EQ(kruskal_trace(3,edges,0).captured_entries,5u);auto copy=t;copy.points[0].x=99;copy.kruskal.input_edges[0]={0,0,99};copy.kruskal.steps[0].accepted=false;copy.kruskal.outcome.result.edges.clear();EXPECT_EQ(t.points[0].x,0);EXPECT_EQ(std::get<2>(t.kruskal.input_edges[0]),1);EXPECT_TRUE(t.kruskal.steps[0].accepted);EXPECT_EQ(t.kruskal.outcome.result.edges.size(),2u);
 auto input=p;const auto owned=euclidean_minimum_spanning_tree_trace(input);input[0].x=99;EXPECT_EQ(owned.points[0].x,0);
}
TEST(EuclideanMstTrace, PairGeometryFiniteDomainsAndOverflowAreCheckedBeforeUnsafeWork){
 for(std::size_t n=0;n<1000;++n)EXPECT_EQ(euclidean_mst_pair_count(n),n*(n-1)/2);EXPECT_EQ(euclidean_mst_pair_count(100000),4999950000ull);
 EXPECT_THROW((void)euclidean_mst_pair_count(std::numeric_limits<std::size_t>::max()),std::length_error);
 const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity(),max=std::numeric_limits<double>::max(),tiny=std::numeric_limits<double>::denorm_min();
 for(double bad:{nan,inf,-inf}){EXPECT_THROW((void)euclidean_minimum_spanning_tree({{bad,0}}),std::invalid_argument);
 EXPECT_THROW((void)euclidean_minimum_spanning_tree_trace({{0,bad},{0,0}},1),std::invalid_argument);
 EXPECT_THROW((void)kruskal(2,{{0,1,bad}}),std::invalid_argument);}
 EXPECT_THROW((void)kruskal(0,{{0,0,1}}),std::invalid_argument);
 EXPECT_THROW((void)kruskal_trace(2,{{0,2,1}},1),std::invalid_argument);
 EXPECT_THROW((void)kruskal(std::numeric_limits<std::size_t>::max(),{}),std::length_error);
 EXPECT_THROW((void)euclidean_minimum_spanning_tree({{-max,0},{max,0}}),std::overflow_error);
 EXPECT_THROW((void)euclidean_minimum_spanning_tree({{0,0},{max,max}}),std::overflow_error);
 EXPECT_THROW((void)euclidean_minimum_spanning_tree({{0,0},{max*.6,0},{0,max*.6}}),std::overflow_error);
 EXPECT_THROW((void)kruskal(3,{{0,1,max},{1,2,max}}),std::overflow_error);
 EXPECT_THROW((void)kruskal_work(3,{{0,1,-max},{1,2,-max}}),std::overflow_error);
 // Known input quota is checked before even attempting an overflowing distance.
 EXPECT_THROW((void)euclidean_minimum_spanning_tree_trace({{-max,0},{max,0}},2),std::length_error);
 EXPECT_EQ(euclidean_minimum_spanning_tree({{0,0},{max,0}}).total_weight,max);EXPECT_EQ(euclidean_minimum_spanning_tree({{0,0},{tiny,0}}).total_weight,tiny);
}
TEST(EuclideanMstTrace, ControlledLineGridScaleAndTranslationFamiliesHaveKnownOptima){
 for(std::size_t n:{0,1,2,4,8,16,32,64}){std::vector<Point2D> p;for(std::size_t i=0;i<n;++i)p.push_back({double(i),0});const auto t=euclidean_minimum_spanning_tree_trace(p);geometric_certificate(t);EXPECT_EQ(t.kruskal.outcome.result.total_weight,n?double(n-1):0);EXPECT_EQ(t.distance_evaluations,n*(n-1)/2);EXPECT_EQ(t.kruskal.outcome.work.accepted_edges,n?n-1:0);}
 for(int side=1;side<=6;++side){std::vector<Point2D> p;for(int y=0;y<side;++y)for(int x=0;x<side;++x)p.push_back({double(x),double(y)});const auto original=euclidean_minimum_spanning_tree(p);EXPECT_EQ(original.total_weight,double(p.size()-1));for(auto& q:p){q.x=4*q.x+100;q.y=4*q.y-70;}const auto transformed=euclidean_minimum_spanning_tree(p);EXPECT_EQ(transformed.total_weight,4*original.total_weight);ASSERT_EQ(transformed.edges.size(),original.edges.size());for(std::size_t i=0;i<original.edges.size();++i){EXPECT_EQ(std::get<0>(transformed.edges[i]),std::get<0>(original.edges[i]));EXPECT_EQ(std::get<1>(transformed.edges[i]),std::get<1>(original.edges[i]));}}
}
