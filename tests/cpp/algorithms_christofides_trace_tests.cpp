#include <datamunge/algorithms/christofides.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
namespace {
namespace a=datamunge::algorithms;
using Matrix=std::vector<std::vector<double>>;
Matrix star(std::size_t leaves,double spoke=1,double rim=1.5) {
  Matrix d(leaves+1,std::vector<double>(leaves+1,rim));
  for(std::size_t i=0;i<d.size();++i){d[i][i]=0;if(i)d[0][i]=d[i][0]=spoke;}return d;
}
long double tour_cost(const std::vector<std::size_t>& order,const Matrix& d) {
  long double result=0;for(std::size_t i=1;i<order.size();++i)result+=d[order[i-1]][order[i]];return result;
}
long double prim(const Matrix& d) {
  if(d.empty())return 0;const auto n=d.size();std::vector<long double> best(n,std::numeric_limits<long double>::infinity());
  std::vector<bool> used(n);best[0]=0;long double total=0;
  for(std::size_t pass=0;pass<n;++pass) {
    std::size_t u=n;for(std::size_t v=0;v<n;++v)if(!used[v] && (u==n || best[v]<best[u]))u=v;
    used[u]=true;total+=best[u];for(std::size_t v=0;v<n;++v)if(!used[v])best[v]=std::min(best[v],static_cast<long double>(d[u][v]));
  }return total;
}
long double match(const Matrix& d,std::vector<std::size_t> vertices) {
  if(vertices.empty())return 0;const auto u=vertices.back();vertices.pop_back();auto minimum=std::numeric_limits<long double>::infinity();
  for(std::size_t i=0;i<vertices.size();++i){auto rest=vertices;const auto v=rest[i];rest.erase(rest.begin()+i);minimum=std::min(minimum,static_cast<long double>(d[u][v])+match(d,std::move(rest)));}return minimum;
}
long double optimum(const Matrix& d) {
  if(d.empty())return 0;std::vector<std::size_t> vertices(d.size()-1);std::iota(vertices.begin(),vertices.end(),1);
  auto minimum=std::numeric_limits<long double>::infinity();
  do {std::vector<std::size_t> order{0};order.insert(order.end(),vertices.begin(),vertices.end());order.push_back(0);minimum=std::min(minimum,tour_cost(order,d));}while(std::next_permutation(vertices.begin(),vertices.end()));return minimum;
}
void verify(const Matrix& d,std::size_t start=0,bool symmetric=true) {
  const auto t=a::christofides_trace(d,start),again=a::christofides_trace(d,start);const auto ordinary=a::christofides_tsp(d,start);const auto n=d.size();
  EXPECT_EQ(t.result.tour,ordinary.tour);EXPECT_EQ(t.result.cost,ordinary.cost);EXPECT_EQ(t.result.tour,again.result.tour);EXPECT_EQ(t.euler_edge_ids,again.euler_edge_ids);
  EXPECT_EQ(t.work.input_entries,n*n);EXPECT_EQ(t.work.complete_edges,n*(n?n-1:0)/2);
  if(!n){EXPECT_TRUE(t.result.tour.empty());EXPECT_EQ(t.result.cost,0);EXPECT_EQ(t.work.matching_table_entries,0u);return;}
  ASSERT_EQ(t.tree.size(),n-1);ASSERT_EQ(t.result.tour.size(),n+1);EXPECT_EQ(t.result.tour.front(),start);EXPECT_EQ(t.result.tour.back(),start);
  std::set<std::size_t> unique(t.result.tour.begin(),t.result.tour.end());EXPECT_EQ(unique.size(),n);
  EXPECT_DOUBLE_EQ(t.result.cost,static_cast<double>(tour_cost(t.result.tour,d)));
  auto edges=t.tree;edges.insert(edges.end(),t.matching.begin(),t.matching.end());std::vector<std::size_t> degree(n),matched(n);
  long double tree_cost=0,matching_cost=0;
  for(const auto [u,v,w]:t.tree){ASSERT_LT(u,n);ASSERT_LT(v,n);EXPECT_EQ(w,d[u][v]);++degree[u];++degree[v];tree_cost+=w;}
  std::vector<std::size_t> odd;for(std::size_t v=0;v<n;++v)if(degree[v]%2)odd.push_back(v);EXPECT_EQ(t.odd_vertices,odd);
  EXPECT_EQ(t.matching.size(),odd.size()/2);
  for(const auto [u,v,w]:t.matching){ASSERT_LT(u,n);ASSERT_LT(v,n);EXPECT_EQ(w,d[u][v]);++matched[u];++matched[v];matching_cost+=w;}
  for(std::size_t v=0;v<n;++v)EXPECT_EQ(matched[v],degree[v]%2);
  EXPECT_EQ(t.tree_cost,tree_cost);EXPECT_EQ(t.matching_cost,matching_cost);
  if(symmetric){EXPECT_NEAR(t.tree_cost,prim(d),1e-10);EXPECT_NEAR(t.matching_cost,match(d,odd),1e-10);}
  ASSERT_EQ(t.euler_vertices.size(),edges.size()+1);ASSERT_EQ(t.euler_edge_ids.size(),edges.size());
  EXPECT_EQ(t.euler_vertices.front(),start);EXPECT_EQ(t.euler_vertices.back(),start);std::set<std::size_t> ids;
  for(std::size_t p=0;p<t.euler_edge_ids.size();++p) {
    const auto id=t.euler_edge_ids[p];ASSERT_LT(id,edges.size());ids.insert(id);const auto [u,v,w]=edges[id];
    const auto x=t.euler_vertices[p],y=t.euler_vertices[p+1];EXPECT_TRUE((x==u && y==v)||(x==v && y==u));
  }
  EXPECT_EQ(ids.size(),edges.size());EXPECT_EQ(t.euler_cost,tour_cost(t.euler_vertices,d));
  if(symmetric)EXPECT_NEAR(t.euler_cost,t.tree_cost+t.matching_cost,1e-10);
  std::set<std::size_t> seen;std::vector<std::size_t> positions,shortcut;
  for(std::size_t p=0;p<t.euler_vertices.size();++p)if(seen.insert(t.euler_vertices[p]).second){positions.push_back(p);shortcut.push_back(t.euler_vertices[p]);}
  EXPECT_EQ(t.kept_positions,positions);shortcut.push_back(start);EXPECT_EQ(t.result.tour,shortcut);
  EXPECT_EQ(t.work.euler_edge_visits,edges.size());EXPECT_EQ(t.work.shortcut_vertices,t.euler_vertices.size());
  const auto k=odd.size();EXPECT_EQ(t.work.matching_table_entries,std::size_t{1}<<k);
  EXPECT_EQ(t.work.matching_masks,k?(std::size_t{1}<<(k-1))-1:0);
  EXPECT_EQ(t.work.matching_candidates,k?(k-2)*(std::size_t{1}<<(k-2))+1:0);
}
}
TEST(Christofides, OriginalRoundedSquareAndEveryStartRetainCostFour) {
  Matrix d{{0,1,1.414,1},{1,0,1,1.414},{1.414,1,0,1},{1,1.414,1,0}};ASSERT_TRUE(a::is_tsp_metric(d));
  for(std::size_t s=0;s<4;++s){verify(d,s);EXPECT_EQ(a::christofides_tsp(d,s).cost,4);}
}
TEST(Christofides, FourOddVerticesExposeActualEulerRepeatsAndShortcutPositions) {
  const auto d=star(4);verify(d);const auto t=a::christofides_trace(d);
  EXPECT_EQ(t.odd_vertices,(std::vector<std::size_t>{1,2,3,4}));EXPECT_EQ(t.tree_cost,4);EXPECT_EQ(t.matching_cost,3);EXPECT_EQ(t.euler_cost,7);EXPECT_EQ(t.result.cost,6.5);
  EXPECT_EQ(t.euler_vertices,(std::vector<std::size_t>{0,4,3,0,2,1,0}));EXPECT_EQ(t.result.tour,(std::vector<std::size_t>{0,4,3,2,1,0}));
  EXPECT_EQ(t.work.matching_table_entries,16u);EXPECT_EQ(t.work.matching_masks,7u);EXPECT_EQ(t.work.matching_candidates,9u);
}
TEST(Christofides, ParallelTreeAndMatchingEdgesKeepDistinctEulerIdentities) {
  verify({{0,2},{2,0}});const auto t=a::christofides_trace({{0,2},{2,0}});
  ASSERT_EQ(t.tree.size(),1u);ASSERT_EQ(t.matching.size(),1u);EXPECT_EQ(t.tree[0],t.matching[0]);EXPECT_NE(t.euler_edge_ids[0],t.euler_edge_ids[1]);
}
TEST(Christofides, ExhaustiveFourVertexZeroOneTwoCostsAgreeWithIndependentStageOracles) {
  std::size_t count=0,metrics=0;
  for(std::size_t code=0;code<729;++code) {
    Matrix d(4,std::vector<double>(4));auto digits=code;
    for(std::size_t i=0;i<4;++i)for(std::size_t j=i+1;j<4;++j){d[i][j]=d[j][i]=digits%3;digits/=3;}
    verify(d);++count;if(a::is_tsp_metric(d)){++metrics;const auto t=a::christofides_trace(d);const auto opt=optimum(d);EXPECT_LE(t.result.cost,1.5L*opt);EXPECT_LE(t.tree_cost,opt);EXPECT_LE(t.matching_cost,opt/2);EXPECT_LE(t.result.cost,t.euler_cost);}
  }EXPECT_EQ(count,729u);EXPECT_GT(metrics,0u);
}
TEST(Christofides, RandomIntegerShortestPathMetricsMeetBoundAgainstExhaustiveTours) {
  std::mt19937_64 random(76);
  for(int trial=0;trial<120;++trial) {
    const auto n=2+random()%7;Matrix d(n,std::vector<double>(n));
    for(std::size_t i=0;i<n;++i)for(std::size_t j=i+1;j<n;++j)d[i][j]=d[j][i]=1+random()%20;
    for(std::size_t k=0;k<n;++k)for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)d[i][j]=std::min(d[i][j],d[i][k]+d[k][j]);
    ASSERT_TRUE(a::is_tsp_metric(d));verify(d,trial%n);const auto t=a::christofides_trace(d,trial%n);const auto opt=optimum(d);
    EXPECT_LE(t.result.cost,1.5L*opt);EXPECT_LE(t.tree_cost,opt);EXPECT_LE(t.matching_cost,opt/2);EXPECT_LE(t.result.cost,t.euler_cost);
  }
}
TEST(Christofides, ThousandPlanarInputsHaveOwnedSymmetricDistancesAndValidStageCertificates) {
  std::mt19937_64 random(5);
  for(int trial=0;trial<1000;++trial) {
    const auto n=random()%10;std::vector<std::pair<double,double>> points;
    for(std::size_t i=0;i<n;++i)points.emplace_back(static_cast<int>(random()%21)-10,static_cast<int>(random()%21)-10);
    const auto d=a::euclidean_tsp_distances(points);ASSERT_TRUE(a::is_tsp_metric(d,1e-12));verify(d,n?trial%n:0);
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){EXPECT_EQ(d[i][j],d[j][i]);EXPECT_DOUBLE_EQ(d[i][j],std::hypot(points[i].first-points[j].first,points[i].second-points[j].second));}
  }
}
TEST(Christofides, EmptySingletonZeroCostsAndOwnedCopiesHaveExplicitSemantics) {
  verify({},999);verify({{0}});verify({{7}});verify(Matrix(4,std::vector<double>(4)));
  auto d=star(4);auto t=a::christofides_trace(d);auto copy=t;d[0][1]=99;t.tree[0]=a::WeightedEdge{0,0,0};t.result.tour.clear();
  EXPECT_EQ(copy.result.cost,6.5);EXPECT_EQ(std::get<2>(copy.tree[0]),1);
}
TEST(Christofides, MetricCheckerSeparatesTriangleSymmetryDiagonalAndTolerance) {
  EXPECT_TRUE(a::is_tsp_metric({}));EXPECT_TRUE(a::is_tsp_metric(Matrix(3,std::vector<double>(3))));
  EXPECT_FALSE(a::is_tsp_metric({{0,1,3},{1,0,1},{3,1,0}}));EXPECT_FALSE(a::is_tsp_metric({{0,1},{2,0}}));EXPECT_FALSE(a::is_tsp_metric({{0.1}}));
  EXPECT_TRUE(a::is_tsp_metric({{0,1,2.0001},{1,0,1},{2.0001,1,0}},0.001));
  const auto max=std::numeric_limits<double>::max();EXPECT_TRUE(a::is_tsp_metric({{0,max,max},{max,0,max},{max,max,0}}));
  EXPECT_THROW(a::is_tsp_metric({{0}},-1),std::invalid_argument);
  EXPECT_THROW(a::is_tsp_metric({{0}},std::numeric_limits<double>::infinity()),std::invalid_argument);
}
TEST(Christofides, NonmetricAndAsymmetricAcceptanceIsPreservedWithoutApproximationClaim) {
  Matrix d{{0,1,10},{9,0,2},{3,4,0}};verify(d,0,false);const auto t=a::christofides_trace(d);
  EXPECT_FALSE(a::is_tsp_metric(d));EXPECT_EQ(t.tree_cost+t.matching_cost,13);EXPECT_EQ(t.euler_cost,23);EXPECT_EQ(t.result.cost,23);
}
TEST(Christofides, ExactMatchingCapsRejectBeforeAllocationAndZeroOddCapIsReal) {
  const auto d=star(8);a::ChristofidesOptions cap;cap.max_odd_vertices=8;cap.max_matching_states=256;
  EXPECT_EQ(a::christofides_trace(d,0,cap).work.matching_table_entries,256u);EXPECT_EQ(a::christofides_tsp_with_options(d,0,cap).cost,12.5);
  cap.max_matching_states=255;
  EXPECT_THROW(a::christofides_trace(d,0,cap),std::length_error);
  EXPECT_THROW(a::christofides_tsp_with_options(d,0,cap),std::length_error);
  cap.max_matching_states=0;cap.max_odd_vertices=7;
  EXPECT_THROW(a::christofides_trace(d,0,cap),std::invalid_argument);
  cap.max_odd_vertices=0;EXPECT_EQ(a::christofides_trace({{0}},0,cap).result.cost,0);
  EXPECT_THROW(a::christofides_trace({{0,1},{1,0}},0,cap),std::invalid_argument);
  cap.max_odd_vertices=23;
  EXPECT_THROW(a::christofides_trace({},0,cap),std::invalid_argument);
  EXPECT_THROW(a::christofides_tsp(star(24)),std::invalid_argument);
}
TEST(Christofides, MalformedWholeMatrixRejectsBeforeEarlyMetricFailureOrIndexing) {
  const auto nan=std::numeric_limits<double>::quiet_NaN();const auto inf=std::numeric_limits<double>::infinity();
  EXPECT_THROW(a::christofides_tsp({{0,1},{1}}),std::invalid_argument);
  EXPECT_THROW(a::christofides_trace({{0,-1},{1,0}}),std::invalid_argument);
  EXPECT_THROW(a::is_tsp_metric({{1,0},{0,nan}}),std::invalid_argument);
  EXPECT_THROW(a::christofides_tsp({{inf}}),std::invalid_argument);
  EXPECT_THROW(a::christofides_tsp({{0}},1),std::invalid_argument);
  EXPECT_THROW(a::euclidean_tsp_distances({{0,nan}}),std::invalid_argument);
}
TEST(Christofides, WidenedMatchingCandidatesCanOverflowDoubleWhileFinalTourStillFits) {
  const auto max=std::numeric_limits<double>::max();auto d=star(4,max/10,max);
  for(const auto [u,v]:std::vector<std::pair<int,int>>{{1,2},{3,4},{2,3}})d[u][v]=d[v][u]=max/5;
  const auto t=a::christofides_trace(d);EXPECT_TRUE(std::isfinite(t.result.cost));EXPECT_NEAR(t.result.cost/max,0.8,1e-15);EXPECT_NEAR(t.matching_cost/max,0.4,1e-15);
  EXPECT_THROW(a::christofides_tsp({{0,max},{max,0}}),std::overflow_error);
  EXPECT_THROW(a::christofides_trace(star(4,max/2,max)),std::overflow_error);
}
TEST(Christofides, PlanarDifferenceOverflowIsCheckedAfterWideningAndDuplicatePointsAreAllowed) {
  const auto max=std::numeric_limits<double>::max();const auto d=a::euclidean_tsp_distances({{max,max},{max,max},{0,max}});
  EXPECT_EQ(d[0][1],0);EXPECT_EQ(d[0][2],max);
  EXPECT_THROW(a::euclidean_tsp_distances({{-max,0},{max,0}}),std::overflow_error);
}
TEST(Christofides, MatchingStorageAndRelaxationCountsFollowOddSetRatherThanTotalVertices) {
  for(std::size_t k=2;k<=16;k+=2) {
    const auto t=a::christofides_trace(star(k));EXPECT_EQ(t.odd_vertices.size(),k);
    EXPECT_EQ(t.work.matching_table_entries,std::size_t{1}<<k);EXPECT_EQ(t.work.matching_masks,(std::size_t{1}<<(k-1))-1);
    EXPECT_EQ(t.work.matching_candidates,(k-2)*(std::size_t{1}<<(k-2))+1);EXPECT_EQ(t.result.cost,1.5*k+0.5);
  }
  constexpr std::size_t n=1200;Matrix line(n,std::vector<double>(n));
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)line[i][j]=i>j?i-j:j-i;
  const auto t=a::christofides_trace(line);EXPECT_EQ(t.result.tour.size(),n+1);EXPECT_EQ(t.result.cost,2*(n-1));EXPECT_EQ(t.odd_vertices.size(),2u);EXPECT_EQ(t.work.matching_table_entries,4u);
}

TEST(Christofides, NativeHubFixtureChecksCostsAndVertexOverflow) {
  EXPECT_EQ(a::hub_tsp_distances(4),star(4));EXPECT_EQ(a::hub_tsp_distances(0),(Matrix{{0}}));
  EXPECT_TRUE(a::is_tsp_metric(a::hub_tsp_distances(4,1,2)));EXPECT_FALSE(a::is_tsp_metric(a::hub_tsp_distances(4,1,3)));
  EXPECT_THROW(a::hub_tsp_distances(std::numeric_limits<std::size_t>::max()),std::length_error);
  EXPECT_THROW(a::hub_tsp_distances(4,-1,1),std::invalid_argument);
  EXPECT_THROW(a::hub_tsp_distances(4,1,std::numeric_limits<double>::infinity()),std::invalid_argument);
}
