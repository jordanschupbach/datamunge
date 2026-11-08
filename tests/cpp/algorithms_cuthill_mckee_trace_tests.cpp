#include <gtest/gtest.h>
#include <datamunge/algorithms/cuthill_mckee.hpp>
#include <algorithm>
#include <climits>
#include <numeric>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
using Graph=std::vector<std::vector<int>>;
Graph from_mask(int n,unsigned long long mask) {
  Graph g(n);unsigned bit=0;
  for(int i=0;i<n;++i) for(int j=i+1;j<n;++j,++bit) if((mask>>bit)&1) {g[i].push_back(j);g[j].push_back(i);}
  return g;
}
// Deliberately independent small dense reference: choose each minimum by linear search.
CuthillMcKeeTrace reference(const Graph& g) {
  const int n=static_cast<int>(g.size());CuthillMcKeeTrace t;
  t.degree.resize(n);t.parent.resize(n);t.level.resize(n);std::vector<bool> seen(n);
  for(int i=0;i<n;++i) t.degree[i]=static_cast<int>(g[i].size());
  const auto less=[&](int a,int b){return std::pair{t.degree[a],a}<std::pair{t.degree[b],b};};
  while(static_cast<int>(t.order.size())<n) {
    int root=-1;for(int v=0;v<n;++v) if(!seen[v]&&(root<0||less(v,root))) root=v;
    t.roots.push_back(root);seen[root]=true;t.parent[root]=root;t.level[root]=0;std::vector<int> q{root};
    for(std::size_t h=0;h<q.size();++h) {
      const int u=q[h];CuthillMcKeeStep step;step.vertex=u;t.order.push_back(u);
      // Dense membership checks, repeated linear minimum, no sorted adjacency/production helper.
      for(;;) {
        int next=-1;
        for(int v=0;v<n;++v) if(!seen[v]&&std::find(g[u].begin(),g[u].end(),v)!=g[u].end()&&(next<0||less(v,next))) next=v;
        if(next<0) break;
        step.newly_enqueued.push_back(next);q.push_back(next);seen[next]=true;t.parent[next]=u;t.level[next]=t.level[u]+1;
      }
      t.adjacency_scans+=g[u].size();t.neighbor_enqueues+=step.newly_enqueued.size();t.steps.push_back(step);
    }
  }
  return t;
}
void compare_reference(const Graph& g) {
  const auto expected=reference(g),t=cuthill_mckee_trace(g);
  EXPECT_EQ(t.order,expected.order);
  EXPECT_EQ(t.order,t.forward_order);
  EXPECT_EQ(t.degree,expected.degree);
  EXPECT_EQ(t.parent,expected.parent);
  EXPECT_EQ(t.level,expected.level);
  EXPECT_EQ(t.roots,expected.roots);
  ASSERT_EQ(t.steps.size(),expected.steps.size());
  for(std::size_t k=0;k<t.steps.size();++k) {EXPECT_EQ(t.steps[k].vertex,expected.steps[k].vertex);
  EXPECT_EQ(t.steps[k].newly_enqueued,expected.steps[k].newly_enqueued);}
  EXPECT_EQ(t.adjacency_scans,expected.adjacency_scans);
  EXPECT_EQ(t.neighbor_enqueues,g.size()-t.roots.size());
  EXPECT_EQ(cuthill_mckee(g),t.order);auto reversed=t.order;std::reverse(reversed.begin(),reversed.end());
  const auto r=cuthill_mckee_trace(g,true);
  EXPECT_EQ(r.order,reversed);
  EXPECT_EQ(r.forward_order,t.order);
  EXPECT_EQ(r.parent,t.parent);
  EXPECT_EQ(cuthill_mckee(g,true),reversed);
  for(std::size_t i=0;i<t.order.size();++i) EXPECT_EQ(t.position[t.order[i]],i);
}
// Independent dense matrix inspection under explicit B_ij=A_order[i],order[j].
void check_metrics(const Graph& g,const std::vector<int>& p) {
  const auto m=symmetric_pattern_metrics(g,p);std::size_t bw=0,lp=0,up=0,edges=0;
  for(std::size_t i=0;i<p.size();++i) {
    std::size_t left=i,right=i;
    for(std::size_t j=0;j<p.size();++j) if(std::find(g[p[i]].begin(),g[p[i]].end(),p[j])!=g[p[i]].end()) {
      left=std::min(left,j);right=std::max(right,j);bw=std::max(bw,i>j?i-j:j-i);edges+=j<i;
    }
    EXPECT_EQ(m.lower_width[i],i-left);
  EXPECT_EQ(m.upper_width[i],right-i);lp+=i-left;up+=right-i;
  }
  EXPECT_EQ(m.bandwidth,bw);
  EXPECT_EQ(m.lower_profile,lp);
  EXPECT_EQ(m.upper_profile,up);
  EXPECT_EQ(m.edges,edges);
  auto rev=p;std::reverse(rev.begin(),rev.end());const auto r=symmetric_pattern_metrics(g,rev);
  EXPECT_EQ(r.bandwidth,m.bandwidth);
  EXPECT_EQ(r.lower_profile,m.upper_profile);
  EXPECT_EQ(r.upper_profile,m.lower_profile);
}
}
TEST(CuthillMcKeeTrace, HandPathAndStarProfiles) {
  const Graph path{{3},{3,4},{4},{0,1},{1,2}};const auto t=cuthill_mckee_trace(path);
  EXPECT_EQ(t.order,(std::vector<int>{0,3,1,4,2}));
  EXPECT_EQ(t.level,(std::vector<int>{0,2,4,1,3}));
  EXPECT_EQ(t.parent,(std::vector<int>{0,3,4,0,1}));
  EXPECT_EQ(t.adjacency_scans,8);
  EXPECT_EQ(t.neighbor_enqueues,4);
  const Graph star{{1,2,3,4},{0},{0},{0},{0}};const auto c=cuthill_mckee(star),r=cuthill_mckee(star,true);
  EXPECT_EQ(c,(std::vector<int>{1,0,2,3,4}));const auto a=symmetric_pattern_metrics(star,c),b=symmetric_pattern_metrics(star,r);
  EXPECT_EQ(a.bandwidth,3);
  EXPECT_EQ(b.bandwidth,3);
  EXPECT_EQ(a.lower_profile,7);
  EXPECT_EQ(a.upper_profile,4);
  EXPECT_EQ(b.lower_profile,4);
  // A legal input order already has smaller bandwidth. CM can worsen it.
  const auto better=permute_symmetric_pattern(star,{1,2,0,3,4});
  EXPECT_EQ(symmetric_pattern_metrics(better,{0,1,2,3,4}).bandwidth,2);
  EXPECT_EQ(symmetric_pattern_metrics(better,cuthill_mckee(better)).bandwidth,3);
}
TEST(CuthillMcKeeTrace, EverySimpleGraphThroughSixVerticesMatchesDenseReference) {
  std::size_t cases=0;
  for(int n=0;n<=6;++n) for(unsigned long long mask=0;mask<(1ULL<<(n*(n-1)/2));++mask) {compare_reference(from_mask(n,mask));++cases;}
  EXPECT_EQ(cases,33868);
}
TEST(CuthillMcKeeTrace, EveryOrderingThroughFiveVerticesHasDenseMetricsAndReverseIdentity) {
  std::size_t cases=0;
  for(int n=0;n<=5;++n) for(unsigned long long mask=0;mask<(1ULL<<(n*(n-1)/2));++mask) {
    const auto g=from_mask(n,mask);std::vector<int> p(n);std::iota(p.begin(),p.end(),0);
    do {check_metrics(g,p);++cases;}while(std::next_permutation(p.begin(),p.end()));
  }
  EXPECT_EQ(cases,124470);
}
TEST(CuthillMcKeeTrace, DuplicateLoopNormalizationAndInputOrderDoNotChangeDegreesOrTraversal) {
  std::mt19937 rng(71);
  for(int trial=0;trial<1000;++trial) {
    const int n=1+static_cast<int>(rng()%30);Graph g(n);
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) if(rng()%5==0) {g[i].push_back(j);g[j].push_back(i);}
    auto raw=g;
    for(int i=0;i<n;++i) {const auto row=raw[i];raw[i].insert(raw[i].end(),row.begin(),row.end());raw[i].push_back(i);raw[i].push_back(i);std::shuffle(raw[i].begin(),raw[i].end(),rng);}
    const auto a=cuthill_mckee_trace(g),b=cuthill_mckee_trace(raw);
    EXPECT_EQ(a.order,b.order);
  EXPECT_EQ(a.degree,b.degree);
  EXPECT_EQ(a.adjacency_scans,b.adjacency_scans);compare_reference(g);
    EXPECT_EQ(permute_symmetric_pattern(raw,a.order),permute_symmetric_pattern(g,a.order));
  }
  EXPECT_EQ(cuthill_mckee({{1,1,0},{0,0,1}}),(std::vector<int>{0,1}));
}
TEST(CuthillMcKeeTrace, DisconnectedRootsAndOwnedHistory) {
  Graph g{{},{2},{1},{4,5},{3,5},{3,4},{}};const auto t=cuthill_mckee_trace(g,true);
  EXPECT_EQ(t.roots,(std::vector<int>{0,6,1,3}));
  EXPECT_EQ(t.forward_order,(std::vector<int>{0,6,1,2,3,4,5}));
  EXPECT_EQ(t.level,(std::vector<int>{0,0,1,0,1,1,0}));
  g.clear();
  EXPECT_EQ(t.order.size(),7);
  EXPECT_EQ(t.steps.size(),7);
}
TEST(CuthillMcKeeTrace, InvalidEndpointsSymmetryPermutationsAndCompleteCaps) {
  EXPECT_THROW(cuthill_mckee({{-1}}),std::invalid_argument);
  EXPECT_THROW(cuthill_mckee({{1},{}}),std::invalid_argument);
  EXPECT_THROW(cuthill_mckee({{2},{0}}),std::invalid_argument);
  EXPECT_THROW(cuthill_mckee_trace({{1},{0}},false,1,0),std::length_error);
  EXPECT_THROW(cuthill_mckee_trace({{1,1},{0}},false,0,2),std::length_error);
  EXPECT_NO_THROW(cuthill_mckee_trace({{1,1},{0}},false,0,0));
  // Full endpoint scan precedes cap rejection even for a later row.
  EXPECT_THROW(cuthill_mckee_trace({{1},{0},{-1}},false,1,1),std::invalid_argument);
  EXPECT_THROW(cuthill_mckee_trace({{1},{}},false,0,0),std::invalid_argument);
  for(const auto& p:std::vector<std::vector<int>>{{0,0},{0,2},{0,-1},{0}}) {
    EXPECT_THROW(symmetric_pattern_metrics({{1},{0}},p),std::invalid_argument);
    EXPECT_THROW(permute_symmetric_pattern({{1},{0}},p),std::invalid_argument);
    EXPECT_THROW(symmetric_pattern_coordinates({{1},{0}},p),std::invalid_argument);
  }
  EXPECT_THROW(inverse_ordering({1,1}),std::invalid_argument);
  EXPECT_THROW(inverse_ordering({-1}),std::invalid_argument);
  EXPECT_TRUE(cuthill_mckee({}).empty());
  EXPECT_TRUE(cuthill_mckee_trace({}).steps.empty());
  EXPECT_EQ(cuthill_mckee({{0,0}}),(std::vector<int>{0}));
}
TEST(CuthillMcKeeTrace, PermutationRoundTripAndCoordinatesMatchDenseMatrix) {
  std::mt19937 rng(8);
  for(int trial=0;trial<500;++trial) {
    const int n=1+static_cast<int>(rng()%20);Graph g(n);
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) if(rng()%4==0) {g[i].push_back(j);g[j].push_back(i);}
    std::vector<int> p(n);std::iota(p.begin(),p.end(),0);std::shuffle(p.begin(),p.end(),rng);
    const auto perm=permute_symmetric_pattern(g,p);
  EXPECT_EQ(permute_symmetric_pattern(perm,inverse_ordering(p)),g);
    for(bool diagonal:{false,true}) {
      const auto coords=symmetric_pattern_coordinates(g,p,diagonal);std::vector<std::vector<int>> actual(n,std::vector<int>(n));
      ASSERT_EQ(coords.row.size(),coords.column.size());
      for(std::size_t k=0;k<coords.row.size();++k) {
        if(k) EXPECT_LT((std::pair{coords.row[k-1],coords.column[k-1]}),(std::pair{coords.row[k],coords.column[k]}));
        ++actual[static_cast<int>(coords.row[k])][static_cast<int>(coords.column[k])];
      }
      for(int i=0;i<n;++i) for(int j=0;j<n;++j) EXPECT_EQ(actual[i][j],(i==j?diagonal:std::find(g[p[i]].begin(),g[p[i]].end(),p[j])!=g[p[i]].end()));
    }
    check_metrics(g,p);
  }
}
TEST(CuthillMcKeeTrace, BandedPatternsDegeneraciesCapsAndOriginalFixture) {
  EXPECT_TRUE(symmetric_band_pattern(0,100).empty());
  EXPECT_EQ(symmetric_band_pattern(1,100),Graph(1));
  EXPECT_EQ(symmetric_band_pattern(3,0),Graph(3));
  EXPECT_EQ(symmetric_band_pattern(3,100),(Graph{{1,2},{0,2},{0,1}}));
  EXPECT_THROW(symmetric_band_pattern(static_cast<std::size_t>(INT_MAX)+1,0),std::length_error);
  EXPECT_THROW(symmetric_band_pattern(INT_MAX,INT_MAX),std::length_error);
  EXPECT_EQ(symmetric_band_pattern(3,std::numeric_limits<std::size_t>::max()),(Graph{{1,2},{0,2},{0,1}}));
  EXPECT_THROW(symmetric_band_pattern(4,2,9),std::length_error);
  EXPECT_NO_THROW(symmetric_band_pattern(4,2,10));
  EXPECT_NO_THROW(symmetric_band_pattern(4,2,0));
  for(int n=0;n<50;++n) for(int w=0;w<55;++w) {
    const auto g=symmetric_band_pattern(n,w,0);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) EXPECT_EQ(std::binary_search(g[i].begin(),g[i].end(),j),i!=j&&std::abs(i-j)<=w);
  }
  std::mt19937 rng(7);std::vector<int> label(40);std::iota(label.begin(),label.end(),0);std::shuffle(label.begin(),label.end(),rng);
  const auto g=permute_symmetric_pattern(symmetric_band_pattern(40,2),inverse_ordering(label));
  std::vector<int> identity(40);std::iota(identity.begin(),identity.end(),0);
  check_metrics(g,identity);
  EXPECT_EQ(symmetric_pattern_metrics(g,cuthill_mckee(g,true)).bandwidth,2);
}
TEST(CuthillMcKeeTrace, LargeDisconnectedForestDoesNotRepeatGlobalRootSearch) {
  Graph g(100000);const auto t=cuthill_mckee_trace(g,false,100000,1);
  EXPECT_EQ(t.roots.size(),g.size());
  EXPECT_EQ(t.order.front(),0);
  EXPECT_EQ(t.order.back(),99999);
  EXPECT_EQ(t.adjacency_scans,0);
}
