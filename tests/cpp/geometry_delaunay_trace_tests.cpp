#include <gtest/gtest.h>
#include <datamunge/geometry/delaunay.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::geometry;
namespace {
using Wide=__int128;
using Face=std::array<std::size_t,3>;
using Edge=std::pair<std::size_t,std::size_t>;
Wide orient(Point2D a,Point2D b,Point2D c) {const auto x=Wide(b.x)-Wide(a.x),y=Wide(b.y)-Wide(a.y);return x*(Wide(c.y)-Wide(a.y))-y*(Wide(c.x)-Wide(a.x));}
Wide circle(Point2D a,Point2D b,Point2D c,Point2D d) {
  const auto ax=Wide(a.x)-Wide(d.x),ay=Wide(a.y)-Wide(d.y),bx=Wide(b.x)-Wide(d.x),by=Wide(b.y)-Wide(d.y),cx=Wide(c.x)-Wide(d.x),cy=Wide(c.y)-Wide(d.y);
  return (ax*ax+ay*ay)*(bx*cy-cx*by)-(bx*bx+by*by)*(ax*cy-cx*ay)+(cx*cx+cy*cy)*(ax*by-bx*ay);
}
Face face(Triangle t) {return {t.a,t.b,t.c};}
std::set<Face> faces(const std::vector<Triangle>& f) {std::set<Face> out;for(auto t:f)out.insert(face(t));return out;}
bool between(Point2D a,Point2D b,Point2D q) {return q!=a&&q!=b&&q.x>=std::min(a.x,b.x)&&q.x<=std::max(a.x,b.x)&&q.y>=std::min(a.y,b.y)&&q.y<=std::max(a.y,b.y);}
void inspect(const std::vector<Point2D>& p) {
  const auto trace=delaunay_trace(p);const auto ordinary=delaunay_triangulation(p);EXPECT_EQ(faces(ordinary),faces(trace.triangles));
  std::vector<std::size_t> ids;for(std::size_t i=0;i<p.size();++i) {std::size_t first=0;while(first<i && p[first]!=p[i])++first;EXPECT_EQ(trace.representative_for_input[i],first);if(first==i)ids.push_back(i);}
  EXPECT_EQ(trace.representatives,ids);
  std::vector<std::size_t> sorted=ids;std::sort(sorted.begin(),sorted.end(),[&](auto a,auto b){return std::pair{p[a].x,p[a].y}<std::pair{p[b].x,p[b].y};});
  std::vector<std::size_t> hull;
  for(auto i:sorted) {while(hull.size()>1 && orient(p[hull[hull.size()-2]],p[hull.back()],p[i])<=0)hull.pop_back();hull.push_back(i);}
  const auto lower=hull.size();for(auto it=sorted.rbegin();it!=sorted.rend();++it) {const auto i=*it;if(it==sorted.rbegin())continue;while(hull.size()>lower && orient(p[hull[hull.size()-2]],p[hull.back()],p[i])<=0)hull.pop_back();hull.push_back(i);}
  if(hull.size()>1)hull.pop_back();
  if(hull.size()<3) {EXPECT_TRUE(ordinary.empty());EXPECT_FALSE(trace.has_seed);return;}
  ASSERT_TRUE(trace.has_seed);ASSERT_FALSE(ordinary.empty());std::set<std::size_t> used;std::map<Edge,unsigned> edges;Wide area=0,hullarea=0;
  for(const auto t:ordinary) {
    ASSERT_LT(t.a,p.size());ASSERT_LT(t.b,p.size());ASSERT_LT(t.c,p.size());ASSERT_GT(orient(p[t.a],p[t.b],p[t.c]),0);EXPECT_LT(t.a,t.b);EXPECT_LT(t.a,t.c);area+=orient(p[t.a],p[t.b],p[t.c]);
    for(auto i:{t.a,t.b,t.c})used.insert(i);for(auto e:{Edge{t.a,t.b},Edge{t.b,t.c},Edge{t.c,t.a}})++edges[std::minmax(e.first,e.second)];
    for(auto i:ids)if(i!=t.a&&i!=t.b&&i!=t.c)EXPECT_LE(circle(p[t.a],p[t.b],p[t.c],p[i]),0);
  }
  std::size_t boundary=0;for(const auto& [e,c]:edges) {EXPECT_LE(c,2);boundary+=c==1;}
  std::size_t expected_boundary=0;
  for(auto i:ids)for(std::size_t j=0;j<hull.size();++j) {const auto a=p[hull[j]],b=p[hull[(j+1)%hull.size()]];if(p[i]==a || (orient(a,b,p[i])==0&&between(a,b,p[i]))) {++expected_boundary;break;}}
  EXPECT_EQ(boundary,expected_boundary);EXPECT_EQ(used.size(),ids.size());EXPECT_EQ(ordinary.size(),2*ids.size()-2-boundary);EXPECT_EQ(edges.size(),3*ids.size()-3-boundary);
  for(std::size_t j=0;j<hull.size();++j) {const auto a=p[hull[j]],b=p[hull[(j+1)%hull.size()]];hullarea+=Wide(a.x)*Wide(b.y)-Wide(a.y)*Wide(b.x);}
  EXPECT_TRUE(area==hullarea);
  for(const auto& [a,ac]:edges)for(const auto& [b,bc]:edges)if(a.first!=b.first&&a.first!=b.second&&a.second!=b.first&&a.second!=b.second) {
    const auto o1=orient(p[a.first],p[a.second],p[b.first]),o2=orient(p[a.first],p[a.second],p[b.second]),o3=orient(p[b.first],p[b.second],p[a.first]),o4=orient(p[b.first],p[b.second],p[a.second]);
    EXPECT_FALSE(((o1>0&&o2<0)||(o1<0&&o2>0))&&((o3>0&&o4<0)||(o3<0&&o4>0)));
  }
  const auto a=delaunay_analyze(p,ordinary);EXPECT_EQ(a.distinct_points,ids.size());EXPECT_EQ(a.used_points,ids.size());EXPECT_EQ(a.edges,edges.size());EXPECT_EQ(a.boundary_edges,boundary);EXPECT_EQ(a.interior_violations,0);EXPECT_EQ(a.circle_tests,ordinary.size()*(ids.size()-3));
  EXPECT_EQ(trace.work.insertions,ids.size()-3);EXPECT_EQ(trace.work.incircle_tests+trace.work.hull_side_tests,ids.size()*(ids.size()-3));
}
}
TEST(DelaunayTrace, OriginalFanAndActualCenterCavity) {
  const std::vector<Point2D> p{{0,0},{4,0},{4,4},{0,4},{2,2}};inspect(p);const auto t=delaunay_trace(p);ASSERT_EQ(t.steps.size(),2);EXPECT_EQ(t.triangles.size(),4);
  EXPECT_EQ(t.steps.back().input_index,4);EXPECT_EQ(t.steps.back().removed.size(),2);EXPECT_EQ(t.steps.back().boundary.size(),4);EXPECT_EQ(t.steps.back().added.size(),4);
}
TEST(DelaunayTrace, EverySubsetOfThreeByThreeIntegerGrid) {
  const std::vector<Point2D> grid{{-1,-1},{0,-1},{1,-1},{-1,0},{0,0},{1,0},{-1,1},{0,1},{1,1}};
  for(unsigned mask=0;mask<512;++mask) {std::vector<Point2D> p;for(unsigned i=0;i<9;++i)if(mask&(1u<<i))p.push_back(grid[i]);inspect(p);std::reverse(p.begin(),p.end());inspect(p);}
}
TEST(DelaunayTrace, SeededIntegerMeshesAndIndependentGeometry) {
  std::mt19937 rng(1981);for(int trial=0;trial<250;++trial) {std::vector<Point2D> p;for(unsigned i=0,n=3+rng()%22;i<n;++i)p.push_back({double(int(rng()%41)-20),double(int(rng()%41)-20)});inspect(p);}
}
TEST(DelaunayTrace, ActualEventsReplayAndCounts) {
  const auto p=delaunay_uniform_cloud(40);const auto t=delaunay_trace(p);const auto s=t.seed;const auto inf=p.size();std::vector<Triangle> current{s,{s.b,s.a,inf},{s.c,s.b,inf},{s.a,s.c,inf}};
  std::size_t removed=0,added=0,boundary=0,incircle=0,ghosts=0;
  for(const auto& e:t.steps) {
    EXPECT_EQ(current.size(),e.faces_before);std::map<Edge,std::pair<Edge,unsigned>> edge_counts;std::vector<Triangle> expected_removed,next;
    for(auto f:current) {
      bool bad;if(f.c==inf) {++ghosts;const int o=orientation_sign(p[f.a],p[f.b],p[e.input_index]);bad=o>0 || (!o&&between(p[f.a],p[f.b],p[e.input_index]));}else {++incircle;bad=incircle_sign(p[f.a],p[f.b],p[f.c],p[e.input_index])>0;}
      if(!bad) {next.push_back(f);continue;}expected_removed.push_back(f);
      for(auto edge:{Edge{f.a,f.b},Edge{f.b,f.c},Edge{f.c,f.a}}) {const auto [it,inserted]=edge_counts.emplace(std::minmax(edge.first,edge.second),std::make_pair(edge,1u));if(!inserted)++it->second.second;}
    }
    EXPECT_EQ(faces(expected_removed),faces(e.removed));std::vector<Edge> expected_edges;for(const auto& [key,value]:edge_counts)if(value.second==1)expected_edges.push_back(value.first);EXPECT_EQ(e.boundary,expected_edges);
    next.insert(next.end(),e.added.begin(),e.added.end());EXPECT_EQ(next.size(),e.faces_after);EXPECT_EQ(e.faces_after,e.faces_before+2);current=next;removed+=e.removed.size();added+=e.added.size();boundary+=e.boundary.size();
  }
  EXPECT_EQ(t.work.removed_faces,removed);EXPECT_EQ(t.work.created_faces,added);EXPECT_EQ(t.work.boundary_edges,boundary);EXPECT_EQ(t.work.incircle_tests,incircle);EXPECT_EQ(t.work.hull_side_tests,ghosts);
}
TEST(DelaunayTrace, DuplicatesCollinearBoundaryAndOwnedEarliestIds) {
  inspect({{0,0},{1,0},{2,0},{3,0},{0,2},{3,2},{2,2},{1,2},{0,1},{3,1},{1,1},{2,1}});
  inspect({{0,0},{1,0},{2,0},{3,0}});inspect({{0,0},{0,0},{-0.,0},{1,0},{0,1},{1,0}});
  std::vector<Point2D> p{{0,0},{1,0},{0,1},{0,0}};const auto t=delaunay_trace(p);p[0]={9,9};EXPECT_EQ(t.points[0],(Point2D{0,0}));EXPECT_EQ(t.representative_for_input.back(),0);
}
TEST(DelaunayTrace, CocircularPermutationsAndFullBinary64Range) {
  std::vector<Point2D> square{{0,0},{1,0},{1,1},{0,1}};std::array<unsigned,4> order{0,1,2,3};do {std::vector<Point2D> p;for(auto i:order)p.push_back(square[i]);inspect(p);}while(std::next_permutation(order.begin(),order.end()));
  for(double s:{std::numeric_limits<double>::denorm_min(),std::ldexp(1.,-500),1.,std::ldexp(1.,500),std::numeric_limits<double>::max()}) {
    EXPECT_EQ(incircle_sign({-s,0},{0,-s},{s,0},{0,0}),1);EXPECT_EQ(incircle_sign({-s,0},{0,-s},{s,0},{0,s}),0);EXPECT_EQ(incircle_sign({-s,0},{0,-s},{s,0},{s,s}),-1);
    const std::vector<Point2D> p{{-s,0},{0,-s},{s,0},{0,s},{0,0}};const auto f=delaunay_triangulation(p);EXPECT_EQ(f.size(),4);EXPECT_EQ(delaunay_analyze(p,f).interior_violations,0);
  }
  const auto big=std::numeric_limits<double>::max(),tiny=std::numeric_limits<double>::denorm_min();const std::vector<Point2D> mixed{{0,0},{big,0},{0,big},{tiny,tiny}};const auto f=delaunay_triangulation(mixed);EXPECT_EQ(f.size(),3);EXPECT_EQ(delaunay_analyze(mixed,f).interior_violations,0);
}
TEST(DelaunayTrace, NarrowTriangleAndNoAngleGuarantee) {
  const std::vector<Point2D> p{{0,0},{1,0},{.5,.000001}};const auto f=delaunay_triangulation(p);ASSERT_EQ(f.size(),1);const auto a=delaunay_analyze(p,f);EXPECT_GT(a.minimum_angle_degrees,0);EXPECT_LT(a.minimum_angle_degrees,.001);
}
TEST(DelaunayTrace, CompleteCapsValidationAndMalformedAnalysis) {
  const std::vector<Point2D> p{{0,0},{4,0},{4,4},{0,4},{2,2}};const auto t=delaunay_trace(p);std::size_t entries=0;for(const auto& e:t.steps)entries+=1+e.removed.size()+e.boundary.size()+e.added.size();
  EXPECT_NO_THROW(delaunay_trace(p,5,10,entries));
  EXPECT_THROW(delaunay_trace(p,4,10,entries),std::length_error);
  EXPECT_THROW(delaunay_trace(p,5,9,entries),std::length_error);
  EXPECT_THROW(delaunay_trace(p,5,10,entries-1),std::length_error);
  EXPECT_NO_THROW(delaunay_trace(p,0,0,0));
  EXPECT_NO_THROW(delaunay_analyze(p,t.triangles,8));
  EXPECT_THROW(delaunay_analyze(p,t.triangles,7),std::length_error);
  EXPECT_THROW(delaunay_analyze(p,{{0,1,5}},1),std::invalid_argument);
  EXPECT_THROW(delaunay_analyze(p,{{0,1,1}},1),std::invalid_argument);
  EXPECT_THROW(delaunay_analyze(p,{{0,4,1}},1),std::invalid_argument);
  EXPECT_THROW(delaunay_analyze(p,{t.triangles[0],t.triangles[0]},1),std::invalid_argument);
  EXPECT_THROW(delaunay_analyze(p,{{0,1,2},{0,1,3},{0,1,4}},1),std::invalid_argument);
  const auto violated=delaunay_analyze({{0,0},{4,0},{0,4},{1,1}},{{0,1,2}});
  EXPECT_EQ(violated.circle_tests,1);EXPECT_EQ(violated.interior_violations,1);EXPECT_EQ(violated.used_points,3);
  const auto nan=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)delaunay_triangulation({{nan,0}}),std::invalid_argument);
  EXPECT_THROW(delaunay_trace({{nan,0},{1,1}},1,1,1),std::invalid_argument);
  EXPECT_THROW(incircle_sign({0,0},{1,0},{0,1},{nan,0}),std::invalid_argument);
  EXPECT_TRUE(std::isnan(delaunay_analyze({},{}).minimum_angle_degrees));
}
TEST(DelaunayTrace, OriginalSeededCloudAndGrowthLaw) {
  std::mt19937 rng(20250728);std::uniform_real_distribution<double> u(0,10);const auto p=delaunay_uniform_cloud(30);for(const auto& x:p) {EXPECT_DOUBLE_EQ(x.x,u(rng));EXPECT_DOUBLE_EQ(x.y,u(rng));}
  for(std::size_t n:{8,16,32,64,128}) {const auto t=delaunay_trace(delaunay_uniform_cloud(n,1981));const auto a=delaunay_analyze(t.points,t.triangles);EXPECT_EQ(a.used_points,n);EXPECT_EQ(t.triangles.size(),2*n-2-a.boundary_edges);EXPECT_EQ(t.work.incircle_tests+t.work.hull_side_tests,n*(n-3));EXPECT_EQ(a.interior_violations,0);}
  EXPECT_THROW(delaunay_uniform_cloud(2,1,1,1),std::length_error);
  EXPECT_THROW(delaunay_uniform_cloud(0,1,0),std::invalid_argument);
  EXPECT_THROW(delaunay_uniform_cloud(0,1,std::numeric_limits<double>::infinity()),std::invalid_argument);
}
