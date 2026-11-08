#include <datamunge/geometry/convex_hull.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
using namespace datamunge::geometry;
using Points=std::vector<Point2D>;
std::int64_t determinant(const Point2D& a,const Point2D& b,const Point2D& c) {
    const auto ax=static_cast<std::int64_t>(a.x),ay=static_cast<std::int64_t>(a.y);
    return (static_cast<std::int64_t>(b.x)-ax)*(static_cast<std::int64_t>(c.y)-ay)-
        (static_cast<std::int64_t>(b.y)-ay)*(static_cast<std::int64_t>(c.x)-ax);
}
Points unique(Points p) {
    std::sort(p.begin(),p.end(),[](auto a,auto b){return a.x<b.x||(a.x==b.x&&a.y<b.y);});
    p.erase(std::unique(p.begin(),p.end()),p.end());return p;
}
// Independent supporting-line edge enumeration, with integer segment bounds.
// No monotone chains, stack scans, angular sort or production predicate.
Points supporting_edges(const Points& input) {
    const auto p=unique(input);if(p.size()<2)return p;
    std::map<std::size_t,std::size_t> next;
    for(std::size_t i=0;i<p.size();++i)for(std::size_t j=0;j<p.size();++j){
        if(i==j)continue;bool valid=true;
        const auto dx=static_cast<std::int64_t>(p[j].x-p[i].x),dy=static_cast<std::int64_t>(p[j].y-p[i].y);
        const auto d2=dx*dx+dy*dy;
        for(const auto& q:p){
            const auto s=determinant(p[i],p[j],q);if(s<0){valid=false;break;}
            if(!s){const auto dot=static_cast<std::int64_t>(q.x-p[i].x)*dx+static_cast<std::int64_t>(q.y-p[i].y)*dy;
                if(dot<0||dot>d2){valid=false;break;}}
        }
        if(valid)next[i]=j;
    }
    Points hull;std::size_t at=0;
    do{hull.push_back(p[at]);auto it=next.find(at);if(it==next.end())return {};at=it->second;}while(at&&hull.size()<=p.size());
    return hull;
}
Points former_hull(Points p) {
    p=unique(std::move(p));if(p.size()<3)return p;
    Points h(2*p.size());std::size_t k=0;
    for(const auto& v:p){while(k>=2&&cross(h[k-2],h[k-1],v)<=0)--k;h[k++]=v;}
    const auto lower=k+1;
    for(std::size_t step=1;step<p.size();++step){const auto& v=p[p.size()-1-step];while(k>=lower&&cross(h[k-2],h[k-1],v)<=0)--k;h[k++]=v;}
    h.resize(k-1);return h;
}
void verify_trace(const Points& input,const ConvexHullTrace& t) {
    EXPECT_EQ(t.analysis.hull,supporting_edges(input));
    EXPECT_EQ(t.sorted.size(),t.analysis.work.distinct_points);
    for(const auto& p:t.sorted){ASSERT_LT(p.input_index,input.size());EXPECT_EQ(p.point,input[p.input_index]);
        for(std::size_t i=0;i<p.input_index;++i)EXPECT_NE(p.point,input[i]);}
    std::vector<std::size_t> lower,upper;std::size_t pushes=0,pops=0,tests=0,peak=0;
    for(const auto& e:t.events){auto& stack=e.pass==HullPass::lower?lower:upper;ASSERT_LT(e.candidate,t.sorted.size());
        if(e.action==HullAction::test){ASSERT_GE(stack.size(),2);EXPECT_EQ(e.a,stack[stack.size()-2]);EXPECT_EQ(e.b,stack.back());
            const auto sign=determinant(t.sorted[e.a].point,t.sorted[e.b].point,t.sorted[e.candidate].point);
            EXPECT_EQ(e.turn,(sign>0)-(sign<0));++tests;
        }else if(e.action==HullAction::pop){ASSERT_FALSE(stack.empty());EXPECT_EQ(e.point,stack.back());stack.pop_back();++pops;
        }else{EXPECT_EQ(e.point,e.candidate);stack.push_back(e.point);++pushes;peak=std::max(peak,stack.size());}
        EXPECT_EQ(e.stack_size,stack.size());
    }
    EXPECT_EQ(lower,t.lower);EXPECT_EQ(upper,t.upper);EXPECT_EQ(tests,t.analysis.work.orientation_tests);
    EXPECT_EQ(pushes,t.analysis.work.pushes);EXPECT_EQ(pops,t.analysis.work.pops);EXPECT_EQ(peak,t.analysis.work.max_stack);
    for(std::size_t i=0;i<t.analysis.hull.size();++i)EXPECT_EQ(input[t.analysis.input_indices[i]],t.analysis.hull[i]);
    if(t.sorted.size()>=2){EXPECT_EQ(pushes,2*t.sorted.size());EXPECT_EQ(pops,2*t.sorted.size()-t.analysis.hull.size()-2);EXPECT_LE(tests,4*t.sorted.size());}
}
}
TEST(HullTrace, ActualFivePointEditsAndOriginalSquare){
    const Points p{{0,0},{0,2},{1,1},{2,0},{2,2}};const auto t=convex_hull_trace(p);verify_trace(p,t);
    EXPECT_EQ(t.analysis.hull,(Points{{0,0},{2,0},{2,2},{0,2}}));
    EXPECT_EQ(t.lower,(std::vector<std::size_t>{0,3,4}));EXPECT_EQ(t.upper,(std::vector<std::size_t>{4,1,0}));
    EXPECT_EQ(t.analysis.work.pushes,10);EXPECT_EQ(t.analysis.work.pops,4);EXPECT_EQ(t.events.size(),20);
    const Points original{{0,0},{4,0},{4,4},{0,4},{2,2},{1,3},{3,1}};
    EXPECT_EQ(convex_hull(original),(Points{{0,0},{4,0},{4,4},{0,4}}));
}
TEST(HullTrace, AllGridSubsetsMatchIndependentSupportingEdges){
    Points grid;for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)grid.push_back({double(x),double(y)});
    for(unsigned mask=0;mask<512;++mask){Points p;for(unsigned i=0;i<9;++i)if(mask&(1u<<i))p.push_back(grid[i]);
        EXPECT_EQ(convex_hull(p),supporting_edges(p));verify_trace(p,convex_hull_trace(p));}
}
TEST(HullTrace, AllOrderedFourRecordGridTuplesIncludeDuplicates){
    Points grid;for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)grid.push_back({double(x),double(y)});
    for(const auto& a:grid)for(const auto& b:grid)for(const auto& c:grid)for(const auto& d:grid){const Points p{a,b,c,d};EXPECT_EQ(convex_hull(p),supporting_edges(p));}
}
TEST(HullTrace, RandomIntegerCloudsAndActualReplayAgreeWithIndependentOracle){
    std::mt19937 rng(702);std::uniform_int_distribution<int> d(-4,4);
    for(int trial=0;trial<1000;++trial){Points p;for(unsigned i=0,n=rng()%21;i<n;++i)p.push_back({double(d(rng)),double(d(rng))});
        const auto t=convex_hull_trace(p,{0,0});verify_trace(p,t);const auto a=convex_hull_analyze(p,{0,1});
        EXPECT_EQ(a.hull,t.analysis.hull);EXPECT_EQ(a.input_indices,t.analysis.input_indices);
        EXPECT_EQ(a.work.orientation_tests,t.analysis.work.orientation_tests);EXPECT_EQ(a.work.sort_comparisons,t.analysis.work.sort_comparisons);
    }
}
TEST(HullTrace, PermutationsAndDuplicatesKeepEarliestIdentity){
    const Points base{{0,0},{4,0},{4,4},{0,4},{2,2},{1,3},{3,1}};auto order=std::vector<int>{0,1,2,3,4,5,6};
    do{Points p;for(int i:order)p.push_back(base[i]);p.push_back(p.front());const auto a=convex_hull_analyze(p);
        EXPECT_EQ(a.hull,(Points{{0,0},{4,0},{4,4},{0,4}}));
        for(std::size_t k=0;k<a.hull.size();++k){EXPECT_EQ(p[a.input_indices[k]],a.hull[k]);for(std::size_t i=0;i<a.input_indices[k];++i)EXPECT_NE(p[i],a.hull[k]);}
    }while(std::next_permutation(order.begin(),order.end()));
    const auto zero=convex_hull_trace({{-0.,-0.},{0.,0.},{1,0}});EXPECT_EQ(zero.sorted.size(),2);EXPECT_EQ(zero.sorted[0].input_index,0);
    EXPECT_TRUE(std::signbit(zero.sorted[0].point.x));EXPECT_TRUE(std::signbit(zero.analysis.hull[0].y));
}
TEST(HullTrace, DegenerateAndCollinearBoundaryContracts){
    EXPECT_TRUE(convex_hull({}).empty());const auto one=convex_hull_trace({{2,3},{2,3}});
    EXPECT_EQ(one.analysis.hull,(Points{{2,3}}));EXPECT_TRUE(one.events.empty());EXPECT_TRUE(one.lower.empty());
    EXPECT_EQ(convex_hull({{2,2},{1,1},{0,0},{-1,-1},{0,0}}),(Points{{-1,-1},{2,2}}));
    EXPECT_EQ(convex_hull({{0,0},{1,0},{2,0},{2,1},{2,2},{1,2},{0,2},{0,1},{1,1}}),(Points{{0,0},{2,0},{2,2},{0,2}}));
}
TEST(HullTrace, ExactCancellationTrianglePreviouslyCollapsedToASegment){
    const double m=std::ldexp(1.,52);const Point2D a{0,0},b{m,m-1},c{m+1,m};
    EXPECT_EQ(orientation_sign(a,b,c),1);EXPECT_EQ(orientation_sign(a,c,b),-1);
    EXPECT_EQ(convex_hull({a,b,c}),(Points{a,b,c}));
    volatile double left=m*m,right=(m-1)*(m+1);EXPECT_EQ(left-right,0);
    EXPECT_EQ(orientation_sign({0,0},{m,m},{m/2,m/2}),0);
}
TEST(HullTrace, EntireFiniteRangeAndSubnormalsDoNotOverflowOrUnderflowSigns){
    const double m=std::numeric_limits<double>::max(),tiny=std::numeric_limits<double>::denorm_min();
    const Points box{{-m,-m},{m,-m},{m,m},{-m,m},{0,0}};
    EXPECT_EQ(convex_hull(box),(Points{{-m,-m},{m,-m},{m,m},{-m,m}}));
    EXPECT_EQ(convex_hull({{0,0},{tiny,0},{0,tiny},{tiny,tiny}}),(Points{{0,0},{tiny,0},{tiny,tiny},{0,tiny}}));
    EXPECT_EQ(orientation_sign({0,0},{tiny,0},{0,tiny}),1);
    EXPECT_EQ(orientation_sign({m,m},{-m,-m},{tiny,tiny}),0);
    EXPECT_EQ(orientation_sign({m,0},{0,tiny},{-m,0}),1);
    EXPECT_EQ(orientation_sign({m,0},{-m,0},{0,tiny}),-1);
}
TEST(HullTrace, OrientationPermutationAndExactPowerOfTwoSimilarities){
    std::mt19937 rng(813);std::uniform_int_distribution<int> d(-20,20);
    for(int k=0;k<10000;++k){const Point2D a{double(d(rng)),double(d(rng))},b{double(d(rng)),double(d(rng))},c{double(d(rng)),double(d(rng))};
        const auto det=determinant(a,b,c);const int s=(det>0)-(det<0);
        EXPECT_EQ(orientation_sign(a,b,c),s);EXPECT_EQ(orientation_sign(b,c,a),s);EXPECT_EQ(orientation_sign(a,c,b),-s);
        const auto transform=[](Point2D p){return Point2D{std::ldexp(-p.y,400),std::ldexp(p.x,400)};};
        EXPECT_EQ(orientation_sign(transform(a),transform(b),transform(c)),s);
    }
}
TEST(HullTrace, FullValidationPrecedesPointAndEventCaps){
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    for(double bad:{nan,inf,-inf})for(const Points p:{Points{{bad,0}},Points{{0,bad}},Points{{0,0},{1,1},{bad,2}}}){
        EXPECT_THROW(convex_hull(p),std::invalid_argument);
        EXPECT_THROW(convex_hull_trace(p,{1,1}),std::invalid_argument);
    }
    EXPECT_THROW(orientation_sign({0,0},{1,0},{nan,1}),std::invalid_argument);
    EXPECT_THROW(orientation_sign({0,inf},{1,0},{0,1}),std::invalid_argument);
}
TEST(HullTrace, CompletePointAndActualCompactEventCaps){
    const Points p{{0,0},{1,0}};EXPECT_EQ(convex_hull_trace(p,{2,4}).events.size(),4);
    EXPECT_THROW(convex_hull_trace(p,{1,100}),std::length_error);
    EXPECT_THROW(convex_hull_trace(p,{2,3}),std::length_error);
    EXPECT_EQ(convex_hull_analyze(p,{2,1}).hull.size(),2);
    EXPECT_TRUE(convex_hull_trace({{0,0}},{1,1}).events.empty());
    EXPECT_EQ(convex_hull_trace(p,{0,0}).events.size(),4);
}
TEST(HullTrace, ActualLinearScanIdentitiesForParabolasAndCollinearClouds){
    for(std::size_t n=2;n<=256;++n){Points p,line;
        for(std::size_t i=0;i<n;++i){p.push_back({double(i),double(i*i)});line.push_back({double(i),0});}
        for(const auto& input:{p,line}){const auto a=convex_hull_analyze(input);const auto h=a.hull.size();
            EXPECT_EQ(a.work.pushes,2*n);EXPECT_EQ(a.work.pops,2*n-h-2);EXPECT_LE(a.work.orientation_tests,4*n);EXPECT_LE(a.work.max_stack,n);
        }
        EXPECT_EQ(convex_hull(p).size(),n);EXPECT_EQ(convex_hull(line).size(),2);
    }
}
TEST(HullTrace, OriginalGaussianStreamAndFormerHullMatchExactly){
    std::mt19937 rng(20240728);std::normal_distribution<double> normal(0,1);Points original;
    for(int i=0;i<40;++i)original.push_back({normal(rng),normal(rng)});
    const auto p=sample_gaussian_points();EXPECT_EQ(p,original);EXPECT_EQ(convex_hull(p),former_hull(original));
    // Normal-distribution coordinate mapping, and hence the realised hull size, is library-specific.
    EXPECT_GE(convex_hull(p).size(),3);EXPECT_LE(convex_hull(p).size(),p.size());
    EXPECT_EQ(sample_gaussian_points(),p);EXPECT_NE(sample_gaussian_points(40,20240729),p);
    EXPECT_TRUE(sample_gaussian_points(0).empty());EXPECT_EQ(sample_gaussian_points(4,1,4).size(),4);
    EXPECT_THROW(sample_gaussian_points(4,1,3),std::length_error);
}
TEST(HullTrace, ReturnedStorageIsIndependentOfCallerAndOtherStates){
    Points p{{0,0},{0,2},{1,1},{2,0},{2,2}};auto t=convex_hull_trace(p);auto copy=t;
    p[0].x=99;t.sorted[0].point.x=99;t.analysis.hull[0].x=99;t.lower[0]=99;t.events[0].point=99;
    EXPECT_EQ(copy.sorted[0].point.x,0);EXPECT_EQ(copy.analysis.hull[0].x,0);EXPECT_EQ(copy.lower[0],0);EXPECT_EQ(copy.events[0].point,0);
}
