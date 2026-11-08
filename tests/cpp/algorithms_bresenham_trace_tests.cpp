#include <datamunge/algorithms/bresenham.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <random>
namespace ra=datamunge::algorithms;
namespace {
// Independent rational nearest-centre oracle, rounding half steps towards the destination.
std::vector<ra::GridPoint> rational_centres(int x0,int y0,int x1,int y1) {
    const auto dx=std::abs(static_cast<std::int64_t>(x1)-x0);
    const auto dy=std::abs(static_cast<std::int64_t>(y1)-y0);
    const bool horizontal=dx>=dy;
    const auto major=std::max(dx,dy),minor=std::min(dx,dy);
    const int sx=x1>x0?1:-1,sy=y1>y0?1:-1;
    if (!major) return {{x0,y0}};
    std::vector<ra::GridPoint> points;
    for (std::int64_t k=0;k<=major;++k) {
        const auto offset=(2*k*minor+major)/(2*major);
        points.emplace_back(static_cast<int>(x0+sx*(horizontal?k:offset)),
                            static_cast<int>(y0+sy*(horizontal?offset:k)));
    }
    return points;
}
void check_segment(int x0,int y0,int x1,int y1) {
    const auto expected=rational_centres(x0,y0,x1,y1);
    const auto ordinary=ra::bresenham_line(x0,y0,x1,y1);
    const auto trace=ra::bresenham_line_trace(x0,y0,x1,y1);
    EXPECT_EQ(ordinary,expected);
    ASSERT_EQ(trace.points,expected);ASSERT_EQ(trace.steps.size(),expected.size());
    EXPECT_EQ(ra::bresenham_point_count(x0,y0,x1,y1),expected.size());
    const auto dx=std::abs(static_cast<std::int64_t>(x1)-x0);
    const auto dy=std::abs(static_cast<std::int64_t>(y1)-y0);
    EXPECT_EQ(trace.dx,dx);EXPECT_EQ(trace.dy,dy);
    EXPECT_EQ(trace.x_moves,dx);EXPECT_EQ(trace.y_moves,dy);
    EXPECT_EQ(trace.diagonal_moves,std::min(dx,dy));
    for (std::size_t i=0;i<trace.steps.size();++i) {
        const auto& row=trace.steps[i];
        const auto nx=std::abs(static_cast<std::int64_t>(row.point.first)-x0);
        const auto ny=std::abs(static_cast<std::int64_t>(row.point.second)-y0);
        EXPECT_EQ(row.error,dx-dy-nx*dy+ny*dx);
        EXPECT_EQ(row.doubled_error,2*row.error);
        EXPECT_EQ(row.terminal,i+1==trace.steps.size());
        EXPECT_EQ(row.step_x,!row.terminal && row.doubled_error>=-dy);
        EXPECT_EQ(row.step_y,!row.terminal && row.doubled_error<=dx);
        if (!row.terminal) {
            const auto& next=trace.steps[i+1].point;
            EXPECT_EQ(next.first-row.point.first,row.step_x?(x0<x1?1:-1):0);
            EXPECT_EQ(next.second-row.point.second,row.step_y?(y0<y1?1:-1):0);
            EXPECT_TRUE(row.step_x || row.step_y);
        }
    }
}
}
TEST(BresenhamTrace, ExhaustiveSmallEndpointsMatchExactRationalNearestCentres) {
    std::size_t segments=0;
    for (int x0=-5;x0<=5;++x0) for (int y0=-5;y0<=5;++y0)
      for (int x1=-5;x1<=5;++x1) for (int y1=-5;y1<=5;++y1) {
        check_segment(x0,y0,x1,y1);++segments;
      }
    EXPECT_EQ(segments,14641u);
}
TEST(BresenhamTrace, OriginalExampleRetainsEveryCentre) {
    EXPECT_EQ(ra::bresenham_line(0,0,14,6),(std::vector<ra::GridPoint>{{0,0},{1,0},{2,1},{3,1},{4,2},
      {5,2},{6,3},{7,3},{8,3},{9,4},{10,4},{11,5},{12,5},{13,6},{14,6}}));
    EXPECT_EQ(ra::bresenham_line(0,0,3,10).size(),11u);
    EXPECT_EQ(ra::bresenham_line(0,0,7,7).size(),8u);
}
TEST(BresenhamTrace, HalfwayTiesAreDirectedAndBothTestsUseTheSavedError) {
    const auto forward=ra::bresenham_line(0,0,2,1);
    EXPECT_EQ(forward,(std::vector<ra::GridPoint>{{0,0},{1,1},{2,1}}));
    auto reverse=ra::bresenham_line(2,1,0,0);std::reverse(reverse.begin(),reverse.end());
    EXPECT_EQ(reverse,(std::vector<ra::GridPoint>{{0,0},{1,0},{2,1}}));
    EXPECT_NE(forward,reverse);
    const auto trace=ra::bresenham_line_trace(0,0,2,1);
    EXPECT_EQ(trace.steps[0].doubled_error,trace.dx);
    EXPECT_TRUE(trace.steps[0].step_x);EXPECT_TRUE(trace.steps[0].step_y);
    // Re-testing y against the error after the x update would select the wrong next row elsewhere.
    check_segment(0,0,5,2);check_segment(0,0,2,5);
}
TEST(BresenhamTrace, UntiedReversalAndAxisExchangeAgree) {
    for (int dx=1;dx<=20;++dx) for (int dy=0;dy<=20;++dy) {
        const auto p=ra::bresenham_line(-3,4,dx-3,dy+4);
        auto backwards=ra::bresenham_line(dx-3,dy+4,-3,4);std::reverse(backwards.begin(),backwards.end());
        const int major=std::max(dx,dy),minor=std::min(dx,dy);
        bool has_tie=false;
        for (int k=0;k<=major;++k) has_tie|=(2*k*minor)% (2*major)==major;
        if (!has_tie) EXPECT_EQ(p,backwards);
        auto swapped=ra::bresenham_line(4,-3,dy+4,dx-3);
        for (auto& point:swapped) std::swap(point.first,point.second);
        EXPECT_EQ(p,swapped);
    }
}
TEST(BresenhamTrace, CoincidentAxisAndSignedOctantControls) {
    for (const auto& end:std::vector<ra::GridPoint>{{5,2},{2,5},{-2,5},{-5,2},{-5,-2},{-2,-5},{2,-5},{5,-2},{0,0},{5,0},{0,-5}})
        check_segment(0,0,end.first,end.second);
    const auto point=ra::bresenham_line_trace(3,-4,3,-4);
    ASSERT_EQ(point.steps.size(),1u);EXPECT_TRUE(point.steps[0].terminal);
    EXPECT_FALSE(point.steps[0].step_x);EXPECT_FALSE(point.steps[0].step_y);
    EXPECT_EQ(point.x_moves,0u);EXPECT_EQ(point.y_moves,0u);
}
TEST(BresenhamTrace, NarrowSegmentsAtIntegerExtremesAvoidCoordinateOverflow) {
    const int lo=std::numeric_limits<int>::min(),hi=std::numeric_limits<int>::max();
    for (int dx=-8;dx<=8;++dx) for (int dy=-8;dy<=8;++dy) {
        const int x=dx<0?hi:lo,y=dy<0?hi:lo;
        check_segment(x,y,x+dx,y+dy);
        check_segment(x+dx,y+dy,x,y);
    }
}
TEST(BresenhamTrace, FullCoordinateSpanCountAndPositiveLimitAreCheckedBeforeAllocation) {
    const int lo=std::numeric_limits<int>::min(),hi=std::numeric_limits<int>::max();
    const auto full=static_cast<std::uint64_t>(static_cast<std::int64_t>(hi)-lo)+1;
    EXPECT_EQ(ra::bresenham_point_count(lo,lo,hi,hi),full);
    EXPECT_EQ(ra::bresenham_point_count(lo,hi,hi,lo),full);
    EXPECT_EQ(ra::bresenham_point_count(hi,0,lo,0),full);
    EXPECT_THROW(ra::bresenham_line_trace(lo,hi,hi,lo,16),std::length_error);
    EXPECT_THROW(ra::bresenham_line_trace(0,0,14,6,14),std::length_error);
    EXPECT_EQ(ra::bresenham_line_trace(0,0,14,6,15).points.size(),15u);
    EXPECT_EQ(ra::bresenham_line_trace(0,0,14,6,0).points.size(),15u);
}
TEST(BresenhamTrace, RandomModerateSegmentsMatchRationalOracle) {
    std::mt19937 rng(1965);
    std::uniform_int_distribution<int> coordinate(-200,200);
    for (std::size_t i=0;i<1000;++i) {
        const int x0=coordinate(rng),y0=coordinate(rng),x1=coordinate(rng),y1=coordinate(rng);
        check_segment(x0,y0,x1,y1);
    }
}
