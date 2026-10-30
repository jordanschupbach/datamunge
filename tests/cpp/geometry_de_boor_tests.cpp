#include <gtest/gtest.h>
#include <datamunge/geometry/de_boor.hpp>
#include <datamunge/fda/bspline.hpp>
#include <datamunge/algorithms/interpolation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

using datamunge::geometry::DeBoorSpline;
using datamunge::geometry::Point2D;

TEST(DeBoorSpline, QuadraticBezierHasExactConstructionTriangle) {
    const DeBoorSpline curve(2,{0,0,0,1,1,1},{{0,0},{2,4},{4,0}});
    const auto trace=curve.trace(.25);
    ASSERT_EQ(trace.steps.size(),6u);
    EXPECT_EQ(trace.span,2u);
    EXPECT_DOUBLE_EQ(trace.point.x,1);
    EXPECT_DOUBLE_EQ(trace.point.y,1.5);
    // Updates descend in j: the right parent must not read an already changed left parent.
    EXPECT_EQ(trace.steps[3].local_index,2u);
    EXPECT_DOUBLE_EQ(trace.steps[3].point.x,2.5);
    EXPECT_DOUBLE_EQ(trace.steps[3].point.y,3);
    EXPECT_DOUBLE_EQ(trace.steps[4].point.x,.5);
    EXPECT_DOUBLE_EQ(trace.steps[4].point.y,1);
    for (std::size_t i=3;i<6;++i) EXPECT_DOUBLE_EQ(trace.steps[i].alpha,.25);
    for (std::size_t i=0;i<3;++i) EXPECT_TRUE(std::isnan(trace.steps[i].alpha));
    EXPECT_DOUBLE_EQ(curve.evaluate(.25).x,trace.point.x);
    const auto table=trace.table();
    EXPECT_EQ(table.nrows(),6u);
    EXPECT_DOUBLE_EQ(table.numeric_column("y").values().back(),1.5);
}

TEST(DeBoorSpline, NonuniformWeightsGiveHandComputedPoint) {
    const DeBoorSpline curve(2,{0,0,0,1,3,3,3},{{0,0},{2,4},{4,0},{6,2}});
    const auto trace=curve.trace(.5);
    EXPECT_NEAR(trace.steps[3].alpha,1.0/6,1e-15);
    EXPECT_DOUBLE_EQ(trace.steps[4].alpha,.5);
    EXPECT_DOUBLE_EQ(trace.steps[5].alpha,.5);
    EXPECT_NEAR(trace.point.x,5.0/3,1e-14);
    EXPECT_NEAR(trace.point.y,8.0/3,1e-14);
}

TEST(DeBoorSpline, ExactEndpointsAndTinyParameterDomainsNeedNoEpsilon) {
    for (const auto interval : {std::pair<double,double>{0,1},{0,1e-15},{-2,-1}}) {
        const auto [a,b]=interval;
        const std::vector<double> knots{a,a,a,b,b,b};
        const std::vector<Point2D> control{{0,0},{2,4},{4,0}};
        const DeBoorSpline curve(2,knots,control);
        const auto points=curve.sample(11);
        EXPECT_EQ(points.front(),control.front());
        EXPECT_EQ(points.back(),control.back());
        EXPECT_EQ(curve.evaluate(b),control.back());
        EXPECT_EQ(datamunge::geometry::de_boor_curve(2,knots,control,2).back(),control.back());
    }
}

TEST(DeBoorSpline, DegreeZeroAndUnclampedEndpointsHaveExplicitSemantics) {
    const DeBoorSpline constant(0,{0,1,2},{{2,3},{5,7}});
    EXPECT_EQ(constant.evaluate(0),Point2D({2,3}));
    EXPECT_EQ(constant.evaluate(1),Point2D({5,7}));
    EXPECT_EQ(constant.evaluate(2),Point2D({5,7}));
    const DeBoorSpline open(2,{-2,-1,0,1,2,3,4},{{0,0},{2,4},{4,0},{6,2}});
    EXPECT_EQ(open.evaluate(0),Point2D({1,2}));
    EXPECT_EQ(open.evaluate(2),Point2D({5,1}));
    EXPECT_EQ(open.trace(1).span,3u);
}

TEST(DeBoorSpline, FullMultiplicityUsesRightValueAndPlotsDisconnectedPieces) {
    const DeBoorSpline jump(1,{0,0,1,1,2,2},{{0,0},{1,1},{2,3},{3,2}});
    EXPECT_NEAR(jump.evaluate(std::nextafter(1.0,0.0)).x,1,1e-15);
    EXPECT_EQ(jump.evaluate(1),Point2D({2,3}));
    EXPECT_EQ(jump.trace(1).span,3u);
    const auto plot=jump.plot_curve(5);
    std::vector<datamunge::graphics::DataSeries> lines;
    for (const auto& series : plot.series())
        if (series.kind==datamunge::graphics::DataSeries::Kind::Line && series.color.b==235) lines.push_back(series);
    ASSERT_EQ(lines.size(),2u);
    EXPECT_DOUBLE_EQ(lines[0].x.back(),1);
    EXPECT_DOUBLE_EQ(lines[0].y.back(),1);
    EXPECT_DOUBLE_EQ(lines[1].x.front(),2);
    EXPECT_DOUBLE_EQ(lines[1].y.front(),3);
}

TEST(DeBoorSpline, DoubleQuadraticKnotHasSharedValueAndDifferentOneSidedTangents) {
    const DeBoorSpline corner(2,{0,0,0,1,1,2,2,2},{{0,0},{1,2},{2,0},{3,2},{4,0}});
    EXPECT_EQ(corner.evaluate(1),Point2D({2,0}));
    const double h=1e-6;
    const auto a=corner.evaluate(1-h),b=corner.evaluate(1+h);
    EXPECT_NEAR((0-a.y)/h,-4,1e-5);
    EXPECT_NEAR((b.y-0)/h,4,1e-5);
}

TEST(DeBoorSpline, LocalPerturbationEqualsItsSingleBasisWeight) {
    const std::vector<double> knots{0,0,0,0,1,2,3,4,5,6,7,7,7,7};
    std::vector<Point2D> control{{0,0},{1,2},{2,-1},{3,3},{4,1},{5,4},{6,0},{7,2},{8,1},{9,0}};
    const DeBoorSpline original(3,knots,control);
    control[4].y+=2;
    const DeBoorSpline changed(3,knots,control);
    const datamunge::fda::BSplineBasis basis(3,knots);
    for (int j=0;j<=140;++j) {
        const double t=j/20.0;
        const auto B=basis.evaluate_dense({t});
        const auto a=original.evaluate(t),b=changed.evaluate(t);
        EXPECT_NEAR(a.x,b.x,1e-14);
        EXPECT_NEAR(b.y-a.y,2*B(0,4),2e-14);
        if (t<=1 || t>=5) EXPECT_DOUBLE_EQ(a.y,b.y);
    }
}

TEST(DeBoorSpline, RandomSplinesAgreeWithIndependentBasisEvaluation) {
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<double> random(-3,3);
    for (int degree=0;degree<=7;++degree) {
        const std::size_t n=static_cast<std::size_t>(degree)+6;
        const auto basis=datamunge::fda::BSplineBasis::open_uniform(degree,n,-1,2);
        std::vector<Point2D> control(n);
        for (auto& p : control) p={random(rng),random(rng)};
        const DeBoorSpline curve(degree,basis.knots(),control);
        for (int i=0;i<=100;++i) {
            const double t=-1+3.0*i/100;
            const auto B=basis.evaluate_dense({t});
            Point2D expected;
            for (std::size_t j=0;j<n;++j) {expected.x+=B(0,j)*control[j].x;expected.y+=B(0,j)*control[j].y;}
            const auto actual=curve.evaluate(t);
            EXPECT_NEAR(actual.x,expected.x,3e-14);
            EXPECT_NEAR(actual.y,expected.y,3e-14);
        }
    }
}

TEST(DeBoorSpline, BezierReductionAndOperationCountsAcrossDegrees) {
    for (int p=1;p<=12;++p) {
        std::vector<double> knots(p+1,0); knots.insert(knots.end(),p+1,1);
        std::vector<Point2D> control;
        std::vector<datamunge::algorithms::PlanarPoint> bezier;
        for (int j=0;j<=p;++j) {
            const double x=static_cast<double>(j)/p;
            control.push_back({x,x*x}); bezier.push_back({x,x*x});
        }
        const DeBoorSpline curve(p,knots,control);
        const auto trace=curve.trace(.375);
        EXPECT_EQ(trace.steps.size()-(p+1),static_cast<std::size_t>(p*(p+1)/2));
        for (int i=0;i<=20;++i) {
            const double t=i/20.0;
            const auto q=curve.evaluate(t);
            const auto r=datamunge::algorithms::de_casteljau(bezier,t);
            EXPECT_NEAR(q.x,t,2e-15);
            EXPECT_NEAR(q.y,t*t+t*(1-t)/p,2e-15);
            EXPECT_NEAR(q.x,r.x,2e-15); EXPECT_NEAR(q.y,r.y,2e-15);
        }
    }
}

TEST(DeBoorSpline, AffineEquivarianceAndConvexHullBounds) {
    const std::vector<double> knots{0,0,0,1,3,3,3};
    const std::vector<Point2D> control{{0,0},{2,4},{4,0},{6,2}};
    std::vector<Point2D> transformed;
    for (auto p : control) transformed.push_back({2*p.x-p.y+7,p.x+3*p.y-2});
    const DeBoorSpline curve(2,knots,control),mapped(2,knots,transformed);
    for (int i=0;i<=30;++i) {
        const double t=i/10.0;
        const auto q=curve.evaluate(t),r=mapped.evaluate(t);
        EXPECT_GE(q.x,0); EXPECT_LE(q.x,6); EXPECT_GE(q.y,0); EXPECT_LE(q.y,4);
        EXPECT_NEAR(r.x,2*q.x-q.y+7,1e-13);
        EXPECT_NEAR(r.y,q.x+3*q.y-2,1e-13);
    }
}

TEST(DeBoorSpline, ExtremeFiniteKnotsAndCoordinatesDoNotOverflowBlends) {
    const double big=std::numeric_limits<double>::max();
    const DeBoorSpline curve(1,{-big,-big,big,big},{{-big,big},{big,-big}});
    EXPECT_EQ(curve.evaluate(0),Point2D({0,0}));
    const auto points=curve.sample(3);
    EXPECT_EQ(points[1],Point2D({0,0}));
    EXPECT_EQ(points.back(),Point2D({big,-big}));
}

TEST(DeBoorSpline, RejectsMalformedInputsAndUndefinedSampling) {
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    EXPECT_THROW(DeBoorSpline(-1,{},{}),std::invalid_argument);
    EXPECT_THROW(DeBoorSpline(2,{0,0,1,1},{{0,0},{1,1}}),std::invalid_argument);
    EXPECT_THROW(DeBoorSpline(1,{0,0,1},{{0,0},{1,1}}),std::invalid_argument);
    EXPECT_THROW(DeBoorSpline(1,{0,1,0,1},{{0,0},{1,1}}),std::invalid_argument);
    EXPECT_THROW(DeBoorSpline(1,{0,0,nan,1},{{0,0},{1,1}}),std::invalid_argument);
    EXPECT_THROW(DeBoorSpline(1,{0,0,inf,inf},{{0,0},{1,1}}),std::invalid_argument);
    EXPECT_THROW(DeBoorSpline(1,{0,0,0,1,1},{{0,0},{1,1},{2,2}}),std::invalid_argument);
    EXPECT_THROW(DeBoorSpline(1,{0,0,1,1},{{nan,0},{1,1}}),std::invalid_argument);
    const DeBoorSpline curve(1,{0,0,1,1},{{0,0},{1,1}});
    for (double t : {-1.0,2.0,nan,inf}) EXPECT_THROW(curve.evaluate(t),std::invalid_argument);
    for (int n : {-1,0,1}) {
        EXPECT_THROW(curve.sample(n),std::invalid_argument);
        EXPECT_THROW(curve.plot_curve(n),std::invalid_argument);
    }
    EXPECT_EQ(curve.sample_table({}).nrows(),0u);
    EXPECT_THROW(curve.sample_table({0,.5,2}),std::invalid_argument);
}
