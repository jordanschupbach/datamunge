#include <datamunge/algorithms/cordic.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
namespace {
using namespace datamunge::algorithms;
constexpr double pi=3.14159265358979323846;
SinCos former(double theta,int iterations) {
    double z=std::remainder(theta,2*pi);bool flip=false;
    if(z>pi/2){z-=pi;flip=true;}else if(z<-pi/2){z+=pi;flip=true;}
    double x=1,y=0,gain=1,scale=1;
    for(int i=0;i<iterations;++i){const double angle=std::atan(scale),d=z>=0?1.:-1.;
        const double nx=x-d*y*scale,ny=y+d*x*scale;x=nx;y=ny;z-=d*angle;
        gain*=1/std::sqrt(1+scale*scale);scale*=.5;}
    SinCos out{y*gain,x*gain};if(flip){out.sin=-out.sin;out.cos=-out.cos;}return out;
}
void same(SinCos a,SinCos b){EXPECT_EQ(a.sin,b.sin);EXPECT_EQ(a.cos,b.cos);}
}
TEST(CordicTrace, FirstThreeActualRotationsAreHandCheckable){
    const auto t=cordic_sincos_trace(1,3);ASSERT_EQ(t.steps.size(),3);
    EXPECT_EQ(t.steps[0].x_before,1);EXPECT_EQ(t.steps[0].y_before,0);EXPECT_EQ(t.steps[0].z_before,1);
    EXPECT_EQ(t.steps[0].direction,1);EXPECT_EQ(t.steps[0].x,1);EXPECT_EQ(t.steps[0].y,1);
    EXPECT_EQ(t.steps[1].direction,1);EXPECT_EQ(t.steps[1].x,.5);EXPECT_EQ(t.steps[1].y,1.5);
    EXPECT_EQ(t.steps[2].direction,-1);EXPECT_EQ(t.steps[2].x,.875);EXPECT_EQ(t.steps[2].y,1.375);
    EXPECT_EQ(t.steps[0].inverse_gain,1/std::sqrt(2.));EXPECT_EQ(t.analysis.work.rotations,3);
    EXPECT_EQ(t.analysis.work.constant_lookups,3);EXPECT_EQ(t.analysis.work.inverse_gain_products,3);
    EXPECT_EQ(t.analysis.work.normalizations,3);EXPECT_EQ(t.analysis.work.final_products,2);
    const auto a=cordic_sincos_analyze(1,3);same(a.result,t.analysis.result);EXPECT_EQ(a.work.normalizations,0);
}
TEST(CordicTrace, FullFormerKernelParityAcrossAnglesAndIterationCounts){
    std::mt19937 rng(1900);std::uniform_real_distribution<double> angle(-100,100);
    for(int trial=0;trial<1000;++trial){const double theta=angle(rng);
        for(int n:{0,1,2,3,12,40,44,64,1075})same(cordic_sincos(theta,n),former(theta,n));}
    for(double theta:{.5,1.,2.5,-1.2})same(cordic_sincos(theta,44),former(theta,44));
}
TEST(CordicTrace, OriginalTwelveNormalizedStatesMatchFormerReportExactly){
    const auto t=cordic_sincos_trace(1,12);double x=1,y=0,z=1,scale=1;
    for(std::size_t i=0;i<t.steps.size();++i){const auto& s=t.steps[i];const double d=z>=0?1.:-1.;
        EXPECT_EQ(s.x_before,x);EXPECT_EQ(s.y_before,y);EXPECT_EQ(s.z_before,z);
        const double nx=x-d*y*scale,ny=y+d*x*scale;x=nx;y=ny;z-=d*std::atan(scale);scale*=.5;
        const double norm=std::hypot(x,y);EXPECT_EQ(s.unit_x,x/norm);EXPECT_EQ(s.unit_y,y/norm);
        EXPECT_EQ(s.x,x);EXPECT_EQ(s.y,y);EXPECT_EQ(s.z,z);}
}
TEST(CordicTrace, ModerateRangeMatchesHostTrigAndIdealResidualBound){
    for(int i=0;i<=10000;++i){const double theta=-10+20.*i/10000;const auto a=cordic_sincos_analyze(theta,44);
        EXPECT_NEAR(a.result.sin,std::sin(theta),2e-13);EXPECT_NEAR(a.result.cos,std::cos(theta),2e-13);
        EXPECT_NEAR(std::hypot(a.result.sin,a.result.cos),1,3e-15);
        EXPECT_LE(std::abs(a.residual),std::atan(std::ldexp(1.,-43))+2e-15);
    }
}
TEST(CordicTrace, RoundedPeriodFoldAndInclusiveBoundaryControls){
    for(double theta:{-100.,-2*pi,-pi,-pi/2,-0.,0.,pi/2,pi,2*pi,100.,1e16,1e100,std::numeric_limits<double>::max()}){
        const auto r=cordic_reduce(theta);EXPECT_EQ(r.principal,std::remainder(theta,2*pi));EXPECT_EQ(r.period,2*pi);
        EXPECT_LE(std::abs(r.reduced),pi/2);EXPECT_EQ(r.flipped,r.principal>pi/2||r.principal<-pi/2);
        const auto a=cordic_sincos_analyze(theta,44);same(a.result,former(theta,44));
        EXPECT_NEAR(a.result.sin,(r.flipped?-1:1)*std::sin(r.reduced),2e-13);
        EXPECT_NEAR(a.result.cos,(r.flipped?-1:1)*std::cos(r.reduced),2e-13);
    }
    EXPECT_FALSE(cordic_reduce(pi/2).flipped);EXPECT_FALSE(cordic_reduce(-pi/2).flipped);
    EXPECT_TRUE(cordic_reduce(std::nextafter(pi/2,pi)).flipped);
    EXPECT_TRUE(cordic_reduce(std::nextafter(-pi/2,-pi)).flipped);
}
TEST(CordicTrace, ZeroStagesAndResidualZeroDoNotPretendToBeAnAccuracyStop){
    const auto a=cordic_sincos_trace(1,0,1);EXPECT_TRUE(a.steps.empty());EXPECT_EQ(a.analysis.residual,1);
    EXPECT_EQ(a.analysis.raw_x,1);EXPECT_EQ(a.analysis.raw_y,0);EXPECT_EQ(a.analysis.inverse_gain,1);
    EXPECT_EQ(a.analysis.result.sin,0);EXPECT_EQ(a.analysis.result.cos,1);EXPECT_EQ(a.analysis.work.rotations,0);
    const auto zero=cordic_sincos_trace(0,2);EXPECT_EQ(zero.steps[0].direction,1);EXPECT_EQ(zero.steps.size(),2);
    const auto exact=cordic_sincos_trace(pi/4,2);EXPECT_EQ(exact.steps[0].z,0);EXPECT_EQ(exact.steps[1].direction,1);
    EXPECT_EQ(cordic_sincos_trace(-0.,1).steps[0].direction,1);
}
TEST(CordicTrace, StageConstantsCoverEveryNonzeroBinary64ScaleAndOwnTheirCopies){
    auto c=cordic_stage_constants(cordic_rotation_limit);ASSERT_EQ(c.size(),1075);
    double scale=1,gain=1;
    for(std::size_t i=0;i<c.size();++i){EXPECT_EQ(c[i].index,i);EXPECT_EQ(c[i].shift,scale);EXPECT_GT(c[i].shift,0);
        EXPECT_EQ(c[i].angle,std::atan(scale));EXPECT_EQ(c[i].inverse_gain_step,1/std::sqrt(1+scale*scale));
        gain*=c[i].inverse_gain_step;scale*=.5;
    }
    EXPECT_EQ(c.back().shift,std::numeric_limits<double>::denorm_min());EXPECT_EQ(scale,0);
    EXPECT_NEAR(gain,.60725293500888125617,2e-15);
    c[0].angle=999;EXPECT_EQ(cordic_stage_constants(1)[0].angle,std::atan(1.));
    EXPECT_TRUE(cordic_stage_constants(0).empty());
    const auto a=cordic_sincos_analyze(1,1075);EXPECT_EQ(a.work.rotations,1075);EXPECT_TRUE(std::isfinite(a.result.sin));
}
TEST(CordicTrace, FullInputValidationAndCompleteTraceCaps){
    for(double theta:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}){
        EXPECT_THROW(cordic_sincos(theta,0),std::invalid_argument);
        EXPECT_THROW(cordic_reduce(theta),std::invalid_argument);
        EXPECT_THROW(cordic_sincos_trace(theta,12,1),std::invalid_argument);
    }
    for(int n:{-1,1076,std::numeric_limits<int>::max()}){
        EXPECT_THROW(cordic_sincos(0,n),std::invalid_argument);
        EXPECT_THROW(cordic_stage_constants(n),std::invalid_argument);
        EXPECT_THROW(cordic_sincos_trace(0,n,1),std::invalid_argument);
    }
    EXPECT_EQ(cordic_sincos_trace(1,12,12).steps.size(),12);
    EXPECT_THROW(cordic_sincos_trace(1,12,11),std::length_error);
    EXPECT_EQ(cordic_sincos_trace(1,12,0).steps.size(),12);
}
TEST(CordicTrace, ErrorSweepMatchesIndependentAccumulationAndFirstTies){
    const std::vector<double> angles{-2.,-.5,0,.5,2.};const std::vector<int> ns{0,4,12,44,4};
    const auto rows=cordic_error_sweep(angles,ns,25);ASSERT_EQ(rows.size(),ns.size());
    for(std::size_t n=0;n<rows.size();++n){double se=0,ce=0,ne=0,score=-1,worst=0;
        for(double theta:angles){const auto s=former(theta,ns[n]);const double sin_error=std::abs(s.sin-std::sin(theta)),cos_error=std::abs(s.cos-std::cos(theta));
            se=std::max(se,sin_error);ce=std::max(ce,cos_error);ne=std::max(ne,std::abs(std::hypot(s.sin,s.cos)-1));
            const auto e=std::max(sin_error,cos_error);if(e>score){score=e;worst=theta;}}
        EXPECT_EQ(rows[n].iterations,ns[n]);EXPECT_EQ(rows[n].samples,angles.size());EXPECT_TRUE(rows[n].has_samples);
        EXPECT_EQ(rows[n].max_sin_error,se);EXPECT_EQ(rows[n].max_cos_error,ce);EXPECT_EQ(rows[n].max_norm_error,ne);
        EXPECT_EQ(rows[n].max_component_error,score);EXPECT_EQ(rows[n].worst_angle,worst);
    }
    EXPECT_EQ(cordic_error_sweep({1,-1},{0})[0].worst_angle,1);EXPECT_EQ(cordic_error_sweep({-1,1},{0})[0].worst_angle,-1);
    EXPECT_THROW(cordic_error_sweep(angles,ns,24),std::length_error);
}
TEST(CordicTrace, EmptySweepsStillValidateEveryInputAndUseExplicitAbsence){
    const auto empty=cordic_error_sweep({},{0,44});ASSERT_EQ(empty.size(),2);
    for(const auto& row:empty){EXPECT_FALSE(row.has_samples);EXPECT_EQ(row.samples,0);EXPECT_EQ(row.max_component_error,0);EXPECT_EQ(row.worst_angle,0);}
    EXPECT_TRUE(cordic_error_sweep({0,1},{},1).empty());
    EXPECT_THROW(cordic_error_sweep({},{-1}),std::invalid_argument);
    EXPECT_THROW(cordic_error_sweep({std::numeric_limits<double>::quiet_NaN()},{},1),std::invalid_argument);
    EXPECT_THROW(cordic_error_sweep({0,std::numeric_limits<double>::infinity()},{44,1076},1),std::invalid_argument);
    EXPECT_EQ(cordic_error_sweep({0,1},{0,44},4).size(),2);
    EXPECT_THROW(cordic_error_sweep({0,1},{0,44},3),std::length_error);
}
TEST(CordicTrace, LargeAnglesExposeRoundedPeriodErrorRatherThanFixingItByMoreSteps){
    const auto a=cordic_sincos_analyze(1e16,64);const auto b=cordic_sincos_analyze(1e16,1075);
    EXPECT_GT(std::hypot(a.result.sin-std::sin(1e16),a.result.cos-std::cos(1e16)),.1);
    EXPECT_NEAR(a.result.sin,b.result.sin,2e-15);EXPECT_NEAR(a.result.cos,b.result.cos,2e-15);
    EXPECT_NEAR(a.result.sin,(a.reduction.flipped?-1:1)*std::sin(a.reduction.reduced),2e-15);
}
TEST(CordicTrace, ActualNormAndPhaseIdentityAndOwnedTraceCopies){
    auto t=cordic_sincos_trace(1,44);const auto copy=t;
    for(const auto& s:t.steps){EXPECT_NEAR(std::hypot(s.unit_x,s.unit_y),1,4e-16);
        EXPECT_NEAR(std::hypot(s.x,s.y)*s.inverse_gain,1,3e-15);
        EXPECT_NEAR(std::atan2(s.y,s.x),t.analysis.reduction.reduced-s.z,2e-15);}
    t.steps[0].x=999;t.analysis.result.sin=999;t.analysis.reduction.reduced=999;
    EXPECT_EQ(copy.steps[0].x,1);EXPECT_NE(copy.analysis.result.sin,999);EXPECT_EQ(copy.analysis.reduction.reduced,1);
    same(copy.analysis.result,cordic_sincos(1,44));
}
