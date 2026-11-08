#include <datamunge/algorithms/ellipsoid.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Matrix=std::vector<std::vector<long double>>;
Matrix shape(const EllipsoidState& e){const auto n=e.center.size();Matrix p(n,std::vector<long double>(n));for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)for(std::size_t k=0;k<n;++k)p[i][j]+=static_cast<long double>(e.lower[i][k])*e.lower[j][k];return p;}
long double membership(const EllipsoidState& e,const std::vector<long double>& point){const auto n=e.center.size();std::vector<long double> z(n);long double norm=0;for(std::size_t i=0;i<n;++i){auto rhs=point[i]-e.center[i];for(std::size_t j=0;j<i;++j)rhs-=e.lower[i][j]*z[j];z[i]=rhs/e.lower[i][i];norm+=z[i]*z[i];}return norm;}
void compare_cut(const EllipsoidState& before,const std::vector<double>& g,const EllipsoidCut& cut){
    const auto n=g.size();const auto p=shape(before),actual=shape(cut.state);std::vector<long double> pg(n);long double gp=0;
    for(std::size_t i=0;i<n;++i){for(std::size_t j=0;j<n;++j)pg[i]+=p[i][j]*g[j];gp+=g[i]*pg[i];}
    const auto norm=std::sqrt(gp);EXPECT_NEAR(cut.support_bound,double(norm),1e-12*double(norm));
    for(std::size_t i=0;i<n;++i){EXPECT_GT(cut.state.lower[i][i],0);const auto center=static_cast<long double>(before.center[i])-pg[i]/norm/(n==1?2:n+1);EXPECT_NEAR(cut.state.center[i],double(center),1e-12*(1+std::abs(double(center))));
        for(std::size_t j=0;j<n;++j){const auto expected=n==1?p[i][j]/4:static_cast<long double>(n)*n/(static_cast<long double>(n)*n-1)*(p[i][j]-2.L/(n+1)*pg[i]*pg[j]/gp);EXPECT_NEAR(double(actual[i][j]),double(expected),1e-11*(1+std::abs(double(expected))));if(j>i)EXPECT_EQ(cut.state.lower[i][j],0);}}
    EXPECT_NEAR(cut.log_volume_after-cut.log_volume_before,ellipsoid_log_volume_ratio(n),2e-13);
}
void certificate(const EllipsoidTrace& t){
    const auto& o=t.outcome;const auto n=o.state.center.size(),triangle=n*(n+1)/2;ASSERT_EQ(t.samples.size(),o.work.updates+1);ASSERT_EQ(t.iterations.size(),o.work.subgradient_calls);
    EXPECT_EQ(o.result.iterations,int(t.iterations.size()));EXPECT_EQ(o.work.objective_calls,t.samples.size());EXPECT_EQ(o.work.factor_entries_updated,o.work.updates*triangle);
    std::size_t state=0,nonzero=0;double best=t.samples[0].objective;auto point=t.samples[0].state.center;
    for(std::size_t k=0;k<t.samples.size();++k){const auto& s=t.samples[k];if(s.objective<best){best=s.objective;point=s.state.center;}EXPECT_EQ(s.best_value,best);EXPECT_EQ(s.log_volume,ellipsoid_log_volume(s.state));if(k)EXPECT_NEAR(s.log_volume-t.samples[k-1].log_volume,ellipsoid_log_volume_ratio(n),3e-12);}
    for(std::size_t k=0;k<t.iterations.size();++k){const auto& it=t.iterations[k];ASSERT_EQ(it.state_index,state);ASSERT_EQ(it.gradient.size(),n);const bool zero=std::all_of(it.gradient.begin(),it.gradient.end(),[](double g){return g==0;});nonzero+=!zero;
        if(it.updated){ASSERT_LT(state+1,t.samples.size());const auto c=ellipsoid_central_cut(t.samples[state].state,it.gradient);EXPECT_EQ(c.state.center,t.samples[state+1].state.center);EXPECT_EQ(c.state.lower,t.samples[state+1].state.lower);EXPECT_EQ(c.support_bound,it.support_bound);++state;}
        else{EXPECT_EQ(k+1,t.iterations.size());if(o.stop==EllipsoidStop::zero_subgradient){EXPECT_TRUE(zero);EXPECT_EQ(it.support_bound,0);}}
    }
    EXPECT_EQ(o.work.support_products,nonzero*triangle);EXPECT_EQ(o.result.value,best);EXPECT_EQ(o.result.x,point);EXPECT_EQ(o.state.center,t.samples.back().state.center);EXPECT_EQ(o.state.lower,t.samples.back().state.lower);
    EXPECT_EQ(t.captured_entries,t.samples.size()*(1+n+n*n)+t.iterations.size()*(1+n));EXPECT_EQ(o.has_support_bound,!t.iterations.empty());if(o.has_support_bound)EXPECT_EQ(o.last_support_bound,t.iterations.back().support_bound);
}
}
TEST(EllipsoidTrace, RandomCentralCutsMatchIndependentMatrixAndContainmentIdentities){
    std::mt19937 rng(1979);std::uniform_real_distribution<double> uniform(-1,1);
    for(int trial=0;trial<250;++trial){const std::size_t n=1+rng()%6;auto e=ellipsoid_ball(std::vector<double>(n,0),1);std::vector<double> g(n);
        for(std::size_t i=0;i<n;++i){e.center[i]=uniform(rng);e.lower[i][i]=.5+2*(1+uniform(rng));for(std::size_t j=0;j<i;++j)e.lower[i][j]=uniform(rng);g[i]=uniform(rng);}const auto c=ellipsoid_central_cut(e,g);compare_cut(e,g,c);
        for(int sample=0;sample<60;++sample){std::vector<long double> z(n),point(n);long double norm=0,dot=0;for(auto& x:z){x=uniform(rng);norm=std::hypot(norm,x);}for(auto& x:z)x/=norm;
            for(std::size_t i=0;i<n;++i){point[i]=e.center[i];for(std::size_t j=0;j<=i;++j)point[i]+=e.lower[i][j]*z[j];dot+=g[i]*(point[i]-e.center[i]);}
            if(dot<=0)EXPECT_LE(membership(c.state,point),1+2e-11L);
        }
        auto scaled=g;for(auto& x:scaled)x*=1e200;const auto big=ellipsoid_central_cut(e,scaled);for(std::size_t i=0;i<n;++i){EXPECT_NEAR(big.state.center[i],c.state.center[i],1e-12);for(std::size_t j=0;j<=i;++j)EXPECT_NEAR(big.state.lower[i][j],c.state.lower[i][j],1e-12);}
        for(std::size_t i=0;i<n;++i)scaled[i]=g[i]*1e-200;const auto small=ellipsoid_central_cut(e,scaled);for(std::size_t i=0;i<n;++i)EXPECT_NEAR(small.state.center[i],c.state.center[i],1e-12);
    }
}
TEST(EllipsoidTrace, OriginalQuadraticsUseActualCallbacksAndReachRequestedAccuracy){
    for(int dim:{2,3}){const std::vector<double> target=dim==2?std::vector<double>{3,-1}:std::vector<double>{2,-3,.5},weights=dim==2?std::vector<double>{1,5}:std::vector<double>{1,4,2};std::vector<std::vector<double>> fp,gp;
        const auto pure=[&](const std::vector<double>& x){double value=0;for(std::size_t i=0;i<x.size();++i)value+=weights[i]*(x[i]-target[i])*(x[i]-target[i]);return value;};
        auto f=[&](const std::vector<double>& x){fp.push_back(x);return pure(x);};auto g=[&](const std::vector<double>& x){gp.push_back(x);std::vector<double> out(x.size());for(std::size_t i=0;i<x.size();++i)out[i]=2*weights[i]*(x[i]-target[i]);return out;};
        const int cap=dim==2?500:2000;const double radius=dim==2?10:20;const auto t=ellipsoid_minimize_trace(f,g,std::vector<double>(dim,0),radius,cap,1e-10);certificate(t);EXPECT_EQ(t.outcome.stop,EllipsoidStop::support_tolerance);EXPECT_LE(t.outcome.last_support_bound,1e-10);EXPECT_LT(t.outcome.result.value,1e-10);ASSERT_EQ(fp.size(),t.samples.size());ASSERT_EQ(gp.size(),t.iterations.size());bool nondescending=false;
        for(std::size_t i=0;i<t.samples.size();++i){EXPECT_EQ(fp[i],t.samples[i].state.center);EXPECT_EQ(pure(fp[i]),t.samples[i].objective);std::vector<long double> optimum(target.begin(),target.end());EXPECT_LE(membership(t.samples[i].state,optimum),1+1e-5L);if(i&&t.samples[i].objective>t.samples[i-1].objective)nondescending=true;}
        EXPECT_TRUE(nondescending);for(std::size_t i=0;i<gp.size();++i)EXPECT_EQ(gp[i],t.samples[t.iterations[i].state_index].state.center);
        const auto first_f=fp,first_g=gp;fp.clear();gp.clear();const auto plain=ellipsoid_minimize_work(f,g,std::vector<double>(dim,0),radius,cap,1e-10);EXPECT_EQ(fp,first_f);EXPECT_EQ(gp,first_g);EXPECT_EQ(plain.result.x,t.outcome.result.x);EXPECT_EQ(plain.result.value,t.outcome.result.value);
    }
    // Independent endpoint bisection checks the initial and every accepted center,
    // including the final cut that the old 1-D midpoint loop failed to evaluate.
    for(int fixture=0;fixture<8;++fixture)for(int cap:{0,1,2,5,12,24}){
        const double star=(fixture+1)/3.,weight=1+fixture%5;double lo=-22,hi=18,center=-2,point=center;
        const auto value=[&](double x){return weight*(x-star)*(x-star);};double best=value(center);
        for(int step=0;step<cap;++step){if(center==star)break;if(center>star)hi=center;else lo=center;center=(lo+hi)/2;if(value(center)<best){best=value(center);point=center;}}
        const auto r=ellipsoid_minimize([&](const std::vector<double>& x){return value(x[0]);},[&](const std::vector<double>& x){return std::vector<double>{2*weight*(x[0]-star)};},{-2},20,cap,0);
        EXPECT_EQ(r.value,best);EXPECT_EQ(r.x[0],point);
    }
    const auto f=[](const std::vector<double>& x){return std::abs(x[0]-1)+std::abs(x[0]+2);};const auto g=[](const std::vector<double>& x){return std::vector<double>{(x[0]>1?1.:-1.)+(x[0]>-2?1.:-1.)};};const auto t=ellipsoid_minimize_trace(f,g,{5},10,200);certificate(t);EXPECT_EQ(t.outcome.result.value,3);EXPECT_EQ(t.outcome.stop,EllipsoidStop::zero_subgradient);EXPECT_EQ(t.outcome.work.updates,1u);
}
TEST(EllipsoidTrace, ExactHistoryQuotasPrecedeTheirCallbacksAndCopiesAreIndependent){
    int fc=0,gc=0;auto f=[&](const std::vector<double>& x){++fc;return std::abs(x[0]-2);};auto g=[&](const std::vector<double>& x){++gc;return std::vector<double>{x[0]<2?-1.:x[0]>2?1.:0.};};
    const auto t=ellipsoid_minimize_trace(f,g,{0},4,10,1e-10,10);certificate(t);EXPECT_EQ(t.captured_entries,10u);EXPECT_EQ(fc,2);EXPECT_EQ(gc,2);EXPECT_EQ(t.outcome.result.x[0],2);EXPECT_EQ(t.outcome.result.value,0);
    for(std::size_t cap=1;cap<10;++cap){fc=gc=0;EXPECT_THROW((void)ellipsoid_minimize_trace(f,g,{0},4,10,1e-10,cap),std::length_error);if(cap<3){EXPECT_EQ(fc,0);EXPECT_EQ(gc,0);}else if(cap<5){EXPECT_EQ(fc,1);EXPECT_EQ(gc,0);}else if(cap<8){EXPECT_EQ(fc,1);EXPECT_EQ(gc,1);}else{EXPECT_EQ(fc,2);EXPECT_EQ(gc,1);}}
    EXPECT_EQ(ellipsoid_minimize_trace(f,g,{0},4,10,1e-10,0).captured_entries,10u);fc=gc=0;const auto no=ellipsoid_minimize_trace(f,g,{0},4,0,1e-10,3);certificate(no);EXPECT_EQ(fc,1);EXPECT_EQ(gc,0);EXPECT_FALSE(no.outcome.has_support_bound);EXPECT_EQ(no.outcome.stop,EllipsoidStop::iteration_limit);
    std::vector<double> center{0};const auto owned=ellipsoid_minimize_trace(f,g,center,4,10);center[0]=99;auto copy=owned;copy.samples[0].state.lower[0][0]=99;copy.iterations[0].gradient[0]=99;copy.outcome.state.center[0]=99;EXPECT_EQ(owned.samples[0].state.center[0],0);EXPECT_EQ(owned.samples[0].state.lower[0][0],4);EXPECT_EQ(owned.iterations[0].gradient[0],-1);EXPECT_EQ(owned.outcome.state.center[0],2);
}
TEST(EllipsoidTrace, FullDomainAndOracleValidationRejectUnsafeInputs){
    const double inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();int fc=0;auto f=[&](const std::vector<double>&){++fc;return 0.;};auto g=[](const std::vector<double>& x){return std::vector<double>(x.size(),0);};
    for(double radius:{0.,-1.,inf,nan}){fc=0;EXPECT_THROW((void)ellipsoid_minimize_trace(f,g,{0,0},radius),std::invalid_argument);EXPECT_EQ(fc,0);}
    for(double tol:{-1.,inf,nan}){EXPECT_THROW((void)ellipsoid_minimize_work(f,g,{0},1,10,tol),std::invalid_argument);}
    EXPECT_THROW((void)ellipsoid_minimize_work(f,g,{},1),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_minimize_work(f,g,{nan},1),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_minimize_work(f,g,{0},1,-1),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_minimize_work({},g,{0},1),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_minimize_work(f,{}, {0},1),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_minimize_work(f,[](const std::vector<double>&){return std::vector<double>{};},{0,0},1),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_minimize_work(f,[nan](const std::vector<double>&){return std::vector<double>{nan};},{0},1),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_minimize_work([nan](const std::vector<double>&){return nan;},g,{0},1),std::invalid_argument);
    struct Marker:std::runtime_error{Marker():std::runtime_error("callback marker") {}};
    EXPECT_THROW((void)ellipsoid_minimize_trace([](const std::vector<double>&)->double{throw Marker{};},g,{0},1),Marker);
    EXPECT_THROW((void)ellipsoid_minimize_trace(f,[](const std::vector<double>&)->std::vector<double>{throw Marker{};},{0},1),Marker);
    auto e=ellipsoid_ball({0,0},1);for(int mode=0;mode<4;++mode){auto bad=e;if(mode==0)bad.lower.pop_back();if(mode==1)bad.lower[0][1]=1;if(mode==2)bad.lower[1][1]=0;if(mode==3)bad.lower[1][0]=nan;
        EXPECT_THROW((void)ellipsoid_central_cut(bad,{1,0}),std::invalid_argument);}
    EXPECT_THROW((void)ellipsoid_central_cut(e,{0,0}),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_central_cut(e,{1}),std::invalid_argument);
}
TEST(EllipsoidTrace, ExactVolumeAndWorkFamiliesSeparateShrinkingVolumeFromAxisGrowth){
    auto e=ellipsoid_ball({0,0},1);const double initial=ellipsoid_log_volume(e),ratio=ellipsoid_log_volume_ratio(2);
    for(int t=1;t<=30;++t){e=ellipsoid_central_cut(e,{1,0}).state;EXPECT_NEAR(e.lower[0][0],std::pow(2./3,t),1e-13);EXPECT_NEAR(e.lower[1][1],std::pow(2/std::sqrt(3.),t),1e-11);EXPECT_GT(e.lower[1][1],1);EXPECT_NEAR(ellipsoid_log_volume(e)-initial,t*ratio,2e-13);}
    EXPECT_LT(std::exp(ratio),std::exp(-.25));EXPECT_NEAR(std::exp(ratio),4/(3*std::sqrt(3.)),1e-15);EXPECT_DOUBLE_EQ(ellipsoid_log_volume_ratio(1),-std::log(2.));
    const auto big=std::numeric_limits<std::size_t>::max();EXPECT_LT(ellipsoid_log_volume_ratio(big),0);EXPECT_NEAR(ellipsoid_log_volume_ratio(big),-1/(2.*static_cast<double>(big)),1e-35);
    const auto f=[](const std::vector<double>& x){return std::abs(x[0]+2);};const auto g=[](const std::vector<double>& x){return std::vector<double>{x[0]<-2?-1.:x[0]>-2?1.:0.,0};};
    for(int cap:{1,2,4,8,16,20}){const auto t=ellipsoid_minimize_trace(f,g,{0,0},4,cap,0);certificate(t);EXPECT_EQ(t.outcome.stop,EllipsoidStop::iteration_limit);EXPECT_EQ(t.outcome.work.updates,std::size_t(cap));EXPECT_EQ(t.outcome.work.objective_calls,std::size_t(cap+1));EXPECT_EQ(t.outcome.work.support_products,std::size_t(3*cap));EXPECT_EQ(t.captured_entries,std::size_t(10*cap+7));}
}
TEST(EllipsoidTrace, FiniteGeometryLimitsStagnationAndNativeOutlinesAreExplicit){
    const double maximum=std::numeric_limits<double>::max(),tiny=std::numeric_limits<double>::denorm_min();const auto huge=ellipsoid_ball({0,0},maximum);EXPECT_TRUE(std::isfinite(ellipsoid_log_volume(huge)));
    EXPECT_THROW((void)ellipsoid_central_cut(huge,{1,0}),std::overflow_error);
    EXPECT_THROW((void)ellipsoid_central_cut(ellipsoid_ball({0},tiny),{1}),std::runtime_error);
    const auto stalled=ellipsoid_central_cut(ellipsoid_ball({1e100},1),{1});EXPECT_FALSE(stalled.center_changed);
    // Numeric stagnation is visible even when caller enclosure assumptions are unverified.
    const auto t=ellipsoid_minimize_trace([](const std::vector<double>& x){return x[0];},[](const std::vector<double>&){return std::vector<double>{1};},{1e100},1,10,0);certificate(t);EXPECT_EQ(t.outcome.stop,EllipsoidStop::center_stagnation);EXPECT_EQ(t.outcome.work.updates,0u);EXPECT_GT(t.outcome.last_support_bound,0);
    const auto e=ellipsoid_ball({0,0},1);const auto p=ellipsoid_outline(e,32,32);ASSERT_EQ(p.size(),33u);EXPECT_EQ(p.front().x,p.back().x);EXPECT_EQ(p.front().y,p.back().y);for(auto q:p)EXPECT_NEAR(q.x*q.x+q.y*q.y,1,1e-14);
    EXPECT_THROW((void)ellipsoid_outline(e,7),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_outline(e,32,31),std::length_error);
    EXPECT_NO_THROW((void)ellipsoid_outline(e,32,0));
    EXPECT_THROW((void)ellipsoid_outline(ellipsoid_ball({0},1),8),std::invalid_argument);
    EXPECT_THROW((void)ellipsoid_outline(e,std::numeric_limits<std::size_t>::max(),0),std::length_error);
    EXPECT_THROW((void)ellipsoid_outline(ellipsoid_ball({maximum,0},maximum),8),std::overflow_error);
    EXPECT_THROW((void)ellipsoid_log_volume_ratio(0),std::invalid_argument);
}
