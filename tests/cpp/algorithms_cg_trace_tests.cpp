#include <datamunge/algorithms/conjugate_gradient.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
using namespace datamunge::algorithms;
using V = std::vector<double>;
using M = std::vector<V>;
long double dot_ref(const V& x, const V& y) {
    long double s=0; for (std::size_t i=0;i<x.size();++i) s+=static_cast<long double>(x[i])*y[i];return s;
}
V mul_ref(const M& a,const V& x) {
    V y(a.size());for(std::size_t i=0;i<a.size();++i)y[i]=static_cast<double>(dot_ref(a[i],x));return y;
}
long double residual_ref(const M& a,const V& x,const V& b) {
    const auto y=mul_ref(a,x);long double z=0;
    for(std::size_t i=0;i<b.size();++i)z=std::hypot(z,static_cast<long double>(b[i])-y[i]);return z;
}
long double energy_ref(const M& a,const V& x,const V& truth) {
    V e=x;for(std::size_t i=0;i<x.size();++i)e[i]-=truth[i];return dot_ref(e,mul_ref(a,e));
}
// Independent long-double Cholesky factorisation and triangular solves.
V cholesky_ref(const M& a,const V& b) {
    const auto n=a.size();std::vector<std::vector<long double>> l(n,std::vector<long double>(n));
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j){
        long double s=a[i][j];for(std::size_t k=0;k<j;++k)s-=l[i][k]*l[j][k];
        l[i][j]=i==j?std::sqrt(s):s/l[j][j];
    }
    std::vector<long double> y(n),z(n);V x(n);
    for(std::size_t i=0;i<n;++i){long double s=b[i];for(std::size_t j=0;j<i;++j)s-=l[i][j]*y[j];y[i]=s/l[i][i];}
    for(std::size_t i=n;i-->0;){long double s=y[i];for(std::size_t j=i+1;j<n;++j)s-=l[j][i]*z[j];z[i]=s/l[i][i];x[i]=static_cast<double>(z[i]);}
    return x;
}
}

TEST(CGTrace, HandCheckableActualStepsAndConjugacy) {
    const M a{{4,1},{1,3}};const V b{1,2};const auto t=conjugate_gradient_trace(a,b);
    ASSERT_TRUE(t.solution.converged);ASSERT_EQ(t.steps.size(),2);EXPECT_EQ(t.products,4);
    EXPECT_EQ(t.steps[0].alpha,.25L);EXPECT_EQ(t.steps[0].x,(V{.25,.5}));
    EXPECT_EQ(t.steps[0].r,(V{-.5,.25}));EXPECT_EQ(t.steps[0].beta,.0625L);
    EXPECT_TRUE(t.steps[0].has_next_direction);EXPECT_FALSE(t.steps[1].has_next_direction);
    EXPECT_EQ(t.steps[0].p,b);EXPECT_EQ(t.steps[0].ap,(V{6,7}));
    EXPECT_EQ(t.steps[1].p,(V{-.4375,.375}));
    EXPECT_NEAR(t.solution.x[0],1./11,1e-15);EXPECT_NEAR(t.solution.x[1],7./11,1e-15);
    EXPECT_EQ(dot_ref(t.steps[0].p,t.steps[1].ap),0);
    EXPECT_NEAR(static_cast<double>(dot_ref(t.steps[0].r,t.steps[1].r)),0,1e-15);
    EXPECT_EQ(t.solution.residual,static_cast<double>(residual_ref(a,t.solution.x,b)));
}

TEST(CGTrace, RandomSPDMatchesIndependentCholeskyAndActualRecurrence) {
    std::mt19937 rng(812);std::uniform_int_distribution<int> d(-4,4);
    for(int trial=0;trial<1000;++trial){
        const std::size_t n=1+rng()%9;M m(n,V(n)),a(n,V(n));V b(n);
        for(auto& row:m)for(auto& v:row)v=d(rng);
        for(std::size_t i=0;i<n;++i){b[i]=d(rng);for(std::size_t j=0;j<n;++j){
            for(std::size_t k=0;k<n;++k)a[i][j]+=m[k][i]*m[k][j];if(i==j)a[i][j]+=n;
        }}
        const auto expected=cholesky_ref(a,b);const auto t=conjugate_gradient_trace(a,b,{100,1e-9,0});
        ASSERT_TRUE(t.solution.converged)<<trial;EXPECT_LE(t.solution.residual,1e-9);
        for(std::size_t i=0;i<n;++i)EXPECT_NEAR(t.solution.x[i],expected[i],1e-9);
        const auto s=conjugate_gradient_solve(a,b,{100,1e-9,1});
        EXPECT_EQ(s.solution.x,t.solution.x);EXPECT_EQ(s.solution.residual,t.solution.residual);
        EXPECT_EQ(s.products,t.products);EXPECT_TRUE(s.steps.empty());
        V x(n),r=b,p=b;long double rs=dot_ref(r,r),energy=energy_ref(a,x,expected);
        for(const auto& step:t.steps){
            EXPECT_EQ(step.p,p);EXPECT_EQ(step.ap,mul_ref(a,p));
            EXPECT_EQ(step.alpha,rs/dot_ref(p,step.ap));
            for(std::size_t i=0;i<n;++i){
                x[i]=static_cast<double>(static_cast<long double>(x[i])+step.alpha*p[i]);
                r[i]=static_cast<double>(static_cast<long double>(r[i])-step.alpha*step.ap[i]);
            }
            EXPECT_EQ(step.x,x);EXPECT_EQ(step.r,r);EXPECT_EQ(step.true_residual,residual_ref(a,x,b));
            const auto next_energy=energy_ref(a,x,expected);EXPECT_LE(next_energy,energy+1e-16L);energy=next_energy;
            if(step.has_next_direction){
                const auto next_rs=dot_ref(r,r);EXPECT_EQ(step.beta,next_rs/rs);
                for(std::size_t i=0;i<n;++i)p[i]=static_cast<double>(static_cast<long double>(r[i])+step.beta*p[i]);
                rs=next_rs;
            }
        }
        EXPECT_EQ(t.products,2*t.steps.size());
    }
}

TEST(CGTrace, ZeroEmptyAndZeroBudgetAreTruthful) {
    for(const auto& a:std::vector<M>{M{},M{{2}}}){
        V b(a.size());const auto t=conjugate_gradient_trace(a,b,{0,0,1});
        EXPECT_TRUE(t.solution.converged);EXPECT_EQ(t.products,0);EXPECT_EQ(t.solution.iterations,0);EXPECT_TRUE(t.steps.empty());
    }
    const auto t=conjugate_gradient_trace(M{{2}},V{4},{0,0,1});
    EXPECT_FALSE(t.solution.converged);EXPECT_EQ(t.stop,CGStop::iteration_limit);EXPECT_EQ(t.solution.residual,4);
    EXPECT_EQ(t.solution.x,V{0});EXPECT_EQ(t.products,0);
    const auto inclusive=conjugate_gradient_trace(M{{2}},V{4},{0,4,1});EXPECT_TRUE(inclusive.solution.converged);
    EXPECT_FALSE(conjugate_gradient(M{{2}},V{4},0,0).converged);
}

TEST(CGTrace, OriginalLaplacianAndMatrixFreeCallsAgree) {
    const auto a=laplacian_1d_matrix(20);const V b(20,1);const auto expected=laplacian_1d_unit_solution(20);
    const auto t=conjugate_gradient_trace(a,b);ASSERT_TRUE(t.solution.converged);EXPECT_EQ(t.solution.iterations,10);
    EXPECT_EQ(t.solution.x,expected);EXPECT_EQ(t.solution.residual,0);EXPECT_EQ(t.products,20);
    std::vector<V> arguments;
    const LinearOperator op=[&](const V& v){arguments.push_back(v);return laplacian_1d_apply(v);};
    const auto u=conjugate_gradient_trace(op,b);EXPECT_EQ(u.solution.x,t.solution.x);EXPECT_EQ(u.products,arguments.size());
    ASSERT_EQ(arguments.size(),2*t.steps.size());
    for(std::size_t i=0;i<t.steps.size();++i){EXPECT_EQ(arguments[2*i],u.steps[i].p);EXPECT_EQ(arguments[2*i+1],u.steps[i].x);}
    const auto old=conjugate_gradient(a,b);EXPECT_EQ(old.x,t.solution.x);EXPECT_EQ(old.residual,t.solution.residual);
    const auto gs=gauss_seidel_trace(a,b,40,0);EXPECT_EQ(gs.solution.iterations,40);EXPECT_GT(gs.solution.residual,1);
    EXPECT_EQ(gs.solution.x,gauss_seidel(a,b,40,0).x);EXPECT_EQ(gs.residuals.size(),41);
}

TEST(CGTrace, SpectrumAndRHSDetermineExactArithmeticDimension) {
    const M repeated{{1,0,0,0},{0,1,0,0},{0,0,4,0},{0,0,0,4}};
    const auto two=conjugate_gradient_trace(repeated,V(4,1));EXPECT_TRUE(two.solution.converged);EXPECT_EQ(two.solution.iterations,2);
    const M three{{1,0,0},{0,4,0},{0,0,9}};
    const auto t=conjugate_gradient_trace(three,V(3,1));EXPECT_TRUE(t.solution.converged);EXPECT_EQ(t.solution.iterations,3);
    const auto one=conjugate_gradient_trace(three,{0,2,0});EXPECT_TRUE(one.solution.converged);EXPECT_EQ(one.solution.iterations,1);
}

TEST(CGTrace, ResidualCanRiseWhileEnergyFalls) {
    const M a{{1,0},{0,100}};const V b{1,.1},truth{1,.001};
    const auto t=conjugate_gradient_trace(a,b);ASSERT_TRUE(t.solution.converged);
    EXPECT_GT(t.steps[0].true_residual,t.initial_residual);
    EXPECT_LT(energy_ref(a,t.steps[0].x,truth),energy_ref(a,V(2),truth));
}

TEST(CGTrace, CurvatureFailureIsNotConvergenceOrSPDCertification) {
    const auto negative=conjugate_gradient_trace(M{{-1}},V{1});
    EXPECT_FALSE(negative.solution.converged);EXPECT_EQ(negative.stop,CGStop::nonpositive_curvature);
    EXPECT_EQ(negative.products,1);EXPECT_EQ(negative.solution.iterations,0);EXPECT_EQ(negative.solution.residual,1);
    const auto flat=conjugate_gradient_trace(M{{0}},V{1});EXPECT_EQ(flat.stop,CGStop::nonpositive_curvature);
    EXPECT_TRUE(conjugate_gradient_trace(M{{-1}},V{0}).solution.converged);
    const auto capped=conjugate_gradient_trace(M{{4,1},{1,3}},V{1,2},{1,0,0});
    EXPECT_EQ(capped.stop,CGStop::iteration_limit);EXPECT_EQ(capped.solution.iterations,1);EXPECT_FALSE(capped.solution.converged);
    EXPECT_FALSE(capped.steps.back().has_next_direction);
}

TEST(CGTrace, ResidualGapCannotSilentlyConverge) {
    bool found=false;
    for(int a=2;a<=100 && !found;++a)for(int b=1;b<=100 && !found;++b){
        const auto t=conjugate_gradient_trace(M{{static_cast<double>(a)}},V{static_cast<double>(b)},{8,0,0});
        if(t.stop==CGStop::residual_gap){found=true;EXPECT_FALSE(t.solution.converged);
            EXPECT_GT(t.solution.residual,0);EXPECT_EQ(t.solution.residual,static_cast<double>(residual_ref(M{{static_cast<double>(a)}},t.solution.x,V{static_cast<double>(b)})));
        }
    }
    EXPECT_TRUE(found);
}

TEST(CGTrace, FullValidationEvenInitiallyConvergedAndBeforeCaps) {
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    for(const M a:{M{{2},{1,2}},M{{2,1},{0,2}},M{{2,0},{0,nan}},M{{2,0},{0,inf}}})
        EXPECT_THROW(conjugate_gradient_trace(a,V(2),{0,0,1}),std::invalid_argument);
    EXPECT_THROW(conjugate_gradient_trace(M{{2}},V{}),std::invalid_argument);
    EXPECT_THROW(conjugate_gradient_trace(M{{2}},V{nan}),std::invalid_argument);
    for(double tol:{-1.,nan,inf})EXPECT_THROW(conjugate_gradient_trace(M{},V{},{0,tol,1}),std::invalid_argument);
    EXPECT_THROW(conjugate_gradient_trace(M{},V{}, {-1,0,1}),std::invalid_argument);
    EXPECT_THROW(gauss_seidel_trace({{0}},{0},0,0,1),std::invalid_argument);
    EXPECT_THROW(gauss_seidel_trace({{2}},{0},-1),std::invalid_argument);
}

TEST(CGTrace, MatrixFreeContractFailuresAndExceptions) {
    const LinearOperator missing{};EXPECT_THROW(conjugate_gradient_solve(missing,{0}),std::invalid_argument);
    const LinearOperator badsize=[](const V&){return V{};};EXPECT_THROW(conjugate_gradient_trace(badsize,{1}),std::invalid_argument);
    const LinearOperator nonfinite=[](const V&){return V{std::numeric_limits<double>::infinity()};};
    EXPECT_THROW(conjugate_gradient_trace(nonfinite,{1}),std::invalid_argument);
    int calls=0;const LinearOperator forbidden=[&](const V&)->V{++calls;throw std::runtime_error("callback");};
    EXPECT_TRUE(conjugate_gradient_trace(forbidden,{0}).solution.converged);EXPECT_EQ(calls,0);
    EXPECT_THROW(conjugate_gradient_trace(forbidden,{1}),std::runtime_error);EXPECT_EQ(calls,1);
    int count=0;const LinearOperator badcheck=[&](const V& v){return ++count==1?V{2*v[0]}:V{};};
    EXPECT_THROW(conjugate_gradient_trace(badcheck,{1}),std::invalid_argument);EXPECT_EQ(count,2);
}

TEST(CGTrace, ActualSnapshotCapsAndOwnership) {
    const M a{{4,1},{1,3}};V b{1,2};auto t=conjugate_gradient_trace(a,b,{1000,1e-10,16});
    EXPECT_EQ(t.steps.size(),2);EXPECT_THROW(conjugate_gradient_trace(a,b,{1000,1e-10,15}),std::length_error);
    EXPECT_TRUE(conjugate_gradient_solve(a,b,{1000,1e-10,1}).solution.converged);
    EXPECT_TRUE(conjugate_gradient_trace(a,V(2),{1000,0,1}).solution.converged);
    auto copied=t;b[0]=99;t.steps[0].x[0]=99;t.solution.x[0]=99;
    EXPECT_EQ(copied.steps[0].x[0],.25);EXPECT_NE(copied.solution.x[0],99);
    EXPECT_EQ(conjugate_gradient_trace(M{{-1}},V{1},{100,0,1}).stop,CGStop::nonpositive_curvature);
}

TEST(CGTrace, HugeAndTinyScalesAvoidSquaredDoubleFailures) {
    for(const double scale:{1e-200,1e150}){
        const auto t=conjugate_gradient_trace(M{{2}},V{scale},{4,0,0});
        EXPECT_TRUE(t.solution.converged);EXPECT_DOUBLE_EQ(t.solution.x[0],scale/2);
    }
    if(std::numeric_limits<long double>::max_exponent>2*std::numeric_limits<double>::max_exponent){
        const auto t=conjugate_gradient_trace(M{{1}},V{std::numeric_limits<double>::max()},{2,0,0});
        EXPECT_TRUE(t.solution.converged);EXPECT_EQ(t.solution.x[0],std::numeric_limits<double>::max());
    }
    EXPECT_THROW(conjugate_gradient_trace(M{{std::numeric_limits<double>::max()}},V{2}),std::overflow_error);
}

TEST(CGTrace, StationaryHistoryMatchesIndependentSweepAndActualCaps) {
    const M a{{4,1},{1,3}};const V b{1,2};const auto t=gauss_seidel_trace(a,b,20,0,21);V x(2);
    EXPECT_EQ(t.residuals.front(),static_cast<double>(std::hypot(1.L,2.L)));
    ASSERT_EQ(t.residuals.size(),static_cast<std::size_t>(t.solution.iterations)+1);
    EXPECT_TRUE(t.solution.converged);
    for(std::size_t k=1;k<t.residuals.size();++k){
        x[0]=static_cast<double>((1.L-x[1])/4);x[1]=static_cast<double>((2.L-x[0])/3);
        EXPECT_EQ(t.residuals[k],static_cast<double>(residual_ref(a,x,b)));
    }
    EXPECT_EQ(t.solution.x,x);
    EXPECT_EQ(gauss_seidel_trace(a,b,20,0,t.residuals.size()).residuals,t.residuals);
    EXPECT_THROW(gauss_seidel_trace(a,b,20,0,t.residuals.size()-1),std::length_error);
    const auto z=gauss_seidel_trace({{2}},{0},100,0,1);EXPECT_EQ(z.residuals,V{0});EXPECT_TRUE(z.solution.converged);
    const auto no=gauss_seidel_trace({{2}},{4},0,0,1);EXPECT_EQ(no.residuals,V{4});EXPECT_FALSE(no.solution.converged);
    const auto exact=gauss_seidel_trace({{2}},{4},100,0,2);EXPECT_EQ(exact.residuals,(V{4,0}));EXPECT_TRUE(exact.solution.converged);
}

TEST(CGTrace, LaplacianFixturesIndependentIdentitiesAndCaps) {
    for(std::size_t n=0;n<=100;++n){
        const auto a=laplacian_1d_matrix(n);const auto x=laplacian_1d_unit_solution(n);
        const auto y=laplacian_1d_apply(x);EXPECT_EQ(y,mul_ref(a,x));EXPECT_EQ(y,V(n,1));
        if(n){V alternating(n);for(std::size_t i=0;i<n;++i)alternating[i]=(i%2?1:-1)*(i+1.);
            EXPECT_EQ(laplacian_1d_apply(alternating),mul_ref(a,alternating));}
    }
    EXPECT_EQ(laplacian_1d_matrix(2,4).size(),2);EXPECT_THROW(laplacian_1d_matrix(2,3),std::length_error);
    EXPECT_THROW(laplacian_1d_matrix(std::numeric_limits<std::size_t>::max(),0),std::length_error);
    EXPECT_THROW(laplacian_1d_unit_solution(3,2),std::length_error);
    EXPECT_EQ(laplacian_1d_unit_solution(3,3),(V{1.5,2,1.5}));
    EXPECT_THROW(laplacian_1d_apply({std::numeric_limits<double>::quiet_NaN()}),std::invalid_argument);
    EXPECT_THROW(laplacian_1d_apply({std::numeric_limits<double>::max()}),std::overflow_error);
}
