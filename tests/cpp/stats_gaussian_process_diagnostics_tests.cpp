#include <gtest/gtest.h>
#include <datamunge/stats/gaussian_process_regression.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <cmath>
#include <limits>

namespace {
using datamunge::dstruct::DataFrame;
using datamunge::stats::GaussianProcessRegression;
using datamunge::stats::GaussianProcessRegressionOptions;
using datamunge::linalg::DenseMatrix;
DataFrame points(std::vector<double> x,std::vector<double> y={}) {
    DataFrame out;out.add_column("x",x);if (!y.empty()) out.add_column("y",y);return out;
}
GaussianProcessRegressionOptions exact_options() {
    GaussianProcessRegressionOptions out;out.length_scale=1/std::sqrt(2*std::log(2.0));out.noise_ratio=.5;
    out.standardize=false;out.jitter_ratio=0;return out;
}
}

TEST(GaussianProcessDiagnostics, TwoPointMeansCovariancesAndDeletionAreExact) {
    const auto data=points({0,1},{-1,1});const GaussianProcessRegression gp(data,"y ~ x",exact_options());
    EXPECT_NEAR(gp.signal_variance(),1,2e-15);EXPECT_NEAR(gp.noise_variance(),.5,2e-15);
    const auto latent=gp.predict_detail(data),observed=gp.predict_observation_detail(data);
    const auto covariance=gp.predict_covariance(data),noisy=gp.predict_covariance(data,true);
    for (std::size_t i=0;i<2;++i) {
        EXPECT_NEAR(latent.fit[i],i?.5:-.5,2e-15);
        EXPECT_NEAR(latent.se_fit[i]*latent.se_fit[i],5.0/16,2e-15);
        EXPECT_NEAR(observed.se_fit[i]*observed.se_fit[i],13.0/16,2e-15);
        EXPECT_NEAR(covariance(i,i),5.0/16,2e-15);
        EXPECT_NEAR(noisy(i,i),13.0/16,2e-15);
        EXPECT_NEAR(gp.fitted_values()[i],i?-1.0/3:1.0/3,2e-15);
    }
    EXPECT_NEAR(covariance(0,1),1.0/16,2e-15);
    EXPECT_NEAR(noisy(0,1),1.0/16,2e-15);
    EXPECT_NEAR(gp.rmse(),4.0/3,2e-15);
    EXPECT_NEAR(gp.predict_detail(points({.5})).se_fit[0]*gp.predict_detail(points({.5})).se_fit[0],1-1/std::sqrt(2.0),2e-15);
    EXPECT_NEAR(gp.log_marginal_likelihood(),-1-.5*std::log(2.0)-std::log(2*3.14159265358979323846),2e-15);
}

TEST(GaussianProcessDiagnostics, RepeatedQueriesShareLatentValueButNotIndependentObservationNoise) {
    const auto data=points({0,0},{-1,1});auto options=exact_options();options.length_scale=1;
    const GaussianProcessRegression gp(data,"y ~ x",options);
    const auto covariance=gp.predict_covariance(data),noisy=gp.predict_covariance(data,true);
    EXPECT_NEAR(gp.signal_variance(),2,2e-15);EXPECT_NEAR(gp.noise_variance(),1,2e-15);
    for (std::size_t i=0;i<2;++i) for (std::size_t j=0;j<2;++j) {
        EXPECT_NEAR(covariance(i,j),.4,2e-15);
        EXPECT_NEAR(noisy(i,j),i==j?1.4:.4,2e-15);
    }
    EXPECT_NEAR(covariance(0,0)+covariance(1,1)-2*covariance(0,1),0,2e-15);
    EXPECT_NEAR(noisy(0,0)+noisy(1,1)-2*noisy(0,1),2,2e-15);
    options.noise_ratio=0;EXPECT_THROW(GaussianProcessRegression(data,"y ~ x",options),std::runtime_error);
}

TEST(GaussianProcessDiagnostics, NonTinyJitterUsesSameCovarianceForLikelihoodAndPrediction) {
    const auto data=points({0,1},{-1,1});auto options=exact_options();options.jitter_ratio=.25;
    const GaussianProcessRegression gp(data,"y ~ x",options);
    const double signal=.8; // z' C^-1 z / 2 = 1/(1+j).
    const DenseMatrix<double> C(2,2,{1.75,.5,.5,1.75});
    const auto K=signal*C;const auto chol=datamunge::linalg::cholesky(K);
    const std::vector<double> z{-1,1};const auto alpha=chol.solve(z);
    const double quad=z[0]*alpha[0]+z[1]*alpha[1];
    const double ll=-.5*(quad+std::log(chol.det())+2*std::log(2*3.14159265358979323846));
    EXPECT_NEAR(gp.signal_variance(),signal,2e-15);EXPECT_NEAR(gp.log_marginal_likelihood(),ll,2e-15);
    const auto posterior=gp.predict_detail(points({0}));const std::vector<double> cross{signal,.5*signal};
    const auto solve=chol.solve(cross);
    EXPECT_NEAR(posterior.fit[0],cross[0]*alpha[0]+cross[1]*alpha[1],2e-15);
    EXPECT_NEAR(posterior.se_fit[0]*posterior.se_fit[0],signal-cross[0]*solve[0]-cross[1]*solve[1],2e-15);
    const auto observation=gp.predict_observation_detail(points({0}));
    EXPECT_NEAR(observation.se_fit[0]*observation.se_fit[0]-posterior.se_fit[0]*posterior.se_fit[0],signal*.5,2e-15);
}

TEST(GaussianProcessDiagnostics, FloorLikelihoodIsTheActualGaussianDensityForConstantResponses) {
    const auto data=points({0,1},{7,7});auto options=exact_options();const GaussianProcessRegression gp(data,"y ~ x",options);
    EXPECT_DOUBLE_EQ(gp.signal_variance(),1e-12);
    const double logdet=2*std::log(1e-12)+std::log(2.0);
    EXPECT_NEAR(gp.log_marginal_likelihood(),-.5*(logdet+2*std::log(2*3.14159265358979323846)),1e-13);
    const auto prediction=gp.predict_observation_detail(points({0,100}));
    for (double mean : prediction.fit) EXPECT_DOUBLE_EQ(mean,7);
    for (double sd : prediction.se_fit) EXPECT_GT(sd,0);
}

TEST(GaussianProcessDiagnostics, OriginalUnitsAndLikelihoodJacobianAreEquivariant) {
    const auto base=points({0,1},{-1,1}),scaled=points({0,1},{7,13});
    const GaussianProcessRegression a(base,"y ~ x",exact_options()),b(scaled,"y ~ x",exact_options());
    const auto query=points({-.5,.5,3});const auto pa=a.predict_detail(query),pb=b.predict_detail(query);
    const auto ca=a.predict_covariance(query),cb=b.predict_covariance(query);
    EXPECT_NEAR(a.signal_variance(),b.signal_variance(),2e-15);
    EXPECT_NEAR(b.log_marginal_likelihood_original(),a.log_marginal_likelihood_original()-2*std::log(3.0),2e-15);
    for (std::size_t i=0;i<3;++i) {
        EXPECT_NEAR(pb.fit[i],10+3*pa.fit[i],1e-14);EXPECT_NEAR(pb.se_fit[i],3*pa.se_fit[i],2e-15);
        for (std::size_t j=0;j<3;++j) EXPECT_NEAR(cb(i,j),9*ca(i,j),5e-15);
    }
}

TEST(GaussianProcessDiagnostics, ConditionalDeletionMatchesIndependentFixedParameterSubmatrixRefits) {
    const std::vector<double> x{0,.2,.5,.8,1},y{-1,.1,.7,-.4,1};
    auto options=exact_options();options.jitter_ratio=.1;
    const GaussianProcessRegression gp(points(x,y),"y ~ x",options);
    const auto details=gp.training_details();const auto& loo=details.numeric_column("loo_mean").values();
    const auto& sd=details.numeric_column("loo_observation_sd").values();
    const double s=gp.signal_variance(),ell=gp.length_scale(),scale=gp.response_scale(),mean=gp.response_mean();
    for (std::size_t removed=0;removed<x.size();++removed) {
        std::vector<double> keep;std::vector<double> targets;
        for (std::size_t i=0;i<x.size();++i) if (i!=removed) {keep.push_back(x[i]);targets.push_back((y[i]-mean)/scale);}
        DenseMatrix<double> K(4,4,0);std::vector<double> cross(4);
        for (std::size_t i=0;i<4;++i) {
            cross[i]=s*std::exp(-std::pow((x[removed]-keep[i])/ell,2)/2);
            for (std::size_t j=0;j<4;++j) K(i,j)=s*(std::exp(-std::pow((keep[i]-keep[j])/ell,2)/2)+(i==j?options.noise_ratio+options.jitter_ratio:0));
        }
        const auto chol=datamunge::linalg::cholesky(K);const auto alpha=chol.solve(targets),white=chol.solve(cross);
        double mu=0,variance=s*(1+options.noise_ratio);
        for (std::size_t i=0;i<4;++i) {mu+=cross[i]*alpha[i];variance-=cross[i]*white[i];}
        EXPECT_NEAR(loo[removed],mean+scale*mu,2e-14);
        EXPECT_NEAR(sd[removed]*sd[removed],scale*scale*variance,3e-14);
    }
}

TEST(GaussianProcessDiagnostics, JointCovarianceIsSymmetricPsdAndMatchesPointwiseVariance) {
    const GaussianProcessRegression gp(points({0,.3,.7,1},{-1,.4,-.2,1}),"y ~ x",exact_options());
    const auto query=points({-.2,.1,.1,.5,1.2});const auto C=gp.predict_covariance(query);
    const auto prediction=gp.predict_detail(query),observed=gp.predict_observation_detail(query);
    const auto eigen=datamunge::linalg::jacobi_eigen(C);
    for (double value : eigen.eigenvalues) EXPECT_GE(value,-1e-13);
    for (std::size_t i=0;i<5;++i) {
        EXPECT_NEAR(C(i,i),prediction.se_fit[i]*prediction.se_fit[i],1e-14);
        EXPECT_NEAR(observed.se_fit[i]*observed.se_fit[i]-C(i,i),gp.noise_variance()*gp.response_scale()*gp.response_scale(),1e-14);
        for (std::size_t j=0;j<5;++j) EXPECT_DOUBLE_EQ(C(i,j),C(j,i));
    }
}

TEST(GaussianProcessDiagnostics, SelectionTableProfileAndOnePointAutomaticFlagsAgree) {
    GaussianProcessRegressionOptions options;options.n_length_scale_grid=4;options.n_noise_grid=3;
    const GaussianProcessRegression gp(points({0,.2,.5,.8,1},{-1,0,.3,.5,1}),"y ~ x",options);
    const auto table=gp.hyperparameter_search();EXPECT_EQ(table.nrows(),12u);
    const auto& ll=table.numeric_column("log_likelihood").values();
    EXPECT_DOUBLE_EQ(gp.log_marginal_likelihood(),*std::max_element(ll.begin(),ll.end()));
    EXPECT_EQ(std::count(table.numeric_column("selected").values().begin(),table.numeric_column("selected").values().end(),1.0),1);
    for (std::size_t i=0;i<4;++i) EXPECT_DOUBLE_EQ(gp.length_scale_profile_log_likelihood()[i],*std::max_element(ll.begin()+3*i,ll.begin()+3*(i+1)));
    options.n_length_scale_grid=1;options.n_noise_grid=1;
    const GaussianProcessRegression one(points({0,1},{-1,1}),"y ~ x",options);
    EXPECT_TRUE(one.length_scale_was_selected());EXPECT_TRUE(one.noise_ratio_was_selected());
    EXPECT_NO_THROW(one.plot_length_scale_profile());
}

TEST(GaussianProcessDiagnostics, FarQueriesRevertToPlugInPriorAndFiniteVariance) {
    const GaussianProcessRegression gp(points({0,1},{9,11}),"y ~ x",exact_options());
    const auto query=points({-100,100,1000});const auto p=gp.predict_detail(query),q=gp.predict_observation_detail(query);
    for (std::size_t i=0;i<3;++i) {
        EXPECT_DOUBLE_EQ(p.fit[i],10);EXPECT_NEAR(p.se_fit[i]*p.se_fit[i],1,2e-15);
        EXPECT_NEAR(q.se_fit[i]*q.se_fit[i],1.5,2e-15);
    }
}

TEST(GaussianProcessDiagnostics, HeldoutScoreSeparatesLatentAndObservationCoverage) {
    const GaussianProcessRegression gp(points({0,1},{-1,1}),"y ~ x",exact_options());
    const auto heldout=points({100,100},{1.5,-1.5});const auto score=gp.score(heldout);
    EXPECT_NEAR(score.numeric_column("rmse").values()[0],1.5,1e-14);
    EXPECT_DOUBLE_EQ(score.numeric_column("n").values()[0],2);
    EXPECT_DOUBLE_EQ(score.numeric_column("observation_coverage").values()[0],1);
    EXPECT_DOUBLE_EQ(score.numeric_column("latent_coverage").values()[0],1);
    // Tighter intervals make the target distinction observable: latent width .674, observation width .826 at 50%.
    const auto half=gp.score(points({100,100},{.75,-.75}),.5);
    EXPECT_DOUBLE_EQ(half.numeric_column("latent_coverage").values()[0],0);
    EXPECT_DOUBLE_EQ(half.numeric_column("observation_coverage").values()[0],1);
    EXPECT_NEAR(score.numeric_column("mean_nlpd").values()[0],.5*std::log(2*3.14159265358979323846*1.5)+.75,2e-15);
}

TEST(GaussianProcessDiagnostics, FiniteOptionsLevelsAndDataAreValidated) {
    const auto data=points({0,1},{-1,1});const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    for (double bad : {nan,inf,-inf}) {
        auto o=exact_options();o.length_scale=bad;EXPECT_THROW(GaussianProcessRegression(data,"y ~ x",o),std::invalid_argument);
        o=exact_options();o.noise_ratio=bad;EXPECT_THROW(GaussianProcessRegression(data,"y ~ x",o),std::invalid_argument);
        o=exact_options();o.jitter_ratio=bad;EXPECT_THROW(GaussianProcessRegression(data,"y ~ x",o),std::invalid_argument);
    }
    auto o=exact_options();o.jitter_ratio=-.1;EXPECT_THROW(GaussianProcessRegression(data,"y ~ x",o),std::invalid_argument);
    EXPECT_THROW(GaussianProcessRegression(points({0,inf},{0,1}),"y ~ x",exact_options()),std::invalid_argument);
    EXPECT_THROW(GaussianProcessRegression(points({0,1},{0,nan}),"y ~ x",exact_options()),std::invalid_argument);
    const GaussianProcessRegression gp(data,"y ~ x",exact_options());
    for (double bad : {0.,1.,nan,inf}) EXPECT_THROW(gp.predict_detail(data,bad),std::invalid_argument);
    EXPECT_THROW(gp.predict_covariance(points({inf})),std::invalid_argument);
    EXPECT_THROW(gp.plot_fit(data,1),std::invalid_argument);
    EXPECT_THROW(gp.score(points({0,1})),std::invalid_argument);
    EXPECT_EQ(gp.predict_detail(DataFrame{}).fit.size(),0u);EXPECT_EQ(gp.predict_covariance(DataFrame{}).rows(),0u);
}

TEST(GaussianProcessDiagnostics, ExtremePositiveLengthScalesHaveFiniteSelfCorrelation) {
    auto tiny=exact_options();tiny.length_scale=1e-300;
    const GaussianProcessRegression a(points({0,1},{-1,1}),"y ~ x",tiny);
    EXPECT_TRUE(std::isfinite(a.predict_detail(points({0,.5})).se_fit[0]));
    auto huge=exact_options();huge.length_scale=1e300;
    const GaussianProcessRegression b(points({0,1e300},{-1,1}),"y ~ x",huge);
    EXPECT_TRUE(std::isfinite(b.log_marginal_likelihood()));
}

TEST(GaussianProcessDiagnostics, NullRowsRetainSourceAlignmentInDiagnosticsAndQueries) {
    DataFrame data;data.add_column("x",std::vector<std::optional<double>>{0,std::nullopt,1,2});
    data.add_column("y",std::vector<std::optional<double>>{-1,0,1,std::nullopt});
    const GaussianProcessRegression gp(data,"y ~ x",exact_options());
    EXPECT_EQ(gp.kept_row_indices(),(std::vector<std::size_t>{0,2}));
    const auto details=gp.training_details();EXPECT_DOUBLE_EQ(details.numeric_column("row").values()[1],3);
    DataFrame query;query.add_column("x",std::vector<std::optional<double>>{0,std::nullopt,1});
    const auto p=gp.predict_detail(query);const auto C=gp.predict_covariance(query);
    ASSERT_EQ(p.fit.size(),3u);EXPECT_TRUE(std::isnan(p.fit[1]));EXPECT_TRUE(std::isnan(C(0,1)));
    EXPECT_TRUE(std::isnan(C(1,2)));EXPECT_NEAR(C(0,2),1.0/16,2e-15);
}

TEST(GaussianProcessDiagnostics, ConditionalDeletionDoesNotPretendToRefitEmpiricalMean) {
    const auto data=points({0,1},{-1,1});const GaussianProcessRegression full(data,"y ~ x",exact_options());
    const GaussianProcessRegression deleted(points({1},{1}),"y ~ x",exact_options());
    EXPECT_NEAR(full.fitted_values()[0],1.0/3,2e-15);
    EXPECT_DOUBLE_EQ(deleted.predict(points({0}))[0],1);
}
