#include <datamunge/stats/gee.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_gee;
using datamunge::stats::GEECorrelation;
using datamunge::stats::GEEFamily;
using datamunge::stats::GEEOptions;
using datamunge::stats::GEEStatus;

TEST(GEE, MatchesIndependentDenseScoresAndSandwichAcrossFamiliesAndCorrelations) {
    const std::vector<std::vector<std::size_t>> visits{
        {4,1,0}, {4,3,1}, {7,3,2,0}, {4,2,1}, {6,5,3,0}, {4,3,2,0}};
    const double effects[]{-.7,.4,1.1,-.9,-.25,.6}, baseline[]{1,3,5,2,4,3};
    const auto n = 21u;
    DenseMatrix<double> x(n, 2, 1);
    std::vector<double> gaussian, binomial, poisson, gamma, offset;
    std::vector<std::size_t> cluster, visit, trials;
    for (std::size_t g = 0; g < visits.size(); ++g) {
        const auto id = g + 1;
        for (const auto v : visits[g]) {
            const auto a = std::cos(id + .4*v), o = .15 * std::cos(v);
            x(cluster.size(),1) = a;
            cluster.push_back(id);
            visit.push_back(v);
            offset.push_back(o);
            trials.push_back(5 + v%3);
            gaussian.push_back(2 + .7*a + effects[g] + .45*std::sin(id*v+v) + .1*v + o);
            poisson.push_back(std::floor(baseline[g] * std::exp(.4*a + .12*v)));
            binomial.push_back(std::round(trials.back() / (1 + std::exp(-effects[g]-.7*a-.3*std::sin(v)-o))));
            gamma.push_back(std::exp(.2+.65*a+.25*effects[g]+.12*std::sin(id+v)+o));
        }
    }
    const std::array<double,8> expected[]{
        {2.410246591312,.521093210659,0,.790232071019,.038039098345,.100796144350,.015396774532,.043942679161},
        {2.335607066495,.778983764443,.770534411692,.830622884803,.115973801805,.107993574382,.007736632217,.038876764751},
        {2.325933043633,.754766455836,.928781428949,.825557754783,.118107276755,.098050252217,.001281956507,.052914849847},
        {1.272392300791,.737892957782,.37,1,.028648742919,.045165843851,.007809088277,.022665737290},
        {.171783975667,.531949539588,-.32,1,.027347683312,.082747863267,.005841420588,.075250825148},
        {.225687321210,.647335423898,0,.036831208698,.001772929777,.004828439696,-.000888926861,.003784602185},
    };
    const GEEOptions options[]{
        {.offset = offset},
        {.correlation = GEECorrelation::Exchangeable, .offset = offset},
        {.correlation = GEECorrelation::AR1, .visit = visit, .offset = offset},
        {.family = GEEFamily::Poisson, .correlation = GEECorrelation::Exchangeable,
         .fixed_correlation = .37, .offset = offset},
        {.family = GEEFamily::Binomial, .correlation = GEECorrelation::AR1,
         .fixed_correlation = -.32, .visit = visit, .trials = trials, .offset = offset},
        {.family = GEEFamily::Gamma, .offset = offset},
    };
    const std::vector<double>* outcomes[]{&gaussian,&gaussian,&gaussian,&poisson,&binomial,&gamma};
    // R assembles and inverts each full working covariance, then sums the
    // cluster score outer products independently of the native whitening.
    for (std::size_t j = 0; j < std::size(options); ++j) {
        SCOPED_TRACE(j);
        const auto fit = fit_gee(x, *outcomes[j], cluster, options[j]);
        ASSERT_EQ(fit.status, GEEStatus::Converged);
        ASSERT_EQ(fit.sandwich_covariance.rows(), 2u);
        const std::array actual{fit.coefficients[0],fit.coefficients[1],fit.correlation_parameter,fit.dispersion,
            fit.model_covariance(0,0),fit.sandwich_covariance(0,0),fit.sandwich_covariance(1,0),
            fit.sandwich_covariance(1,1)};
        for (std::size_t k = 0; k < actual.size(); ++k)
            EXPECT_NEAR(actual[k], expected[j][k], 1e-8);
        const auto prediction = fit.predict(DenseMatrix<double>(2, 2, {1,0, 1,0}), {0,std::log(2.)});
        if (j == 3 || j == 5) {
            EXPECT_NEAR(prediction[1], 2 * prediction[0], 1e-13);
        }
        if (j == 4) {
            EXPECT_NEAR(prediction[0], 1 / (1 + std::exp(-fit.coefficients[0])), 1e-13);
        }
    }
}

TEST(GEE, HandlesImplicitVisitsAndReportsFiniteIterationLimit) {
    const DenseMatrix<double> x(6, 1, 1);
    const std::vector<double> y{1,2,5,4,3,2};
    const std::vector<std::size_t> cluster{3,8,3,8,3,8};
    GEEOptions options{.correlation = GEECorrelation::AR1,.fixed_correlation = .7};
    const auto a = fit_gee(x,y,cluster,options);
    options.visit = {10,5,11,6,12,7};
    const auto b = fit_gee(x,y,cluster,options);
    ASSERT_EQ(a.status, GEEStatus::Converged);
    EXPECT_DOUBLE_EQ(a.coefficients[0], b.coefficients[0]);
    EXPECT_DOUBLE_EQ(a.sandwich_covariance(0,0), b.sandwich_covariance(0,0));
    options.family = GEEFamily::Poisson;
    options.max_iterations = 1;
    const auto limited = fit_gee(x,y,cluster,options);
    EXPECT_EQ(limited.status, GEEStatus::IterationLimit);
    EXPECT_EQ(limited.iterations, 1u);
    EXPECT_EQ(limited.fitted.size(), y.size());
}

TEST(GEE, ValidatesVisitCorrelationAndResponseDomains) {
    const DenseMatrix<double> x(4,1,1);
    const std::vector<double> y{0,1,0,1};
    const std::vector<std::size_t> cluster{1,1,2,2};
    EXPECT_THROW((void)fit_gee(x,y,cluster,{.correlation = GEECorrelation::AR1,.visit = {1,1,2,3}}),
                 std::invalid_argument);
    EXPECT_THROW((void)fit_gee(x,y,cluster,{.correlation = GEECorrelation::Exchangeable,.fixed_correlation = 1}),
                 std::invalid_argument);
    EXPECT_THROW((void)fit_gee(x,y,cluster,{.family = GEEFamily::Gamma}), std::invalid_argument);
    EXPECT_THROW((void)fit_gee(x,y,{7,7,7,7}), std::invalid_argument);
    const DenseMatrix<double> no_intercept(4,1,{-1,1,-1,1});
    const auto zero = fit_gee(no_intercept,{0,0,0,0},cluster,{.family = GEEFamily::Poisson});
    EXPECT_EQ(zero.status, GEEStatus::Converged);
    EXPECT_NEAR(zero.coefficients[0], 0, 1e-13);
}
}
