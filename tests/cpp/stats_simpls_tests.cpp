#include <datamunge/stats/simpls.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::cross_validate_simpls;
using datamunge::stats::fit_simpls;

struct Data {
    DenseMatrix<double> x{28,5};
    DenseMatrix<double> y{28,2};
    std::vector<std::int64_t> fold;
};

Data sample() {
    Data data;
    for (std::size_t row = 0; row < data.x.rows(); ++row) {
        const int i = static_cast<int>(row)+1;
        const auto a = std::sin(.33*i), b = std::cos(.27*i);
        data.x(row,0) = a+.35*std::cos(1.9*i);
        data.x(row,1) = 2.8*a+.25*std::sin(1.53*i);
        data.x(row,2) = b+.3*std::sin(1.27*i);
        data.x(row,3) = .57*std::cos(2.3*i);
        data.x(row,4) = .63*std::sin(2.37*i);
        data.y(row,0) = 4+1.2*a-.7*b+.48*std::cos(2.07*i);
        data.y(row,1) = 2-.7*a+1.05*b+.42*std::sin(2.04*i);
        data.fold.push_back(i <= 7 ? -3 : (i > 26 ? 5 : i%4));
    }
    return data;
}

TEST(SIMPLS, MatchesRMultiresponsePathsInOriginalAndStandardizedUnits) {
    const auto data = sample();
    for (const bool scaled : {false,true}) {
        const auto fit = fit_simpls(data.x,data.y,5,{.scale_predictors=scaled,.scale_responses=scaled});
        ASSERT_EQ(fit.components(),5u);
        const auto first = fit.coefficients(1);
        const auto third = fit.coefficients(3);
        const auto fitted = fit.predict(data.x,3);
        EXPECT_NEAR(first(1,0),scaled ? .0852169996753399 : .197953320302559577,1e-13);
        EXPECT_NEAR(third(0,1),scaled ? -.1621004976399743 : .3325261735820769,2e-12);
        EXPECT_NEAR(third(2,1),scaled ? .8344938201886428 : .8837471044328183,2e-12);
        EXPECT_NEAR(fitted(0,0),scaled ? 3.65957008486582 : 3.75528356243767,1e-12);
        EXPECT_NEAR(fitted(0,1),scaled ? 2.89678215766420 : 2.76925518349572,1e-12);
        EXPECT_NEAR(fit.y_loadings(1,1),scaled ? 2.24048053640532 : 2.45448576509779,1e-12);
        EXPECT_NEAR(fit.score_weights(2,1),scaled ? .3313746502969420 : .35194305300180284,1e-12);

        const auto orthogonality = fit.scores.transpose()*fit.scores;
        for (std::size_t k = 0; k < fit.components(); ++k) {
            EXPECT_NEAR(orthogonality(k,k),1,1e-13);
            for (std::size_t j = 0; j < k; ++j)
                EXPECT_NEAR(orthogonality(k,j),0,1e-13);
            for (std::size_t i = 0; i < data.x.rows(); ++i) {
                double score = 0;
                for (std::size_t j = 0; j < data.x.cols(); ++j)
                    score += (data.x(i,j)-fit.predictor_means[j])*fit.score_weights(j,k);
                EXPECT_NEAR(score,fit.scores(i,k),1e-13);
            }
        }
    }
}

TEST(SIMPLS, UnivariatePathReachesLeastSquaresAndSupportsNullModel) {
    const auto data = sample();
    const DenseMatrix<double> y(data.y.rows(),1,data.y.col(0));
    const auto fit = fit_simpls(data.x,y,300);
    ASSERT_EQ(fit.components(),5u);
    EXPECT_NEAR(fit.coefficients(1)(1,0),.202897088654473423,1e-13);
    const auto final = fit.coefficients();
    const double expected[]{-.217727980608108,.481027982754753,-.628491942413509,-.252063097232947,.285461371362619};
    for (std::size_t j = 0; j < 5; ++j)
        EXPECT_NEAR(final(j,0),expected[j],1e-12);
    EXPECT_DOUBLE_EQ(fit.coefficients(0)(2,0),0);
    EXPECT_DOUBLE_EQ(fit.predict(data.x,0)(3,0),fit.response_means[0]);
}

TEST(SIMPLS, PooledHeldOutLossMatchesRFoldTrainingWithUnequalSizes) {
    const auto data = sample();
    const double expected[2][6]{
        {.424538731425361,.328618825476161,.168108136364020,.193060604265025,.199042099460994,.202881829757464},
        {.424538731425361,.354810517021895,.215656446563979,.235210306458469,.222336112986793,.202881829757464}};
    for (const bool scaled : {false,true}) {
        const auto cv = cross_validate_simpls(data.x,data.y,data.fold,5,
            {.scale_predictors=scaled,.scale_responses=scaled});
        ASSERT_EQ(cv.mean_squared_error.size(),6u);
        for (std::size_t k = 0; k < cv.mean_squared_error.size(); ++k)
            EXPECT_NEAR(cv.mean_squared_error[k],expected[scaled][k],1e-12);
        EXPECT_EQ(cv.selected_components,scaled ? 5u : 2u);
        EXPECT_EQ(cv.fit.components(),cv.selected_components);
        const auto full = fit_simpls(data.x,data.y,cv.selected_components,
            {.scale_predictors=scaled,.scale_responses=scaled});
        EXPECT_NEAR(cv.fit.coefficients()(1,1),full.coefficients()(1,1),1e-14);
    }
}

TEST(SIMPLS, ShortensDegeneratePathsAndValidatesInputAndPartition) {
    const DenseMatrix<double> x(6,2,{1,2,2,4,3,6,4,8,5,10,6,12});
    const DenseMatrix<double> y(6,1,{2,4,6,8,10,12}), constant(6,1,3);
    ASSERT_EQ(fit_simpls(x,y,4).components(),1u);
    EXPECT_EQ(fit_simpls(x,constant,4).components(),0u);
    const auto cv = cross_validate_simpls(x,constant,{-1,4,4,7,-1,7},3);
    EXPECT_EQ(cv.mean_squared_error,(std::vector<double>{0}));
    EXPECT_EQ(cv.selected_components,0u);
    EXPECT_DOUBLE_EQ(cv.fit.predict(x)(0,0),3);
    EXPECT_THROW((void)cross_validate_simpls(x,y,{1,1,1,1,1,2},2),std::invalid_argument);
    EXPECT_THROW((void)cross_validate_simpls(x,y,{1,1,1,1,1,1},2),std::invalid_argument);
    EXPECT_THROW((void)fit_simpls(x,y,1,{.tolerance=0}),std::invalid_argument);
    auto invalid = y;
    invalid(0,0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)fit_simpls(x,invalid,1),std::invalid_argument);
    const auto fit = fit_simpls(x,y,1);
    EXPECT_THROW((void)fit.predict(x,2),std::invalid_argument);
}
} // namespace
