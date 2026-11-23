#include <datamunge/stats/nested_cross_validation.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::NestedCandidate;
using datamunge::stats::nested_cross_validate;

TEST(NestedCrossValidation, SelectsOnInnerRowsAndRefitsWithoutOuterOutcomes) {
    const DenseMatrix<double> x(7,1,{0,1,2,3,4,5,6});
    const std::vector<double> y{0,0,0,1,1,1,.2};
    const std::vector<std::vector<std::size_t>> held_out{{0,1,2},{3,4},{5,6}};
    std::size_t calls = 0;
    const auto candidate = [&](double prediction) {
        return [&,prediction](const auto& train,const auto& response,const auto& query) {
            const auto outer = calls/5;
            const bool final = calls%5 == 4;
            ++calls;
            EXPECT_EQ(train.rows(),response.size());
            for (std::size_t i = 0; i < train.rows(); ++i) {
                const auto row = static_cast<std::size_t>(train(i,0));
                EXPECT_DOUBLE_EQ(response[i],y[row]);
                EXPECT_FALSE(std::ranges::contains(held_out[outer],row));
                EXPECT_FALSE(std::ranges::contains(query.data(),static_cast<double>(row)));
            }
            for (auto row : query.data())
                EXPECT_EQ(std::ranges::contains(held_out[outer],static_cast<std::size_t>(row)),final);
            return std::vector<double>(query.rows(),prediction);
        };
    };
    const auto fit = nested_cross_validate(x,y,{{"zero",candidate(0)},{"one",candidate(1)}},
        {.outer_folds=3,.inner_folds=2,.shuffle=false});
    ASSERT_EQ(fit.folds.size(),3u);
    EXPECT_EQ(calls,15u);
    EXPECT_EQ(fit.predictions,(std::vector<double>{1,1,1,0,0,0,0}));
    EXPECT_EQ(fit.folds[0].test_rows,held_out[0]);
    EXPECT_EQ(fit.folds[0].selected_name,"one");
    EXPECT_EQ(fit.folds[1].selected_name,"zero");
    EXPECT_EQ(fit.folds[2].selected_name,"zero");
    EXPECT_NEAR(fit.folds[0].inner_loss[1],.16,1e-14);
    EXPECT_NEAR(fit.folds[1].inner_loss[0],.208,1e-14);
    EXPECT_NEAR(fit.folds[2].test_loss,.52,1e-14);
    EXPECT_NEAR(fit.mean_loss,6.04/7,1e-14);
}

TEST(NestedCrossValidation, TiesChooseFirstAndShuffleIsReproducible) {
    const DenseMatrix<double> x(10,1,{0,1,2,3,4,5,6,7,8,9});
    const auto candidate = [](const auto&,const auto&,const auto& query) {
        return query.data();
    };
    const std::vector<NestedCandidate> candidates{{"first",candidate},{"second",candidate}};
    const auto controls = datamunge::stats::NestedCVOptions{.outer_folds=3,.inner_folds=3,.seed=20};
    const auto first = nested_cross_validate(x,x.data(),candidates,controls,
        [](double observed,double prediction) { return observed == prediction ? 0.0 : 1.0; });
    const auto second = nested_cross_validate(x,x.data(),candidates,controls);
    for (std::size_t i = 0; i < first.folds.size(); ++i) {
        EXPECT_EQ(first.folds[i].selected_index,0u);
        EXPECT_EQ(first.folds[i].test_rows,second.folds[i].test_rows);
    }
    EXPECT_EQ(first.predictions,x.data());
    EXPECT_DOUBLE_EQ(first.mean_loss,0);
}

TEST(NestedCrossValidation, RejectsInvalidFoldsAndCandidatePredictions) {
    const DenseMatrix<double> x(6,1,1);
    const std::vector<double> y(6,1);
    const NestedCandidate correct{"constant",[](const auto&,const auto&,const auto& query) {
        return std::vector<double>(query.rows(),1);
    }};
    const NestedCandidate wrong_size{"wrong",[](const auto&,const auto&,const auto&) {
        return std::vector<double>{};
    }};
    const NestedCandidate nonfinite{"invalid",[](const auto&,const auto&,const auto& query) {
        return std::vector<double>(query.rows(),std::numeric_limits<double>::infinity());
    }};
    EXPECT_THROW((void)nested_cross_validate(x,y,{correct},{.outer_folds=2,.inner_folds=4}),
        std::invalid_argument);
    EXPECT_THROW((void)nested_cross_validate(x,y,{wrong_size},{.outer_folds=3,.inner_folds=2}),
        std::invalid_argument);
    EXPECT_THROW((void)nested_cross_validate(x,y,{nonfinite},{.outer_folds=3,.inner_folds=2}),
        std::domain_error);
}
}
