#include <datamunge/stats/stability_selection.hpp>

#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::StabilitySelector;
using datamunge::stats::stability_selection;

TEST(StabilitySelection, ReusesRowsAcrossLevelsAndMatchesIndependentCounts) {
    DenseMatrix<double> x(10,3);
    std::vector<double> y(10);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        x(i,0) = i;
        y[i] = 2*i;
    }
    const std::vector<double> levels{.8,.2,2.0};
    std::vector<double> draw;
    std::size_t calls = 0, with_first = 0;
    const StabilitySelector selector = [&](const auto& rows,const auto& response,double level) {
        EXPECT_DOUBLE_EQ(level,levels[calls%3]);
        if (calls%3 == 0) {
            draw = response;
            with_first += rows(0,0) == 0;
        }
        ++calls;
        EXPECT_EQ(draw,response);
        for (std::size_t i = 0; i < rows.rows(); ++i) {
            EXPECT_DOUBLE_EQ(response[i],2*rows(i,0));
            if (i > 0)
                EXPECT_LT(rows(i-1,0),rows(i,0));
        }
        return level == .2 ? std::vector<std::size_t>{0,2}
             : level == .8 && rows(0,0) == 0 ? std::vector<std::size_t>{1,0}
             : std::vector<std::size_t>{0};
    };
    const auto fit = stability_selection(x,y,levels,selector,
        {.subsamples=12,.selection_threshold=1,.seed=20});
    ASSERT_EQ(calls,36u);
    EXPECT_EQ(fit.sample_size,5u);
    EXPECT_EQ(fit.levels,levels);
    EXPECT_EQ(fit.counts(0,1),with_first);
    EXPECT_DOUBLE_EQ(fit.frequencies(0,1),static_cast<double>(with_first)/12);
    EXPECT_EQ(fit.counts(1,2),12u);
    EXPECT_EQ(fit.counts(2,2),0u);
    EXPECT_EQ(fit.stable_features,(std::vector<std::size_t>{0,2}));
}

TEST(StabilitySelection, ReportsZeroFrequencyAndReproducibleSubsamples) {
    const DenseMatrix<double> x(5,2,{0,1,1,2,2,3,3,4,4,5});
    const auto selector = [](const auto& x,const auto&,double) {
        return x(0,0) == 0 ? std::vector<std::size_t>{0} : std::vector<std::size_t>{};
    };
    const auto one = stability_selection(x,{0,1,2,3,4},{1,2},selector,{.subsamples=40,.seed=123});
    const auto two = stability_selection(x,{0,1,2,3,4},{1,2},selector,{.subsamples=40,.seed=123});
    EXPECT_EQ(one.sample_size,2u);
    EXPECT_EQ(one.counts.data(),two.counts.data());
    EXPECT_DOUBLE_EQ(one.maximum_frequency[1],0);
    EXPECT_EQ(one.counts(0,0),one.counts(1,0));
    EXPECT_GT(one.counts(0,0),0u);
    EXPECT_LT(one.counts(0,0),40u);
}

TEST(StabilitySelection, RejectsInvalidLabelsGridAndControls) {
    const DenseMatrix<double> x(4,2,0);
    const std::vector<double> y(4);
    const auto empty = [](const auto&,const auto&,double) { return std::vector<std::size_t>{}; };
    const auto duplicate = [](const auto&,const auto&,double) {
        return std::vector<std::size_t>{0,0};
    };
    const auto bounds = [](const auto&,const auto&,double) { return std::vector<std::size_t>{2}; };
    EXPECT_THROW((void)stability_selection(x,y,{1,1},empty),std::invalid_argument);
    EXPECT_THROW((void)stability_selection(x,y,{1},duplicate),std::invalid_argument);
    EXPECT_THROW((void)stability_selection(x,y,{1},bounds),std::invalid_argument);
    EXPECT_THROW((void)stability_selection(x,y,{1},empty,{.sample_fraction=.1}),std::invalid_argument);
    EXPECT_THROW((void)stability_selection(x,y,{1},empty,{.selection_threshold=.5}),
        std::invalid_argument);
}
}
