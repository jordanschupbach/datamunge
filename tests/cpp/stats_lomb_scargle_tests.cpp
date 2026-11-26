#include <datamunge/stats/lomb_scargle.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_lomb_scargle_data.inc"

void expect_relative(const std::vector<double>& got, const std::vector<double>& want, double tol) {
    ASSERT_EQ(got.size(),want.size());
    for (std::size_t i = 0; i < want.size(); ++i)
        EXPECT_NEAR(got[i],want[i],tol*std::max(std::abs(want[i]),1e-300)) << i;
}

const std::vector<double> grid = [] {
    std::vector<double> g;
    for (int i = 0; i < 40; ++i)
        g.push_back(.05+.95*i/39.0);
    return g;
}();

// References: astropy 7.2.0 LombScargle with method = "slow".
TEST(LombScargle, AutopowerAndPeakMatchAstropy) {
    const auto r = lomb_scargle(ls_t,ls_y);
    expect_relative(r.frequencies,ls_freq,1e-13);
    expect_relative(r.power,ls_power,1e-10);
    EXPECT_NEAR(r.best_frequency,ls_best[0],1e-14);
    EXPECT_NEAR(r.best_power,ls_best[1],1e-12);
    EXPECT_NEAR(r.false_alarm_probability,ls_fap[3],1e-8*ls_fap[3]);
    double mean = 0;
    for (const double v : ls_y)
        mean += v/double(ls_y.size());
    EXPECT_NEAR(r.offset,mean+ls_params[0],1e-10);
    EXPECT_NEAR(r.amplitude,std::hypot(ls_params[1],ls_params[2]),1e-10);
    const FalseAlarmMethod methods[] = {FalseAlarmMethod::single,FalseAlarmMethod::naive,FalseAlarmMethod::davies,
                                        FalseAlarmMethod::baluev};
    for (std::size_t m = 0; m < 4; ++m)
        EXPECT_NEAR(lomb_scargle_false_alarm_probability(r.best_power,ls_t,ls_y,{},{.method=methods[m]}),ls_fap[m],
                    1e-8*ls_fap[m]) << m;
    const FalseAlarmMethod inverse[] = {FalseAlarmMethod::single,FalseAlarmMethod::naive,FalseAlarmMethod::baluev};
    for (std::size_t m = 0; m < 3; ++m)
        EXPECT_NEAR(lomb_scargle_false_alarm_level(.01,ls_t,ls_y,{},{.method=inverse[m]}),ls_fal[m],1e-8) << m;
    expect_relative(lomb_scargle_model(ls_t,ls_y,{},r.best_frequency,{0,12.5,51}),ls_model,1e-10);
}

TEST(LombScargle, WeightedNormalizationsMatchAstropy) {
    const LombScargleNormalization norms[] = {LombScargleNormalization::standard,LombScargleNormalization::model,
                                              LombScargleNormalization::log,LombScargleNormalization::psd};
    const std::vector<double>* powers[] = {&lsw_standard,&lsw_model,&lsw_log,&lsw_psd};
    const std::vector<double>* faps[] = {&lsw_standard_fap,&lsw_model_fap,&lsw_log_fap,&lsw_psd_fap};
    const FalseAlarmMethod methods[] = {FalseAlarmMethod::single,FalseAlarmMethod::naive,FalseAlarmMethod::davies,
                                        FalseAlarmMethod::baluev};
    for (std::size_t k = 0; k < 4; ++k) {
        expect_relative(lomb_scargle_power(ls_t,ls_y,ls_dy,grid,{.normalization=norms[k]}),*powers[k],1e-10);
        const auto& f = *faps[k];
        for (std::size_t m = 0; m < 4; ++m)
            EXPECT_NEAR(lomb_scargle_false_alarm_probability(f[0],ls_t,ls_y,ls_dy,
                                                             {.normalization=norms[k],.maximum_frequency=1.0,
                                                              .method=methods[m]}),
                        f[m+1],1e-9*std::max(f[m+1],1e-300)) << k << " " << m;
    }
}

TEST(LombScargle, MeanHandlingAndGrid) {
    expect_relative(lomb_scargle_power(ls_t,ls_y,ls_dy,grid,{.fit_mean=false,.center_data=true}),ls_nofit_center,1e-10);
    expect_relative(lomb_scargle_power(ls_t,ls_y,ls_dy,grid,{.fit_mean=false,.center_data=false}),ls_nofit_raw,1e-10);
    expect_relative(lomb_scargle_model(ls_t,ls_y,ls_dy,.27,{3,7},{.fit_mean=false,.center_data=false}),ls_nofit_model,1e-10);
    const auto f = lomb_scargle_frequencies(ls_t,{.samples_per_peak=10,.minimum_frequency=.1,.maximum_frequency=.6});
    EXPECT_EQ(double(f.size()),ls_grid2[0]);
    EXPECT_NEAR(f.front(),ls_grid2[1],1e-15);
    EXPECT_NEAR(f.back(),ls_grid2[2],1e-13);
}

TEST(LombScargle, Validation) {
    EXPECT_THROW((void)lomb_scargle({1,2},{1,2}),std::invalid_argument);
    EXPECT_THROW((void)lomb_scargle({1,2,3},{1,2}),std::invalid_argument);
    EXPECT_THROW((void)lomb_scargle({1,1,1,1},{1,2,3,4}),std::invalid_argument);
    EXPECT_THROW((void)lomb_scargle(ls_t,ls_y,std::vector<double>(ls_y.size(),0.0)),std::invalid_argument);
    EXPECT_THROW((void)lomb_scargle_power(ls_t,ls_y,{},{0.0}),std::invalid_argument);
    EXPECT_THROW((void)lomb_scargle_false_alarm_probability(.3,ls_t,ls_y,{},{.fit_mean=false,.center_data=false}),
                 std::invalid_argument);
    EXPECT_THROW((void)lomb_scargle_false_alarm_level(1,ls_t,ls_y,{}),std::invalid_argument);
    EXPECT_THROW((void)lomb_scargle_frequencies(ls_t,{.minimum_frequency=.5,.maximum_frequency=.2}),std::invalid_argument);
}

} // namespace
