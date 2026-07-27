#include <gtest/gtest.h>

#include <datamunge/stats/bootstrap.hpp>
#include <datamunge/stats/jackknife.hpp>
#include <datamunge/stats/permutation_test.hpp>
#include <datamunge/stats/u_statistic.hpp>
#include <datamunge/stats/v_statistic.hpp>

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace datamunge::stats;

namespace {

const std::vector<double> kData = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}; // mean 5, textbook sample

double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}
double sum_sq_dev(const std::vector<double>& v) {
    const double m = mean_of(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return s;
}
double unbiased_var(const std::vector<double>& v) { return sum_sq_dev(v) / static_cast<double>(v.size() - 1); }
double biased_var(const std::vector<double>& v) { return sum_sq_dev(v) / static_cast<double>(v.size()); }

// Degree-2 symmetric kernel whose U-statistic is the unbiased variance and whose V-statistic is
// the biased (plug-in) variance.
const SymmetricKernel kVarKernel = [](const std::vector<double>& t) {
    const double d = t[0] - t[1];
    return 0.5 * d * d;
};

} // namespace

// ---- Bootstrap ----

TEST(Bootstrap, MeanEstimateExactAndStandardErrorMatchesPlugIn) {
    Bootstrap boot(BootstrapOptions{20000, 0.95, 7});
    const auto r = boot.run(kData, mean_of);
    EXPECT_NEAR(r.estimate, 5.0, 1e-12);                       // statistic on original data is exact
    // Bootstrap SE of the mean converges to the plug-in sigma_hat / sqrt(n) (sigma_hat^2 = biased var).
    const double plug_in_se = std::sqrt(biased_var(kData) / static_cast<double>(kData.size()));
    EXPECT_NEAR(r.standard_error, plug_in_se, 0.05);
    EXPECT_NEAR(r.bias, 0.0, 0.05);                            // the mean is unbiased under resampling
}

TEST(Bootstrap, IntervalsBracketTheEstimate) {
    Bootstrap boot(BootstrapOptions{5000, 0.90, 3});
    const auto r = boot.run(kData, mean_of);
    for (const auto& ci : {r.percentile_interval, r.basic_interval, r.normal_interval, r.bca_interval}) {
        EXPECT_LT(ci.first, r.estimate);
        EXPECT_GT(ci.second, r.estimate);
    }
    EXPECT_EQ(r.replicates.size(), 5000u);
}

TEST(Bootstrap, RejectsBadInput) {
    EXPECT_THROW(Bootstrap(BootstrapOptions{0, 0.95, 1}), std::invalid_argument);
    EXPECT_THROW(Bootstrap(BootstrapOptions{100, 1.5, 1}), std::invalid_argument);
    EXPECT_THROW(Bootstrap().run({}, mean_of), std::invalid_argument);
}

// ---- Jackknife ----

TEST(Jackknife, BiasCorrectsPlugInVarianceToUnbiasedVariance) {
    // The delete-1 jackknife of the plug-in (/n) variance is exactly the unbiased (/(n-1)) variance.
    const auto r = jackknife(kData, biased_var);
    EXPECT_NEAR(r.estimate, biased_var(kData), 1e-12);
    EXPECT_NEAR(r.bias_corrected, unbiased_var(kData), 1e-9);
}

TEST(Jackknife, MeanHasZeroBiasAndStandardErrorEqualsStdErrorOfMean) {
    const auto r = jackknife(kData, mean_of);
    EXPECT_NEAR(r.bias, 0.0, 1e-12);                                    // linear statistic: exact
    EXPECT_NEAR(r.standard_error, std::sqrt(unbiased_var(kData) / static_cast<double>(kData.size())), 1e-9);
}

TEST(Jackknife, DeleteDCompleteEnumeratesAllSubsets) {
    DeleteDJackknifeOptions opt;
    opt.d = 2;
    opt.max_subsets = 100000;
    const auto r = jackknife_delete_d(kData, mean_of, opt);
    EXPECT_TRUE(r.complete);
    EXPECT_EQ(r.num_subsets, 28u); // C(8,2)
    EXPECT_GT(r.standard_error, 0.0);
}

// ---- U-statistic ----

TEST(UStatistic, PairwiseKernelEqualsUnbiasedVariance) {
    UStatistic u(2);
    const auto r = u.run(kData, kVarKernel);
    EXPECT_TRUE(r.complete);
    EXPECT_EQ(r.num_terms, 28u); // C(8,2)
    EXPECT_NEAR(r.estimate, unbiased_var(kData), 1e-9);
    EXPECT_GT(r.standard_error, 0.0);
}

TEST(UStatistic, DegreeOneKernelEqualsMean) {
    UStatistic u(1);
    const auto r = u.run(kData, [](const std::vector<double>& t) { return t[0]; });
    EXPECT_NEAR(r.estimate, 5.0, 1e-12);
}

TEST(UStatistic, IncompleteApproximatesComplete) {
    UStatistic u(2, UStatisticOptions{500, 11});
    const auto r = u.run(kData, kVarKernel);
    EXPECT_FALSE(r.complete);
    EXPECT_EQ(r.num_terms, 500u);
    EXPECT_NEAR(r.estimate, unbiased_var(kData), 1.0); // Monte Carlo: close, not exact
}

// ---- V-statistic ----

TEST(VStatistic, PairwiseKernelEqualsBiasedVarianceAndRelatesToU) {
    VStatistic v(2);
    const auto r = v.run(kData, kVarKernel);
    EXPECT_TRUE(r.exact);
    EXPECT_NEAR(r.estimate, biased_var(kData), 1e-9);
    // V_n = (n-1)/n * U_n for this kernel.
    const double n = static_cast<double>(kData.size());
    EXPECT_NEAR(r.estimate, (n - 1.0) / n * unbiased_var(kData), 1e-9);
}

// ---- Permutation test ----

TEST(PermutationTest, SeparatedSamplesRejectAndSimilarSamplesDoNot) {
    const std::vector<double> a = {1, 2, 3, 2, 1, 3, 2};
    const std::vector<double> far = {20, 21, 19, 22, 20, 21};
    const auto sep = permutation_test_two_sample(a, far, {}, PermutationTestOptions{5000, Alternative::TwoSided, 1});
    EXPECT_LT(sep.p_value, 0.01);
    EXPECT_NEAR(sep.observed, mean_of(a) - mean_of(far), 1e-9);

    const std::vector<double> b = {1, 3, 2, 2, 1, 3, 3}; // same rough distribution as a
    const auto same = permutation_test_two_sample(a, b, {}, PermutationTestOptions{5000, Alternative::TwoSided, 1});
    EXPECT_GT(same.p_value, 0.05);
}

TEST(PermutationTest, RejectsEmptySample) {
    EXPECT_THROW(permutation_test_two_sample({}, {1.0}, {}, {}), std::invalid_argument);
}
