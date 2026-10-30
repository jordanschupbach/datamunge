#include <gtest/gtest.h>

#include "datamunge/nn/build_info.hpp"
#include "datamunge/nn/parallel.hpp"

namespace nn = datamunge::nn;

TEST(NnBuildInfo, ReportsBackendsAndDtypes) {
    const auto info = nn::build_info();
    EXPECT_EQ(info.dtypes.size(), 5u);
    EXPECT_EQ(info.gemm_backend, nn::gemm_backend());
    EXPECT_EQ(info.threads, nn::get_num_threads());
    EXPECT_FALSE(info.compiler.empty());
    EXPECT_FALSE(info.version.empty());
    const auto md = info.to_markdown();
    EXPECT_NE(md.find("| GEMM backend | " + info.gemm_backend + " |"), std::string::npos) << md;
    EXPECT_NE(md.find("float64 (8 B)"), std::string::npos) << md;
}

TEST(NnBuildInfo, MatmulFlops) {
    EXPECT_DOUBLE_EQ(nn::matmul_flops(2, 3, 4), 48.0);
    EXPECT_DOUBLE_EQ(nn::matmul_flops(100000, 100000, 100000), 2e15);
}
