#include <datamunge/stats/factorial_design.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

std::vector<std::string> joined(const std::vector<std::vector<std::string>>& sets) {
    std::vector<std::string> out;
    for (const auto& s : sets) {
        std::string line = s[0];
        for (std::size_t i = 1; i < s.size(); ++i)
            line += "="+s[i];
        out.push_back(line);
    }
    return out;
}

// Reference values: FrF2 2.3 FrF2(nruns, nfactors, generators, randomize = FALSE), desnum() and
// design.info()$aliased in R 4.5.3.
TEST(FactorialDesign, ResolutionIVDesignMatchesFrF2) {
    const auto d = fractional_factorial_design(7,{"ABC","BCD","ACD"});
    const std::vector<double> frf2{
        -1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,
        -1,1,-1,-1,1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,-1,1,1,-1,1,1,1,-1,
        1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,1,1,1,1};
    ASSERT_EQ(d.runs.rows(),16U);
    ASSERT_EQ(d.runs.cols(),7U);
    for (std::size_t r = 0; r < 16; ++r)
        for (std::size_t c = 0; c < 7; ++c)
            EXPECT_EQ(d.runs(r,c),frf2[r*7+c]) << r << "," << c;
    EXPECT_EQ(d.resolution,4U);
    EXPECT_EQ(d.word_length_pattern,(std::vector<std::size_t>{0,7,0,0,0}));
    EXPECT_EQ(d.generators,(std::vector<std::string>{"E=ABC","F=BCD","G=ACD"}));
    EXPECT_EQ(joined(alias_sets(d)),(std::vector<std::string>{"AB=CE=FG","AC=BE=DG","AD=CG=EF","AE=BC=DF",
                                                               "AF=BG=DE","AG=BF=CD","BD=CF=EG"}));
}

TEST(FactorialDesign, ResolutionIIIAliasesMainEffectsWithInteractions) {
    const auto d = fractional_factorial_design(5,{"AB","AC"});
    const std::vector<double> frf2{-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,
                                   -1,-1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,1,1,1,1,1};
    for (std::size_t r = 0; r < 8; ++r)
        for (std::size_t c = 0; c < 5; ++c)
            EXPECT_EQ(d.runs(r,c),frf2[r*5+c]);
    EXPECT_EQ(d.resolution,3U);
    EXPECT_EQ(d.defining_relation,(std::vector<std::string>{"ABD","ACE","BCDE"}));
    EXPECT_EQ(joined(alias_sets(d)),(std::vector<std::string>{"A=BD=CE","B=AD","C=AE","D=AB","E=AC","BC=DE",
                                                               "BE=CD"}));
    // Main effects only: nothing is aliased among them in a resolution III design.
    EXPECT_TRUE(alias_sets(d,1).empty());
}

TEST(FactorialDesign, FullFactorialIsBalancedAndOrthogonal) {
    const auto d = full_factorial_design(4);
    EXPECT_EQ(d.runs.rows(),16U);
    EXPECT_EQ(d.resolution,0U);
    EXPECT_TRUE(d.defining_relation.empty());
    EXPECT_TRUE(alias_sets(d,4).empty());
    for (std::size_t a = 0; a < 4; ++a)
        for (std::size_t b = 0; b < 4; ++b) {
            double dot = 0;
            for (std::size_t r = 0; r < 16; ++r)
                dot += d.runs(r,a)*d.runs(r,b);
            EXPECT_EQ(dot,a == b ? 16.0 : 0.0);
        }
    EXPECT_EQ(d.factors[3],"D");
    EXPECT_EQ(full_factorial_design(9).factors[8],"J"); // I is skipped.
}

TEST(FactorialDesign, RejectsInvalidGenerators) {
    EXPECT_THROW((void)full_factorial_design(0),std::invalid_argument);
    EXPECT_THROW((void)fractional_factorial_design(4,{}),std::invalid_argument);
    EXPECT_THROW((void)fractional_factorial_design(4,{"ABD"}),std::invalid_argument);
    EXPECT_THROW((void)fractional_factorial_design(4,{"A"}),std::invalid_argument);
    EXPECT_THROW((void)fractional_factorial_design(4,{"AAB"}),std::invalid_argument);
    EXPECT_THROW((void)fractional_factorial_design(5,{"AB","BA"}),std::invalid_argument);
    EXPECT_THROW((void)alias_sets(full_factorial_design(3),4),std::invalid_argument);
}

} // namespace
