#include <gtest/gtest.h>

#include <datamunge/algorithms/todd_coxeter.hpp>

#include <vector>

using namespace datamunge::algorithms;

namespace {

// Build a word: `sym` repeated `k` times.
std::vector<int> pow_word(int sym, int k) {
    std::vector<int> w;
    for (int i = 0; i < k; ++i) w.push_back(sym);
    return w;
}

} // namespace

TEST(ToddCoxeter, CyclicGroups) {
    for (int n = 1; n <= 8; ++n) {
        // <a | a^n>
        const long long idx = todd_coxeter_index(1, {pow_word(1, n)}, {});
        EXPECT_EQ(idx, n) << "C_" << n;
    }
}

TEST(ToddCoxeter, DihedralGroups) {
    for (int n = 2; n <= 6; ++n) {
        // <a, b | a^n, b^2, (ab)^2>  -> order 2n
        const std::vector<std::vector<int>> rels = {pow_word(1, n), {2, 2}, {1, 2, 1, 2}};
        const long long                     idx  = todd_coxeter_index(2, rels, {});
        EXPECT_EQ(idx, 2 * n) << "D_" << n;
    }
}

TEST(ToddCoxeter, KleinFourAndTrivial) {
    // Klein four: <a, b | a^2, b^2, (ab)^2> -> order 4
    EXPECT_EQ(todd_coxeter_index(2, {{1, 1}, {2, 2}, {1, 2, 1, 2}}, {}), 4);
    // Trivial group: <a | a> -> order 1
    EXPECT_EQ(todd_coxeter_index(1, {{1}}, {}), 1);
}

TEST(ToddCoxeter, SymmetricGroupS4) {
    // Coxeter presentation of S_4: <a, b | a^2, b^3, (ab)^4> -> order 24
    const std::vector<std::vector<int>> rels = {{1, 1}, {2, 2, 2}, {1, 2, 1, 2, 1, 2, 1, 2}};
    EXPECT_EQ(todd_coxeter_index(2, rels, {}), 24);
}

TEST(ToddCoxeter, SymmetricGroupS3) {
    // S_3 = <a, b | a^2, b^3, (ab)^2> -> order 6
    EXPECT_EQ(todd_coxeter_index(2, {{1, 1}, {2, 2, 2}, {1, 2, 1, 2}}, {}), 6);
}

TEST(ToddCoxeter, SubgroupIndex) {
    // G = C_6 = <a | a^6>, H = <a^2>  ->  index [G:H] = 2
    EXPECT_EQ(todd_coxeter_index(1, {pow_word(1, 6)}, {{1, 1}}), 2);
    // G = C_6, H = <a^3>  ->  index 3
    EXPECT_EQ(todd_coxeter_index(1, {pow_word(1, 6)}, {{1, 1, 1}}), 3);
    // D_4 (order 8), H = <a> (the rotation subgroup, order 4) -> index 2
    EXPECT_EQ(todd_coxeter_index(2, {{1, 1, 1, 1}, {2, 2}, {1, 2, 1, 2}}, {{1}}), 2);
}
