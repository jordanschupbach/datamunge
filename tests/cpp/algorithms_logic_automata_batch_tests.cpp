#include <gtest/gtest.h>

#include <datamunge/algorithms/automaton.hpp>
#include <datamunge/algorithms/dfa_minimization.hpp>
#include <datamunge/algorithms/petrick.hpp>
#include <datamunge/algorithms/powerset_construction.hpp>
#include <datamunge/algorithms/quine_mccluskey.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

// ---- Powerset construction --------------------------------------------------

namespace {
// NFA over {0,1} accepting strings whose last symbol is 1.
NFA ends_in_one() {
    NFA n;
    n.num_states  = 2;
    n.num_symbols = 2;
    n.delta       = {{{0}, {0, 1}}, {{}, {}}};
    n.start_states = {0};
    n.accept       = {0, 1};
    return n;
}
}  // namespace

TEST(Powerset, DeterminizesAndPreservesLanguage) {
    DFA d = powerset_construction(ends_in_one());
    EXPECT_TRUE(dfa_accepts(d, {1}));
    EXPECT_TRUE(dfa_accepts(d, {0, 1}));
    EXPECT_TRUE(dfa_accepts(d, {1, 1}));
    EXPECT_FALSE(dfa_accepts(d, {0}));
    EXPECT_FALSE(dfa_accepts(d, {1, 0}));
    EXPECT_FALSE(dfa_accepts(d, {}));
}

// ---- DFA minimization -------------------------------------------------------

namespace {
// A 4-state DFA for "ends in 1" with a redundant duplicate of each state.
DFA redundant_ends_in_one() {
    DFA d;
    d.num_symbols = 2;
    d.num_states  = 4;
    d.start       = 0;
    d.delta       = {{2, 1}, {2, 1}, {0, 3}, {0, 3}};  // states {0,2} and {1,3} are equivalent
    d.accept      = {0, 1, 0, 1};
    return d;
}

bool same_language(const DFA& a, const DFA& b, int max_len) {
    for (int len = 0; len <= max_len; ++len) {
        std::vector<int> w(len, 0);
        while (true) {
            if (dfa_accepts(a, w) != dfa_accepts(b, w)) return false;
            int i = len - 1;
            while (i >= 0 && w[i] == a.num_symbols - 1) { w[i] = 0; --i; }
            if (i < 0) break;
            ++w[i];
        }
    }
    return true;
}
}  // namespace

TEST(DFAMinimization, AllThreeAgree) {
    DFA r  = redundant_ends_in_one();
    DFA mo = moore_minimize(r);
    DFA hp = hopcroft_minimize(r);
    DFA bz = brzozowski_minimize(r);

    EXPECT_EQ(mo.num_states, 2);
    EXPECT_EQ(hp.num_states, 2);
    EXPECT_EQ(bz.num_states, 2);

    EXPECT_TRUE(same_language(r, mo, 6));
    EXPECT_TRUE(same_language(r, hp, 6));
    EXPECT_TRUE(same_language(r, bz, 6));
}

TEST(DFAMinimization, AlreadyMinimalUnchangedSize) {
    DFA d = powerset_construction(ends_in_one(), /*complete=*/true);
    DFA m = hopcroft_minimize(d);
    EXPECT_EQ(m.num_states, 2);  // "ends in 1" needs exactly 2 states
    EXPECT_TRUE(same_language(d, m, 6));
}

// ---- Petrick's method -------------------------------------------------------

TEST(Petrick, SelectsMinimumCover) {
    // Minterms 0,1,2 covered by PIs: m0 by {0}, m1 by {0,1}, m2 by {1}.
    // Minimum cover must include PI 0 (only cover of m0) and PI 1 (only cover of m2).
    PetrickResult r = petrick({{0}, {0, 1}, {1}});
    ASSERT_FALSE(r.minimal_covers.empty());
    EXPECT_EQ(r.minimal_covers.front().size(), 2u);
    std::vector<int> cover = r.minimal_covers.front();
    std::sort(cover.begin(), cover.end());
    EXPECT_EQ(cover, (std::vector<int>{0, 1}));
}

// ---- Quine-McCluskey --------------------------------------------------------

TEST(QuineMcCluskey, ClassicFourVariableExample) {
    // f(A,B,C,D) = sum m(4,8,10,11,12,15) + d(9,14).
    auto qm = quine_mccluskey(4, {4, 8, 10, 11, 12, 15}, {9, 14});
    // Known minimum SOP has three product terms.
    EXPECT_EQ(qm.minimal_cover.size(), 3u);

    // Verify the cover reproduces the function on all 16 inputs (don't-cares free).
    std::vector<std::uint32_t> ones = {4, 8, 10, 11, 12, 15};
    std::vector<std::uint32_t> dc   = {9, 14};
    for (std::uint32_t x = 0; x < 16; ++x) {
        bool covered = false;
        for (const auto& imp : qm.minimal_cover)
            if (imp.covers(x)) { covered = true; break; }
        bool is_one = std::find(ones.begin(), ones.end(), x) != ones.end();
        bool is_dc  = std::find(dc.begin(), dc.end(), x) != dc.end();
        if (is_one) EXPECT_TRUE(covered) << "minterm " << x << " uncovered";
        else if (!is_dc) EXPECT_FALSE(covered) << "off-set input " << x << " wrongly covered";
    }
}

TEST(QuineMcCluskey, SingleVariableFunction) {
    // f(A,B) = A (minterms 2,3) -> one prime implicant "1-".
    auto qm = quine_mccluskey(2, {2, 3});
    ASSERT_EQ(qm.minimal_cover.size(), 1u);
    EXPECT_EQ(implicant_to_string(qm.minimal_cover[0], 2), "1-");
    EXPECT_EQ(qm.essential.size(), 1u);
}
