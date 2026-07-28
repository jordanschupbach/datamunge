#include <gtest/gtest.h>

#include <datamunge/algorithms/knuth_bendix.hpp>

#include <algorithm>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

namespace {
using Eq = std::pair<std::string, std::string>;
}

TEST(KnuthBendix, CommutingMonoid) {
    // <a, b | ab = ba> : normal forms sort a's before b's.
    const auto res = knuth_bendix({{"ab", "ba"}});
    EXPECT_TRUE(res.complete);
    EXPECT_EQ(kb_normal_form("ba", res.rules), "ab");
    EXPECT_EQ(kb_normal_form("baba", res.rules), "aabb");
    EXPECT_EQ(kb_normal_form("bbaa", res.rules), "aabb");
    // Word problem: two words are equal iff they have the same letter multiset.
    EXPECT_EQ(kb_normal_form("abab", res.rules), kb_normal_form("bbaa", res.rules));
    EXPECT_NE(kb_normal_form("aab", res.rules), kb_normal_form("abb", res.rules));
}

TEST(KnuthBendix, CyclicGroupOrder2) {
    // <a | aa = e> : normal forms are "" and "a".
    const auto res = knuth_bendix({{"aa", ""}});
    EXPECT_TRUE(res.complete);
    EXPECT_EQ(kb_normal_form("aaa", res.rules), "a");
    EXPECT_EQ(kb_normal_form("aaaa", res.rules), "");
    EXPECT_EQ(kb_normal_form("aaaaa", res.rules), "a");
}

TEST(KnuthBendix, KleinFourGroup) {
    // <a, b | a^2 = e, b^2 = e, (ab)^2 = e>  is the Klein four-group (abelian, order 4).
    const auto res = knuth_bendix({{"aa", ""}, {"bb", ""}, {"abab", ""}});
    EXPECT_TRUE(res.complete);

    // Enumerate normal forms of all words up to length 5: exactly 4 distinct.
    std::vector<std::string> forms;
    std::string              alphabet = "ab";
    std::vector<std::string> frontier = {""};
    for (int len = 0; len <= 5; ++len) {
        std::vector<std::string> next;
        for (const auto& w : frontier) {
            forms.push_back(kb_normal_form(w, res.rules));
            for (char c : alphabet) next.push_back(w + c);
        }
        frontier = next;
    }
    std::sort(forms.begin(), forms.end());
    forms.erase(std::unique(forms.begin(), forms.end()), forms.end());
    EXPECT_EQ(forms.size(), 4u);

    // The group is abelian: ab = ba.
    EXPECT_EQ(kb_normal_form("ab", res.rules), kb_normal_form("ba", res.rules));
    // a and b are involutions.
    EXPECT_EQ(kb_normal_form("aa", res.rules), "");
    EXPECT_EQ(kb_normal_form("bb", res.rules), "");
}

TEST(KnuthBendix, ConfluenceIdentityWords) {
    // After completion, every word for the identity reduces to "".
    const auto res = knuth_bendix({{"aa", ""}, {"bb", ""}, {"abab", ""}});
    for (const std::string& e : {"aa", "bb", "abab", "baba", "abababab", "aabb", "aaaa", "bbbb", "abba"})
        EXPECT_EQ(kb_normal_form(e, res.rules), "") << e;
    // ab and ba are the same (non-identity) element.
    EXPECT_EQ(kb_normal_form("ab", res.rules), kb_normal_form("ba", res.rules));
    EXPECT_NE(kb_normal_form("ab", res.rules), "");
}
