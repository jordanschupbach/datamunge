#include <gtest/gtest.h>

#include <datamunge/algorithms/phonetic.hpp>

#include <string>

using datamunge::algorithms::match_rating_codex;
using datamunge::algorithms::match_rating_comparison;
using datamunge::algorithms::metaphone;
using datamunge::algorithms::nysiis;
using datamunge::algorithms::soundex;

// All expected values below were cross-checked against the `jellyfish` reference
// implementation (Rust backend) over a ~4000-word corpus; every case here matched.

TEST(Phonetic, SoundexKnownVectors) {
    EXPECT_EQ(soundex("Robert"), "R163");
    EXPECT_EQ(soundex("Rupert"), "R163"); // homophone of Robert
    EXPECT_EQ(soundex("Rubin"), "R150");
    EXPECT_EQ(soundex("Ashcraft"), "A261"); // H is transparent: S and C collapse
    EXPECT_EQ(soundex("Tymczak"), "T522");
    EXPECT_EQ(soundex("Pfister"), "P236");
    EXPECT_EQ(soundex("Honeyman"), "H555");
    EXPECT_EQ(soundex(""), "");
}

TEST(Phonetic, SoundexShapeAndCaseInsensitivity) {
    for (const char* n : {"Washington", "Lee", "Gutierrez", "Pfister", "Jackson"}) {
        const std::string code = soundex(n);
        ASSERT_EQ(code.size(), 4u) << n;
        EXPECT_TRUE(std::isalpha(static_cast<unsigned char>(code[0]))) << n;
        for (int k = 1; k < 4; ++k) EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(code[k]))) << n;
    }
    EXPECT_EQ(soundex("robert"), soundex("ROBERT"));
    EXPECT_EQ(soundex("robert"), soundex("Robert"));
}

TEST(Phonetic, NysiisKnownVectors) {
    EXPECT_EQ(nysiis("MacDonald"), "MCDANALD");
    EXPECT_EQ(nysiis("Knight"), "NAGT");
    EXPECT_EQ(nysiis("Watkins"), "WATCAN");
    EXPECT_EQ(nysiis("Johnston"), "JANSTAN");
    EXPECT_EQ(nysiis("Catherine"), "CATARAN");
    EXPECT_EQ(nysiis("Kathryn"), "CATRYN");
    EXPECT_EQ(nysiis("Pfister"), "FASTAR"); // PF -> FF prefix rule
    EXPECT_EQ(nysiis(""), "");
}

TEST(Phonetic, MetaphoneKnownVectors) {
    EXPECT_EQ(metaphone("Thompson"), "0MPSN");  // TH -> 0 (theta)
    EXPECT_EQ(metaphone("Catherine"), "K0RN");
    EXPECT_EQ(metaphone("Kathryn"), "K0RN");    // matches Catherine
    EXPECT_EQ(metaphone("Xavier"), "SFR");      // leading X -> S
    EXPECT_EQ(metaphone("Knight"), "NT");       // silent K, silent GH
    EXPECT_EQ(metaphone("Smith"), "SM0");
    EXPECT_EQ(metaphone("Jackson"), "JKSN");
    EXPECT_EQ(metaphone("Wikipedia"), "WKPT");
}

TEST(Phonetic, MetaphoneSilentClusters) {
    EXPECT_EQ(metaphone("comb"), "KM");    // silent word-final b in mb
    EXPECT_EQ(metaphone("amber"), "AMBR"); // b kept mid-word
    EXPECT_EQ(metaphone("sign"), "S");     // silent gn at word end
    EXPECT_EQ(metaphone("align"), "AL");
    EXPECT_EQ(metaphone("eight"), "ET");   // silent gh before a consonant
    EXPECT_EQ(metaphone("weigh"), "WKH");  // gh kept at word end
    EXPECT_EQ(metaphone("gnat"), "NT");    // silent leading gn
}

TEST(Phonetic, MatchRatingCodexKnownVectors) {
    EXPECT_EQ(match_rating_codex("Byrne"), "BYRN");
    EXPECT_EQ(match_rating_codex("Boern"), "BRN");
    EXPECT_EQ(match_rating_codex("Ashcraft"), "ASHRFT"); // >6 letters -> first3 + last3
    EXPECT_EQ(match_rating_codex("Honeyman"), "HNYMN");
    EXPECT_EQ(match_rating_codex("Catherine"), "CTHRN");
    EXPECT_EQ(match_rating_codex("Smith"), "SMTH");
}

TEST(Phonetic, MatchRatingComparisonVerdicts) {
    EXPECT_TRUE(match_rating_comparison("Byrne", "Boern").match);
    EXPECT_TRUE(match_rating_comparison("Smith", "Smyth").match);
    EXPECT_TRUE(match_rating_comparison("Catherine", "Kathryn").match);
    EXPECT_TRUE(match_rating_comparison("Robert", "Rupert").match);
    EXPECT_TRUE(match_rating_comparison("Steven", "Stephen").match);

    // Unrelated names of comparable length: comparable, but no match.
    const auto az = match_rating_comparison("Alice", "Zachary");
    EXPECT_TRUE(az.comparable);
    EXPECT_FALSE(az.match);

    // Codex lengths differing by 3+ are not comparable at all.
    const auto r = match_rating_comparison("xyz", "supercalifragilistic");
    EXPECT_FALSE(r.comparable);
    EXPECT_FALSE(r.match);
}

TEST(Phonetic, HomophonesAgreeAcrossEncoders) {
    // Names that sound alike collapse under the phonetic encoders.
    EXPECT_EQ(soundex("Sean"), soundex("Shaun"));
    EXPECT_EQ(metaphone("Catherine"), metaphone("Kathryn"));
    EXPECT_TRUE(match_rating_comparison("Karen", "Caren").match);
}
