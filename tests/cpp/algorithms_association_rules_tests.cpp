#include <gtest/gtest.h>

#include <datamunge/algorithms/association_rules.hpp>

#include <cstddef>
#include <map>
#include <string>
#include <vector>

using datamunge::algorithms::apriori_frequent_itemsets;
using datamunge::algorithms::association_rules;
using datamunge::algorithms::eclat_frequent_itemsets;
using datamunge::algorithms::fp_growth_frequent_itemsets;
using datamunge::algorithms::FrequentItemset;

namespace {

// A small market-basket database. Items: 0=bread 1=milk 2=butter 3=beer 4=diapers.
std::vector<std::vector<std::size_t>> groceries() {
    return {
        {0, 1, 2},     // bread, milk, butter
        {0, 1},        // bread, milk
        {1, 2},        // milk, butter
        {0, 1, 2},     // bread, milk, butter
        {3, 4},        // beer, diapers
        {0, 3, 4},     // bread, beer, diapers
        {3, 4},        // beer, diapers
        {0, 1, 2, 3},  // bread, milk, butter, beer
    };
}

// Turn a result into a comparable {sorted-items -> support} map.
std::map<std::vector<std::size_t>, std::size_t> to_map(const std::vector<FrequentItemset>& v) {
    std::map<std::vector<std::size_t>, std::size_t> m;
    for (const auto& fs : v) m[fs.items] = fs.support;
    return m;
}

}  // namespace

TEST(AssociationRules, ThreeMinersAgree) {
    const auto txns = groceries();
    for (std::size_t min_sup = 1; min_sup <= 4; ++min_sup) {
        const auto a = to_map(apriori_frequent_itemsets(txns, min_sup));
        const auto e = to_map(eclat_frequent_itemsets(txns, min_sup));
        const auto f = to_map(fp_growth_frequent_itemsets(txns, min_sup));
        EXPECT_EQ(a, e) << "apriori vs eclat disagree at min_support=" << min_sup;
        EXPECT_EQ(a, f) << "apriori vs fp-growth disagree at min_support=" << min_sup;
    }
}

TEST(AssociationRules, KnownSupports) {
    const auto m = to_map(apriori_frequent_itemsets(groceries(), 2));
    // beer(3) & diapers(4) occur together in transactions 4,5,6 -> 3 (t7 has beer but no diapers).
    EXPECT_EQ(m.at((std::vector<std::size_t>{3, 4})), 3u);
    // bread(0), milk(1), butter(2) together: transactions 0,3,7 -> 3.
    EXPECT_EQ(m.at((std::vector<std::size_t>{0, 1, 2})), 3u);
    // milk(1) alone: transactions 0,1,2,3,7 -> 5.
    EXPECT_EQ(m.at((std::vector<std::size_t>{1})), 5u);
    // {beer, diapers} present but {beer, diapers} with support 4; a rare pair absent below threshold.
    EXPECT_EQ(m.count(std::vector<std::size_t>{2, 4}), 0u);  // butter & diapers never co-occur
}

TEST(AssociationRules, RuleConfidenceAndLift) {
    const auto txns = groceries();
    const auto freq = apriori_frequent_itemsets(txns, 2);
    const auto rules = association_rules(freq, txns.size(), 0.9);

    // diapers(4) -> beer(3): diapers always comes with beer, so confidence 1.0 (support 3).
    // (The reverse beer -> diapers is only 3/4 = 0.75 and is filtered out below 0.9.)
    bool found = false;
    for (const auto& r : rules)
        if (r.antecedent == std::vector<std::size_t>{4} && r.consequent == std::vector<std::size_t>{3}) {
            EXPECT_NEAR(r.confidence, 1.0, 1e-12);
            EXPECT_EQ(r.support, 3u);
            EXPECT_GT(r.lift, 1.0);  // strongly associated (lift = 2.0)
            found = true;
        }
    EXPECT_TRUE(found);

    // Every reported rule meets the confidence floor.
    for (const auto& r : rules) EXPECT_GE(r.confidence, 0.9);
}

TEST(AssociationRules, MinSupportMonotonicity) {
    // Higher min-support yields a subset of the itemsets found at lower min-support.
    const auto txns = groceries();
    const auto low  = to_map(apriori_frequent_itemsets(txns, 2));
    const auto high = to_map(apriori_frequent_itemsets(txns, 4));
    for (const auto& [items, sup] : high) {
        ASSERT_TRUE(low.count(items)) << "itemset frequent at 4 but missing at 2";
        EXPECT_EQ(low.at(items), sup);
    }
}

TEST(AssociationRules, AprioriLevelsAccountForEveryFrequentSet) {
    const auto txns   = groceries();
    const auto freq   = apriori_frequent_itemsets(txns, 2);
    const auto levels = datamunge::algorithms::apriori_levels(txns, 2);
    ASSERT_FALSE(levels.empty());
    EXPECT_EQ(levels[0].k, 1u);
    EXPECT_EQ(levels[0].joined, 5u);
    EXPECT_EQ(levels[0].frequent, 5u);
    std::size_t total = 0;
    for (std::size_t i = 0; i < levels.size(); ++i) {
        EXPECT_EQ(levels[i].k, i + 1);
        EXPECT_LE(levels[i].candidates, levels[i].joined);
        EXPECT_LE(levels[i].frequent, levels[i].candidates);
        std::size_t of_size = 0;
        for (const auto& f : freq) of_size += f.items.size() == levels[i].k;
        EXPECT_EQ(levels[i].frequent, of_size);
        total += levels[i].frequent;
    }
    EXPECT_EQ(total, freq.size());
}

TEST(AssociationRules, EncodesDataFrameRowsAsItems) {
    datamunge::dstruct::DataFrame df;
    df.add_column("cls", std::vector<double>{1, 3, 1});
    df.add_column("sex", std::vector<std::string>{"f", "m", "m"});
    const auto enc = datamunge::algorithms::encode_transactions(df, {"cls", "sex"});
    ASSERT_EQ(enc.transactions.size(), 3u);
    EXPECT_EQ(enc.labels, (std::vector<std::string>{"cls=1", "sex=f", "cls=3", "sex=m"}));
    EXPECT_EQ(enc.transactions[2], (std::vector<std::size_t>{0, 3}));
    EXPECT_EQ(datamunge::algorithms::itemset_label({0, 3}, enc.labels), "{cls=1, sex=m}");
    EXPECT_THROW(datamunge::algorithms::encode_transactions(df, {"nope"}), std::exception);
}

TEST(AssociationRules, ImprovementComparesWithShorterRules) {
    const auto txns  = groceries();
    const auto freq  = apriori_frequent_itemsets(txns, 2);
    const auto rules = association_rules(freq, txns.size(), 0.0);
    const auto imp   = datamunge::algorithms::rule_improvements(rules, freq, txns.size());
    ASSERT_EQ(imp.size(), rules.size());
    for (std::size_t i = 0; i < rules.size(); ++i)
        if (rules[i].antecedent == std::vector<std::size_t>{4} && rules[i].consequent == std::vector<std::size_t>{3}) {
            // diapers => beer: confidence 1, beer alone in 4 of 8 baskets.
            EXPECT_NEAR(imp[i], 1.0 - 4.0 / 8.0, 1e-12);
        }
}
