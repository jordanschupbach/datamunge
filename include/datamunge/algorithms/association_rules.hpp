#pragma once

/// \file association_rules.hpp
/// \brief Frequent-itemset mining and association-rule generation: the Apriori,
///        Eclat, and FP-growth algorithms, plus rule extraction with
///        support/confidence/lift.
///
/// Given a set of *transactions* (each a set of items -- a shopping basket, a set
/// of symptoms, a document's words), *frequent-itemset mining* finds every set of
/// items that co-occurs in at least a *minimum support* count of transactions, and
/// *association-rule mining* turns those itemsets into implications
/// \f$A \Rightarrow C\f$ ("customers who buy \f$A\f$ tend to buy \f$C\f$") ranked by
/// confidence and lift.
///
/// For positive minimum support, the three mining algorithms compute the *same* frequent-itemset collection by
/// very different strategies:
///   - *Apriori* (Agrawal & Srikant 1994): breadth-first, level by level, using the
///     downward-closure principle -- every subset of a frequent set is frequent --
///     to prune candidates before an expensive support-counting scan.
///   - *Eclat* (Zaki 2000): depth-first over a *vertical* layout, representing each
///     itemset by its transaction-id set (tidset) and computing supports by tidset
///     *intersection*, so no repeated database scans are needed.
///   - *FP-growth* (Han, Pei & Yin 2000): compresses the database into a
///     *frequent-pattern tree* and mines it recursively via conditional pattern
///     bases, avoiding explicit candidate generation entirely.
///
/// Items are represented as `std::size_t` identifiers and transactions as sets of
/// them; all three functions return frequent itemsets (with support counts) in a
/// canonical sorted form, which makes them directly cross-checkable.

#include <datamunge/dstruct/dataframe.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// A set of items found frequent, with the number of transactions containing it.
struct FrequentItemset {
    std::vector<std::size_t> items;        ///< The itemset, sorted ascending.
    std::size_t              support = 0;  ///< Transactions containing all of \c items.
};

/// An association rule \f$A \Rightarrow C\f$ with its quality metrics.
struct AssociationRule {
    std::vector<std::size_t> antecedent;   ///< Left-hand side \f$A\f$ (sorted).
    std::vector<std::size_t> consequent;   ///< Right-hand side \f$C\f$ (sorted).
    std::size_t              support   = 0;///< Support count of \f$A\cup C\f$.
    double                   confidence = 0.0;  ///< \f$\mathrm{supp}(A\cup C)/\mathrm{supp}(A)\f$.
    double                   lift       = 0.0;  ///< \f$\mathrm{conf}/(\mathrm{supp}(C)/N)\f$.
};

/// Work done by Apriori at one itemset size \f$k\f$.
struct AprioriLevel {
    std::size_t k          = 0;  ///< Itemset size.
    std::size_t joined     = 0;  ///< Candidates produced by the prefix join (all singletons at \f$k=1\f$).
    std::size_t candidates = 0;  ///< Candidates left after downward-closure pruning, i.e. counted.
    std::size_t frequent   = 0;  ///< Candidates meeting the minimum support.
};

/// Transactions built from table rows, with a label for every item id.
struct TransactionData {
    std::vector<std::vector<std::size_t>> transactions;  ///< One item-id set per row.
    std::vector<std::string>              labels;        ///< `labels[id]` is the item's `column=value` text.
};

/// \brief Apriori: level-wise frequent-itemset mining with downward-closure pruning.
///
/// Starting from frequent singletons, candidate \f$(k{+}1)\f$-itemsets are generated
/// by joining pairs of frequent \f$k\f$-itemsets that share their first \f$k-1\f$
/// items, then pruned by discarding any candidate that has an infrequent
/// \f$k\f$-subset (downward closure). Surviving candidates are counted in one scan of
/// the database, and the process repeats until no candidate is frequent.
///
/// \param transactions      Baskets of item ids.
/// \param min_support_count Minimum number of transactions an itemset must appear in.
/// \return Every frequent itemset, by size and then lexicographically.
std::vector<FrequentItemset> apriori_frequent_itemsets(const std::vector<std::vector<std::size_t>>& transactions,
                                                       std::size_t min_support_count);

/// \brief The per-level work of apriori_frequent_itemsets(): joined, counted and frequent candidates.
/// \param transactions      Baskets of item ids.
/// \param min_support_count Minimum support count.
/// \return One entry per itemset size that had candidates to count.
std::vector<AprioriLevel> apriori_levels(const std::vector<std::vector<std::size_t>>& transactions,
                                         std::size_t min_support_count);

/// @brief Eclat: canonical frequent sets from conditional sorted tidset intersections.
///
/// Within-basket duplicates count once; empty/repeated baskets retain their row
/// identities. Opaque size_t ids include SIZE_MAX. Empty input/universe yields no
/// sets. Threshold zero enumerates all nonempty observed-universe subsets, even
/// with zero support (FP-growth currently omits those zero-support combinations).
/// Explicit DFS frames avoid call-stack recursion. Results own their buffers;
/// input is unchanged and independent calls share no mutable state. There is no
/// optional capture quota. Output may contain 2^U-1 sets for U observed items.
/// @param transactions Raw baskets of item ids.
/// @param min_support_count Absolute minimum row support; above N gives no sets.
/// @return Frequent sets ordered by size, then lexicographically by item id.
/// @throws std::length_error On shape, counter or container overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto f = dm::algorithms::eclat_frequent_itemsets({{0,1},{0}},1);
/// std::print("{} {}\n", f.size(), f.back().support); // 3 1
/// @endcode
/// @par References
/// - M. J. Zaki (2000), Scalable Algorithms for Association Mining, IEEE TKDE 12(3), 372–390. doi:10.1109/69.846291. Uses plain sorted tidsets, no diffsets.
/// @see apriori_frequent_itemsets(), eclat_frequent_itemsets_work(), eclat_trace()
std::vector<FrequentItemset> eclat_frequent_itemsets(const std::vector<std::vector<std::size_t>>& transactions,
                                                     std::size_t min_support_count);

/// @brief Mine canonical positive-support patterns from shared FP-tree prefixes.
/// Baskets are sets: sort and deduplicate each without changing the caller.
/// Empty/repeated rows retain their identities; IDs are opaque size_t including
/// SIZE_MAX. No empty itemset is returned. Positive thresholds return all sets
/// whose row count meets the threshold; above N returns none. Threshold0 retains
/// the historical positive-support-only FP behavior, unlike Apriori/Eclat0.
/// Build order is support descending then ID ascending; mining order is support
/// ascending then ID ascending. Explicit DFS preserves the original emit/base/
/// child-build/descent schedule without recursive call-stack depth. No single-path
/// shortcut, Apriori candidate join or extra database scan per candidate is used.
/// Output can have exponentially many sets and item IDs. Sorting, hash operations,
/// conditional rebuilding and output copying remain real work; no universal
/// compression/runtime guarantee. Independent calls share no mutable state.
/// Counts/capacities are checked; errors return no partial result.
///
/// Ordinary calls keep no history and impose no optional output/capture cap.
/// Working memory includes normalized/weighted input, active ancestor trees and
/// conditional buffers, in addition to all returned itemsets. Tree construction
/// sorts item frequencies and each retained row; it is not generally O(N*width).
/// @param transactions Raw item baskets, never mutated.
/// @param min_support_count Absolute row count threshold;0 retains positive-support sets.
/// @return Owned itemsets ordered by cardinality then lexicographic item IDs.
/// @throws std::length_error Count or container capacity overflows.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto f=dm::algorithms::fp_growth_frequent_itemsets({{0,1},{0},{1}},2);
/// std::cout << f.size(); // 2
/// @endcode
/// @par References
/// - J. Han, J. Pei and Y. Yin (2000), Mining Frequent Patterns without Candidate Generation, SIGMOD2000,1–12, doi:10.1145/342009.335372; https://sigmodrecord.org/?download_id=6623&smd_process_download=1; direct conditional-tree variant without single-path shortcut.
/// @see fp_growth_frequent_itemsets_work(), fp_growth_trace()
std::vector<FrequentItemset> fp_growth_frequent_itemsets(const std::vector<std::vector<std::size_t>>& transactions,
                                                         std::size_t min_support_count);

/// \brief Generate association rules from a set of frequent itemsets.
///
/// For every frequent itemset \f$I\f$ of size \f$\ge 2\f$ and every non-empty proper
/// subset \f$A\subset I\f$, the rule \f$A \Rightarrow I\setminus A\f$ is emitted if its
/// *confidence* \f$\mathrm{supp}(I)/\mathrm{supp}(A)\f$ meets \p min_confidence. The
/// *lift* \f$\mathrm{conf}/(\mathrm{supp}(C)/N)\f$ is also reported: lift \f$>1\f$ means
/// \f$A\f$ and \f$C\f$ co-occur more than chance would predict.
///
/// \param frequent          Frequent itemsets with support counts (from any miner above).
/// \param num_transactions  Total transaction count \f$N\f$ (for lift).
/// \param min_confidence    Minimum confidence to keep a rule.
/// \return Rules sorted by descending confidence.
std::vector<AssociationRule> association_rules(const std::vector<FrequentItemset>& frequent,
                                               std::size_t num_transactions, double min_confidence);

/// \brief Improvement of each rule over its simpler versions (Bayardo, Agrawal & Gunopulos 1999).
///
/// For \f$A\Rightarrow C\f$ it is
/// \f$\mathrm{conf}(A\Rightarrow C)-\max_{A'\subsetneq A}\mathrm{conf}(A'\Rightarrow C)\f$, where the
/// empty antecedent has confidence \f$\mathrm{supp}(C)/N\f$. A rule whose improvement is near zero
/// or negative says nothing that a shorter rule does not already say.
/// \param rules             Rules from association_rules().
/// \param frequent          The frequent itemsets the rules came from (every needed subset is in it,
///                          by downward closure).
/// \param num_transactions  Total transaction count \f$N\f$.
/// \return One improvement per rule, in the order of \p rules.
std::vector<double> rule_improvements(const std::vector<AssociationRule>& rules,
                                      const std::vector<FrequentItemset>& frequent, std::size_t num_transactions);

/// \brief One transaction per row of \p data, one item per non-null cell of \p columns.
///
/// Each distinct `column=value` pair becomes an item. Numbers are written with
/// `std::format("{}")`, so 3.0 becomes `3`. Ids follow first appearance, scanning rows
/// in order and columns in the order given.
/// \param data    The table.
/// \param columns Columns to turn into items.
/// \return The transactions and the label of every item id.
/// \throws std::out_of_range if a column does not exist.
TransactionData encode_transactions(const dstruct::DataFrame& data, const std::vector<std::string>& columns);

/// \brief Text for an itemset, such as `{Sex=female, Pclass=1}`.
/// \param items  Item ids.
/// \param labels Labels indexed by item id, e.g. TransactionData::labels.
/// \return The labels joined by commas inside braces.
std::string itemset_label(const std::vector<std::size_t>& items, const std::vector<std::string>& labels);

}  // namespace datamunge::algorithms
