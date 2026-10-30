/// @file association_rules.cpp
/// @brief Apriori, Eclat and FP-growth miners, rule generation and DataFrame encoding.

#include <datamunge/algorithms/association_rules.hpp>

#include <algorithm>
#include <format>
#include <iterator>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::algorithms {

/// @brief Shared helpers for the miners.
namespace {

/// Canonicalize transactions: sort each and drop duplicate items.
std::vector<std::vector<std::size_t>> normalize(const std::vector<std::vector<std::size_t>>& txns) {
    std::vector<std::vector<std::size_t>> out;
    out.reserve(txns.size());
    for (auto t : txns) {
        std::sort(t.begin(), t.end());
        t.erase(std::unique(t.begin(), t.end()), t.end());
        out.push_back(std::move(t));
    }
    return out;
}

/// Sort a set of frequent itemsets into a canonical order (by size, then lexicographically).
void canonical_sort(std::vector<FrequentItemset>& sets) {
    for (auto& s : sets) std::sort(s.items.begin(), s.items.end());
    std::sort(sets.begin(), sets.end(), [](const FrequentItemset& a, const FrequentItemset& b) {
        if (a.items.size() != b.items.size()) return a.items.size() < b.items.size();
        return a.items < b.items;
    });
}

/// @brief Apriori proper, also recording the work done at each level.
/// @param transactions Baskets of item ids.
/// @param min_support_count Minimum support count.
/// @param levels Receives one AprioriLevel per itemset size.
/// @return The frequent itemsets in canonical order.
std::vector<FrequentItemset> apriori_core(const std::vector<std::vector<std::size_t>>& transactions,
                                          std::size_t min_support_count, std::vector<AprioriLevel>& levels) {
    const auto                   txns = normalize(transactions);
    std::vector<FrequentItemset> result;

    // Level 1: singleton counts.
    std::map<std::size_t, std::size_t> singles;
    for (const auto& t : txns)
        for (std::size_t it : t) ++singles[it];
    std::vector<std::vector<std::size_t>> level;
    for (const auto& [it, c] : singles)
        if (c >= min_support_count) {
            level.push_back({it});
            result.push_back({{it}, c});
        }
    levels.push_back({1, singles.size(), singles.size(), level.size()});

    // Level k -> k+1.
    while (!level.empty()) {
        std::vector<std::vector<std::size_t>> candidates;
        std::size_t joined = 0;
        for (std::size_t a = 0; a < level.size(); ++a)
            for (std::size_t b = a + 1; b < level.size(); ++b) {
                const auto& x = level[a];
                const auto& y = level[b];
                bool        share_prefix = true;  // first k-1 items equal
                for (std::size_t i = 0; i + 1 < x.size(); ++i)
                    if (x[i] != y[i]) {
                        share_prefix = false;
                        break;
                    }
                if (!share_prefix || x.back() >= y.back()) continue;
                std::vector<std::size_t> cand = x;
                cand.push_back(y.back());
                ++joined;
                // Prune: every k-subset of cand must be frequent (present in `level`).
                bool ok = true;
                for (std::size_t drop = 0; drop < cand.size() && ok; ++drop) {
                    std::vector<std::size_t> sub;
                    for (std::size_t i = 0; i < cand.size(); ++i)
                        if (i != drop) sub.push_back(cand[i]);
                    ok = std::find(level.begin(), level.end(), sub) != level.end();
                }
                if (ok) candidates.push_back(std::move(cand));
            }

        // Count candidate supports in one database scan.
        std::vector<std::size_t> counts(candidates.size(), 0);
        for (const auto& t : txns)
            for (std::size_t c = 0; c < candidates.size(); ++c)
                if (std::includes(t.begin(), t.end(), candidates[c].begin(), candidates[c].end())) ++counts[c];

        level.clear();
        for (std::size_t c = 0; c < candidates.size(); ++c)
            if (counts[c] >= min_support_count) {
                level.push_back(candidates[c]);
                result.push_back({candidates[c], counts[c]});
            }
        if (joined > 0) levels.push_back({levels.back().k + 1, joined, candidates.size(), level.size()});
    }

    canonical_sort(result);
    return result;
}

}  // namespace

std::vector<FrequentItemset> apriori_frequent_itemsets(const std::vector<std::vector<std::size_t>>& transactions,
                                                       std::size_t min_support_count) {
    std::vector<AprioriLevel> levels;
    return apriori_core(transactions, min_support_count, levels);
}

std::vector<AprioriLevel> apriori_levels(const std::vector<std::vector<std::size_t>>& transactions,
                                         std::size_t min_support_count) {
    std::vector<AprioriLevel> levels;
    apriori_core(transactions, min_support_count, levels);
    return levels;
}

/// @brief Helpers for Eclat.
namespace {

/// Recursive Eclat over (item, tidset) extensions sharing a common prefix.
void eclat_recurse(const std::vector<std::size_t>&                                     prefix,
                          const std::vector<std::pair<std::size_t, std::vector<std::size_t>>>& items,
                          std::size_t min_support, std::vector<FrequentItemset>& out) {
    for (std::size_t i = 0; i < items.size(); ++i) {
        std::vector<std::size_t> new_prefix = prefix;
        new_prefix.push_back(items[i].first);
        out.push_back({new_prefix, items[i].second.size()});

        std::vector<std::pair<std::size_t, std::vector<std::size_t>>> ext;
        for (std::size_t j = i + 1; j < items.size(); ++j) {
            std::vector<std::size_t> inter;
            std::set_intersection(items[i].second.begin(), items[i].second.end(), items[j].second.begin(),
                                  items[j].second.end(), std::back_inserter(inter));
            if (inter.size() >= min_support) ext.push_back({items[j].first, std::move(inter)});
        }
        if (!ext.empty()) eclat_recurse(new_prefix, ext, min_support, out);
    }
}

}  // namespace

/// \brief Eclat: depth-first mining over a vertical (tidset) database layout.
///
/// Each item is represented by its *tidset* -- the sorted set of transaction ids
/// that contain it -- so an itemset's support is simply the size of the
/// intersection of its items' tidsets. Eclat descends depth-first, extending a
/// prefix by intersecting tidsets, and never rescans the raw database.
std::vector<FrequentItemset> eclat_frequent_itemsets(
    const std::vector<std::vector<std::size_t>>& transactions, std::size_t min_support_count) {
    const auto txns = normalize(transactions);

    std::map<std::size_t, std::vector<std::size_t>> tidsets;
    for (std::size_t t = 0; t < txns.size(); ++t)
        for (std::size_t it : txns[t]) tidsets[it].push_back(t);

    std::vector<std::pair<std::size_t, std::vector<std::size_t>>> items;
    for (auto& [it, tids] : tidsets)
        if (tids.size() >= min_support_count) items.push_back({it, tids});

    std::vector<FrequentItemset> result;
    eclat_recurse({}, items, min_support_count, result);
    canonical_sort(result);
    return result;
}

/// @brief Helpers for FP-growth.
namespace {

/// A frequent-pattern tree for FP-growth.
struct FPTree {
    /// One tree node: an item on a shared transaction prefix.
    struct Node {
        std::size_t                          item;      ///< the item at this node.
        std::size_t                          count;     ///< weighted transactions through this node.
        int                                  parent;    ///< parent index (-1 for the root).
        std::unordered_map<std::size_t, int> children;  ///< child index by item.
    };
    std::vector<Node>                                   nodes;   ///< nodes[0] is the root.
    std::unordered_map<std::size_t, std::vector<int>>   header;  ///< item -> node indices.

    /// Creates a tree holding only the root.
    FPTree() { nodes.push_back({static_cast<std::size_t>(-1), 0, -1, {}}); }
    /// Whether the tree holds only the root.
    /// @return true when no transaction has been inserted.
    bool empty() const { return nodes.size() == 1; }

    /// Insert an item list (already ordered by the tree's item order) with a weight.
    void insert(const std::vector<std::size_t>& ordered, std::size_t count) {
        int cur = 0;
        for (std::size_t it : ordered) {
            auto found = nodes[cur].children.find(it);
            int  next;
            if (found == nodes[cur].children.end()) {
                next = static_cast<int>(nodes.size());
                nodes.push_back({it, 0, cur, {}});
                nodes[cur].children[it] = next;  // nodes[cur] re-indexed after push_back
                header[it].push_back(next);
            } else {
                next = found->second;
            }
            nodes[next].count += count;
            cur = next;
        }
    }
};

/// Order candidate items by descending (weighted) frequency, breaking ties by id.
std::vector<std::size_t> frequency_order(const std::unordered_map<std::size_t, std::size_t>& freq,
                                                std::size_t                                          min_support) {
    std::vector<std::pair<std::size_t, std::size_t>> v;
    for (const auto& [it, f] : freq)
        if (f >= min_support) v.push_back({it, f});
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    std::vector<std::size_t> order;
    for (const auto& [it, f] : v) order.push_back(it);
    return order;
}

/// Build an FP-tree from weighted transactions, keeping only frequent items.
FPTree build_tree(const std::vector<std::pair<std::vector<std::size_t>, std::size_t>>& weighted,
                         std::size_t min_support) {
    std::unordered_map<std::size_t, std::size_t> freq;
    for (const auto& [items, w] : weighted)
        for (std::size_t it : items) freq[it] += w;
    const std::vector<std::size_t>               order = frequency_order(freq, min_support);
    std::unordered_map<std::size_t, std::size_t> rank;
    for (std::size_t i = 0; i < order.size(); ++i) rank[order[i]] = i;

    FPTree tree;
    for (const auto& [items, w] : weighted) {
        std::vector<std::size_t> kept;
        for (std::size_t it : items)
            if (rank.count(it)) kept.push_back(it);
        std::sort(kept.begin(), kept.end(), [&](std::size_t a, std::size_t b) { return rank[a] < rank[b]; });
        if (!kept.empty()) tree.insert(kept, w);
    }
    return tree;
}

/// Mine an FP-tree recursively via conditional pattern bases.
void fp_mine(const FPTree& tree, std::size_t min_support, const std::vector<std::size_t>& prefix,
                    std::vector<FrequentItemset>& out) {
    // Total count of each item in this tree.
    std::vector<std::pair<std::size_t, std::size_t>> items;
    for (const auto& [it, idxs] : tree.header) {
        std::size_t tot = 0;
        for (int idx : idxs) tot += tree.nodes[idx].count;
        if (tot >= min_support) items.push_back({it, tot});
    }
    // Process least-frequent items first (canonical, deterministic).
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });

    for (const auto& [it, tot] : items) {
        std::vector<std::size_t> new_prefix = prefix;
        new_prefix.push_back(it);
        out.push_back({new_prefix, tot});

        // Conditional pattern base: prefix paths ending at `it`, weighted by node count.
        std::vector<std::pair<std::vector<std::size_t>, std::size_t>> cond;
        for (int idx : tree.header.at(it)) {
            const std::size_t       cnt = tree.nodes[idx].count;
            std::vector<std::size_t> path;
            int                      p = tree.nodes[idx].parent;
            while (p > 0) {  // stop before the root (index 0)
                path.push_back(tree.nodes[p].item);
                p = tree.nodes[p].parent;
            }
            if (!path.empty()) cond.push_back({path, cnt});
        }
        if (cond.empty()) continue;
        FPTree cond_tree = build_tree(cond, min_support);
        if (!cond_tree.empty()) fp_mine(cond_tree, min_support, new_prefix, out);
    }
}

}  // namespace

/// \brief FP-growth: mine frequent itemsets from a compressed frequent-pattern tree.
///
/// Two database scans build the FP-tree: the first counts item frequencies (to keep
/// and order frequent items by descending support), the second inserts each
/// transaction, sharing common prefixes so repeated patterns compress into shared
/// paths. Mining then proceeds recursively without candidate generation: for each
/// item, its *conditional pattern base* (the prefix paths leading to it) forms a
/// smaller conditional FP-tree that is mined for the patterns containing that item.
std::vector<FrequentItemset> fp_growth_frequent_itemsets(
    const std::vector<std::vector<std::size_t>>& transactions, std::size_t min_support_count) {
    const auto txns = normalize(transactions);
    std::vector<std::pair<std::vector<std::size_t>, std::size_t>> weighted;
    weighted.reserve(txns.size());
    for (const auto& t : txns) weighted.push_back({t, 1});

    FPTree              tree = build_tree(weighted, min_support_count);
    std::vector<FrequentItemset> result;
    fp_mine(tree, min_support_count, {}, result);
    canonical_sort(result);
    return result;
}

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
std::vector<AssociationRule> association_rules(const std::vector<FrequentItemset>& frequent,
                                                      std::size_t                         num_transactions,
                                                      double                              min_confidence) {
    std::map<std::vector<std::size_t>, std::size_t> support;
    for (const auto& fs : frequent) support[fs.items] = fs.support;

    std::vector<AssociationRule> rules;
    for (const auto& fs : frequent) {
        const std::size_t n = fs.items.size();
        if (n < 2) continue;
        const std::size_t full_support = fs.support;
        // Enumerate non-empty proper subsets as antecedents via bitmask.
        for (std::size_t mask = 1; mask + 1 < (std::size_t{1} << n); ++mask) {
            std::vector<std::size_t> a, c;
            for (std::size_t i = 0; i < n; ++i)
                ((mask >> i) & 1u ? a : c).push_back(fs.items[i]);
            const auto ita = support.find(a);
            if (ita == support.end() || ita->second == 0) continue;
            const double confidence = static_cast<double>(full_support) / static_cast<double>(ita->second);
            if (confidence < min_confidence) continue;
            double     lift  = 0.0;
            const auto itc   = support.find(c);
            if (itc != support.end() && num_transactions > 0)
                lift = confidence / (static_cast<double>(itc->second) / static_cast<double>(num_transactions));
            rules.push_back({a, c, full_support, confidence, lift});
        }
    }
    std::sort(rules.begin(), rules.end(), [](const AssociationRule& x, const AssociationRule& y) {
        if (x.confidence != y.confidence) return x.confidence > y.confidence;
        if (x.antecedent != y.antecedent) return x.antecedent < y.antecedent;
        return x.consequent < y.consequent;
    });
    return rules;
}

std::vector<double> rule_improvements(const std::vector<AssociationRule>& rules,
                                      const std::vector<FrequentItemset>& frequent, std::size_t num_transactions) {
    std::map<std::vector<std::size_t>, std::size_t> support;
    for (const auto& fs : frequent) support[fs.items] = fs.support;
    auto supp = [&](const std::vector<std::size_t>& items) {
        if (items.empty()) return num_transactions;
        const auto it = support.find(items);
        return it == support.end() ? std::size_t{0} : it->second;
    };
    std::vector<double> out;
    out.reserve(rules.size());
    for (const auto& r : rules) {
        const std::size_t n    = r.antecedent.size();
        double            best = -1.0;
        for (std::size_t mask = 0; mask + 1 < (std::size_t{1} << n); ++mask) {  // proper subsets A'
            std::vector<std::size_t> a;
            for (std::size_t i = 0; i < n; ++i)
                if ((mask >> i) & 1u) a.push_back(r.antecedent[i]);
            std::vector<std::size_t> ac = a;
            ac.insert(ac.end(), r.consequent.begin(), r.consequent.end());
            std::sort(ac.begin(), ac.end());
            const std::size_t sa = supp(a);
            if (sa > 0) best = std::max(best, static_cast<double>(supp(ac)) / static_cast<double>(sa));
        }
        out.push_back(r.confidence - best);
    }
    return out;
}

TransactionData encode_transactions(const dstruct::DataFrame& data, const std::vector<std::string>& columns) {
    using ColumnType = dstruct::DataFrame::ColumnType;
    std::vector<ColumnType> types;
    for (const auto& c : columns) types.push_back(data.column_type(c));
    TransactionData                              out;
    std::unordered_map<std::string, std::size_t> id;
    out.transactions.resize(data.nrows());
    for (std::size_t r = 0; r < data.nrows(); ++r)
        for (std::size_t j = 0; j < columns.size(); ++j) {
            if (data.is_null(columns[j], r)) continue;
            const std::string value = types[j] == ColumnType::String ? data.string_at(columns[j], r)
                                                                     : std::format("{}", data.double_at(columns[j], r));
            const std::string label = columns[j] + "=" + value;
            auto [it, inserted] = id.try_emplace(label, out.labels.size());
            if (inserted) out.labels.push_back(label);
            out.transactions[r].push_back(it->second);
        }
    for (auto& t : out.transactions) std::sort(t.begin(), t.end());
    return out;
}

std::string itemset_label(const std::vector<std::size_t>& items, const std::vector<std::string>& labels) {
    std::string s = "{";
    for (std::size_t i = 0; i < items.size(); ++i) s += (i ? ", " : "") + labels.at(items[i]);
    return s + "}";
}

}  // namespace datamunge::algorithms
