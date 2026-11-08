/// @file association_rules.cpp
/// @brief Apriori and Eclat mining, association rules and DataFrame encoding.

#include <datamunge/algorithms/eclat.hpp>

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

/// @brief Checked counters and explicit conditional-database frames for Eclat.
namespace {
/// @brief Add two logical counts without wrapping.
/// @param a Existing count.
/// @param b Additional count.
/// @return Their representable sum.
/// @throws std::length_error On size_t overflow.
std::size_t eclat_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max()-a) throw std::length_error("Eclat count overflow");
    return a+b;
}
/// @brief Reserve retained history entries before copying the associated records.
/// @param trace Owned capture in progress, null for work-only mining.
/// @param entries Additional logical entries.
/// @param limit Optional quota; zero disables only the quota.
/// @throws std::length_error On overflow or quota exhaustion.
void eclat_capture(EclatTrace* trace, std::size_t entries, std::size_t limit) {
    if (!trace) return;
    const auto total=eclat_add(trace->captured_entries,entries);
    if (limit && total>limit) throw std::length_error("Eclat capture limit exceeded");
    trace->captured_entries=total;
}
/// @brief An owned conditional suffix tidset, independent of reallocating history.
struct EclatCandidate {
    std::size_t item; ///< Last item of the conditional set.
    std::vector<std::size_t> tids; ///< Owned conditional tids, ascending and unique.
    std::size_t state; ///< State id when capturing, otherwise zero.
};
/// @brief One explicit DFS call, preserving the original extension schedule.
struct EclatFrame {
    std::vector<std::size_t> prefix; ///< Common item prefix for every candidate.
    std::vector<EclatCandidate> candidates; ///< Frequent extensions in ascending suffix order.
    std::size_t next=0; ///< Next extension to visit and emit.
};
/// @brief Store a root or actual intersection before its working buffer is moved.
/// @param trace Optional owned trace.
/// @param items Full sorted itemset.
/// @param tids Sorted intersection result or singleton tids.
/// @param left Earlier left operand or root sentinel.
/// @param right Earlier right operand or root sentinel.
/// @param steps Paired-head merge iterations.
/// @param frequent Support meets threshold.
/// @param limit Optional capture quota.
/// @return New state index, or zero without capture.
/// @throws std::length_error On capture quota/arithmetic/container limits.
std::size_t eclat_state(EclatTrace* trace, const std::vector<std::size_t>& items,
                        const std::vector<std::size_t>& tids, std::size_t left,
                        std::size_t right, std::size_t steps, bool frequent, std::size_t limit) {
    if (!trace) return 0;
    eclat_capture(trace,eclat_add(1,eclat_add(items.size(),tids.size())),limit);
    const auto id=trace->states.size();
    trace->states.push_back({{items,tids.size()},tids,left,right,steps,frequent});
    return id;
}
/// @brief Common iterative kernel, retaining no history when trace is null.
/// @param transactions Raw unchanged baskets.
/// @param threshold Absolute support threshold, including zero.
/// @param trace Optional capture already holding its validated raw base/input.
/// @param limit Optional retained-history quota.
/// @return Canonical sets and checked work counters.
/// @throws std::length_error On arithmetic, quota or container limits.
EclatResult eclat_core(const std::vector<std::vector<std::size_t>>& transactions,
                       std::size_t threshold, EclatTrace* trace, std::size_t limit) {
    EclatResult out;
    out.work.transactions=transactions.size();
    for (const auto& row:transactions) out.work.raw_items=eclat_add(out.work.raw_items,row.size());
    auto roots=vertical_tidsets(transactions);
    if (trace) trace->singleton_count=roots.size();
    std::vector<EclatCandidate> candidates;
    const auto none=std::numeric_limits<std::size_t>::max();
    for (auto& root:roots) {
        out.work.incidences=eclat_add(out.work.incidences,root.tids.size());
        const bool keep=root.tids.size()>=threshold;
        const auto state=eclat_state(trace,{root.item},root.tids,none,none,0,keep,limit);
        if (keep) {
            out.work.frequent_singletons=eclat_add(out.work.frequent_singletons,1);
            candidates.push_back({root.item,std::move(root.tids),state});
        }
    }
    std::vector<EclatFrame> frames;
    if (!candidates.empty()) frames.push_back({{},std::move(candidates),0});
    out.work.peak_frames=frames.size();
    while (!frames.empty()) {
        auto& frame=frames.back();
        if (frame.next==frame.candidates.size()) {frames.pop_back();continue;}
        const auto i=frame.next++;
        const auto& left=frame.candidates[i];
        auto prefix=frame.prefix;prefix.push_back(left.item);
        eclat_capture(trace,1,limit);
        if (trace) trace->emitted.push_back(left.state);
        out.work.emitted_itemsets=eclat_add(out.work.emitted_itemsets,1);
        out.itemsets.push_back({prefix,left.tids.size()});
        std::vector<EclatCandidate> children;
        for (std::size_t j=i+1;j<frame.candidates.size();++j) {
            const auto& right=frame.candidates[j];
            out.work.intersections=eclat_add(out.work.intersections,1);
            std::vector<std::size_t> tids;
            std::size_t a=0,b=0,steps=0;
            while (a<left.tids.size() && b<right.tids.size()) {
                steps=eclat_add(steps,1);
                if (left.tids[a]<right.tids[b]) ++a;
                else if (right.tids[b]<left.tids[a]) ++b;
                else {tids.push_back(left.tids[a]);++a;++b;}
            }
            out.work.intersection_steps=eclat_add(out.work.intersection_steps,steps);
            out.work.matched_tids=eclat_add(out.work.matched_tids,tids.size());
            auto items=prefix;items.push_back(right.item);
            const bool keep=tids.size()>=threshold;
            const auto state=eclat_state(trace,items,tids,left.state,right.state,steps,keep,limit);
            if (keep) children.push_back({right.item,std::move(tids),state});
        }
        // All sibling intersections precede descent, exactly as in the original recursion.
        // No reference to frame/left survives the potentially reallocating frame push.
        if (!children.empty()) {
            frames.push_back({std::move(prefix),std::move(children),0});
            out.work.peak_frames=std::max(out.work.peak_frames,frames.size());
        }
    }
    canonical_sort(out.itemsets);
    return out;
}
} // namespace

std::vector<ItemTidset> vertical_tidsets(const std::vector<std::vector<std::size_t>>& transactions) {
    const auto normalized=normalize(transactions);
    std::map<std::size_t,std::vector<std::size_t>> index;
    for (std::size_t t=0;t<normalized.size();++t)
        for (auto item:normalized[t]) index[item].push_back(t);
    std::vector<ItemTidset> out;out.reserve(index.size());
    for (auto& [item,tids]:index) out.push_back({item,std::move(tids)});
    return out;
}
std::size_t eclat_trace_base_entries(const std::vector<std::vector<std::size_t>>& transactions) {
    auto entries=transactions.size();
    for (const auto& row:transactions) entries=eclat_add(entries,row.size());
    return entries;
}
EclatResult eclat_frequent_itemsets_work(const std::vector<std::vector<std::size_t>>& transactions,
                                        std::size_t min_support_count) {
    return eclat_core(transactions,min_support_count,nullptr,0);
}
std::vector<FrequentItemset> eclat_frequent_itemsets(
    const std::vector<std::vector<std::size_t>>& transactions,std::size_t min_support_count) {
    return eclat_frequent_itemsets_work(transactions,min_support_count).itemsets;
}
EclatTrace eclat_trace(const std::vector<std::vector<std::size_t>>& transactions,
                       std::size_t min_support_count,std::size_t max_entries) {
    EclatTrace trace;
    eclat_capture(&trace,eclat_trace_base_entries(transactions),max_entries);
    trace.transactions=transactions;trace.min_support_count=min_support_count;
    auto result=eclat_core(transactions,min_support_count,&trace,max_entries);
    trace.itemsets=std::move(result.itemsets);trace.work=result.work;
    return trace;
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
