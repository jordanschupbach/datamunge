/// @file aho_corasick.cpp
/// @brief The Aho-Corasick automaton: trie, breadth-first failure and dictionary links, scans.

#include <datamunge/algorithms/aho_corasick.hpp>

#include <algorithm>
#include <queue>

namespace datamunge::algorithms {

AhoCorasick::AhoCorasick(std::vector<std::string> patterns) : patterns_(std::move(patterns)), nodes_(1) {
    // The trie (goto function). Node 0 is the root.
    for (std::size_t pi = 0; pi < patterns_.size(); ++pi) {
        const std::string& p = patterns_[pi];
        if (p.empty()) continue;  // empty patterns match nowhere
        std::size_t cur = 0;
        for (const char ch : p) {
            const auto it = nodes_[cur].next.find(ch);
            if (it != nodes_[cur].next.end()) {
                cur = it->second;
                continue;
            }
            const std::size_t created = nodes_.size();
            Node node;
            node.depth = nodes_[cur].depth + 1;
            node.parent = cur;
            node.label = ch;
            nodes_.push_back(std::move(node));
            nodes_[cur].next[ch] = created;
            cur = created;
        }
        nodes_[cur].output.push_back(pi);
    }
    // Failure and dictionary links in breadth-first order: a state's links depend only on
    // shallower states. For an edge u -ch-> v, fail[v] is the ch-child of the deepest state on
    // u's failure chain that has one (or the root).
    std::queue<std::size_t> bfs;
    bfs.push(0);
    while (!bfs.empty()) {
        const std::size_t u = bfs.front();
        bfs.pop();
        bfs_order_.push_back(u);
        for (const auto& [ch, v] : nodes_[u].next) {
            std::size_t f = nodes_[u].fail;
            while (f != 0 && !nodes_[f].next.contains(ch)) f = nodes_[f].fail;
            std::size_t target = 0;
            if (const auto it = nodes_[f].next.find(ch); it != nodes_[f].next.end() && it->second != v) target = it->second;
            nodes_[v].fail = target;
            nodes_[v].dict = nodes_[target].output.empty() ? nodes_[target].dict : target;
            bfs.push(v);
        }
    }
}

std::size_t AhoCorasick::step(std::size_t state, char ch, AhoCorasickScanStats* stats) const {
    while (state != 0 && !nodes_[state].next.contains(ch)) {
        state = nodes_[state].fail;
        if (stats) ++stats->failure_steps;
    }
    if (const auto it = nodes_[state].next.find(ch); it != nodes_[state].next.end()) {
        if (stats) ++stats->goto_steps;
        return it->second;
    }
    return 0;
}

std::vector<AhoCorasickMatch> AhoCorasick::search(std::string_view text) const {
    std::vector<AhoCorasickMatch> matches;
    std::size_t state = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        state = step(state, text[i], nullptr);
        for (std::size_t u = state; u != 0; u = nodes_[u].dict)
            for (const std::size_t pi : nodes_[u].output) matches.push_back({i + 1 - patterns_[pi].size(), pi});
    }
    return matches;
}

std::vector<std::size_t> AhoCorasick::count(std::string_view text) const {
    std::vector<std::size_t> counts(patterns_.size(), 0);
    std::size_t state = 0;
    for (const char ch : text) {
        state = step(state, ch, nullptr);
        for (std::size_t u = state; u != 0; u = nodes_[u].dict)
            for (const std::size_t pi : nodes_[u].output) ++counts[pi];
    }
    return counts;
}

std::map<std::size_t, std::size_t> AhoCorasick::longest_at(std::string_view text) const {
    std::map<std::size_t, std::size_t> best;
    for (const auto& m : search(text)) {
        const auto [it, inserted] = best.emplace(m.position, m.pattern_index);
        if (!inserted && patterns_[m.pattern_index].size() > patterns_[it->second].size()) it->second = m.pattern_index;
    }
    return best;
}

AhoCorasickScanStats AhoCorasick::scan_stats(std::string_view text) const {
    AhoCorasickScanStats stats;
    std::size_t state = 0;
    for (const char ch : text) {
        ++stats.characters;
        state = step(state, ch, &stats);
        for (std::size_t u = state; u != 0; u = nodes_[u].dict) stats.matches += nodes_[u].output.size();
    }
    return stats;
}

std::string AhoCorasick::prefix(std::size_t state) const {
    std::string out;
    for (; state != 0; state = nodes_[state].parent) out += nodes_[state].label;
    std::ranges::reverse(out);
    return out;
}

dstruct::DataFrame AhoCorasick::automaton() const {
    std::vector<double> state, depth, fail, dict;
    std::vector<std::string> prefixes, fail_prefixes, outputs, edges;
    for (const std::size_t s : bfs_order_) {
        const Node& node = nodes_[s];
        state.push_back(static_cast<double>(s));
        prefixes.push_back(prefix(s));
        depth.push_back(static_cast<double>(node.depth));
        fail.push_back(static_cast<double>(node.fail));
        fail_prefixes.push_back(prefix(node.fail));
        dict.push_back(static_cast<double>(node.dict));
        std::string out, edge;
        for (const std::size_t pi : node.output) out += (out.empty() ? "" : ",") + std::to_string(pi);
        for (const auto& [ch, v] : node.next) edge += ch;
        outputs.push_back(out);
        edges.push_back(edge);
    }
    dstruct::DataFrame frame;
    frame.add_column("state", state);
    frame.add_column("prefix", prefixes);
    frame.add_column("depth", depth);
    frame.add_column("fail", fail);
    frame.add_column("fail_prefix", fail_prefixes);
    frame.add_column("dict", dict);
    frame.add_column("outputs", outputs);
    frame.add_column("edges", edges);
    return frame;
}

std::vector<AhoCorasickMatch> aho_corasick_search(const std::string& text, const std::vector<std::string>& patterns) {
    return AhoCorasick(patterns).search(text);
}

} // namespace datamunge::algorithms
