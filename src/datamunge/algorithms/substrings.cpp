#include <datamunge/algorithms/substrings.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

MaxSubarray kadane(const std::vector<int>& a) {
    if (a.empty()) return MaxSubarray{0.0, 0, 0};
    double      best = a[0];
    std::size_t bb = 0, be = 1;
    double      cur = a[0];
    std::size_t cb = 0;
    for (std::size_t i = 1; i < a.size(); ++i) {
        if (cur < 0) {          // the running prefix hurts: start a fresh subarray at i
            cur = a[i];
            cb  = i;
        } else {
            cur += a[i];        // extend the current subarray
        }
        if (cur > best) {
            best = cur;
            bb   = cb;
            be   = i + 1;
        }
    }
    return MaxSubarray{best, bb, be};
}

std::vector<int> longest_common_substring(const std::vector<int>& a, const std::vector<int>& b) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<int>  prev(m + 1, 0);
    std::vector<int>  cur(m + 1, 0);
    std::size_t       best_len = 0;
    std::size_t       best_end = 0; // one past the last matched index in a
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) {
                cur[j] = prev[j - 1] + 1;
                if (static_cast<std::size_t>(cur[j]) > best_len) {
                    best_len = static_cast<std::size_t>(cur[j]);
                    best_end = i;
                }
            } else {
                cur[j] = 0;
            }
        }
        std::swap(prev, cur);
    }
    return std::vector<int>(a.begin() + static_cast<std::ptrdiff_t>(best_end - best_len),
                            a.begin() + static_cast<std::ptrdiff_t>(best_end));
}

bool wildcard_match(const std::string& text, const std::string& pattern) {
    const std::size_t n = text.size();
    const std::size_t m = pattern.size();
    std::size_t       i = 0;                          // index into text
    std::size_t       j = 0;                          // index into pattern
    std::size_t       star = std::string::npos;       // last '*' position in pattern, if any
    std::size_t       mark = 0;                        // text index matched when that '*' was seen
    while (i < n) {
        if (j < m && (pattern[j] == '?' || pattern[j] == text[i])) {
            ++i;
            ++j;
        } else if (j < m && pattern[j] == '*') {
            star = j;                                  // remember the star and where we are in text
            mark = i;
            ++j;                                       // first try matching '*' with the empty string
        } else if (star != std::string::npos) {
            j = star + 1;                              // backtrack: let the star swallow one more char
            ++mark;
            i = mark;
        } else {
            return false;                              // literal mismatch with no star to fall back on
        }
    }
    while (j < m && pattern[j] == '*') ++j;            // trailing stars may match the empty string
    return j == m;
}

SuffixTree build_suffix_tree(const std::string& text) {
    SuffixTree tree;
    tree.text     = text;
    const int n   = static_cast<int>(text.size());
    auto&     nds = tree.nodes;

    // The character at position i; position n is a unique virtual terminal (code 256).
    auto char_at = [&](int i) -> int { return i < n ? static_cast<int>(static_cast<unsigned char>(text[i])) : 256; };
    auto new_node = [&](int start, int end) -> int {
        nds.push_back(SuffixTreeNode{start, end, -1, {}});
        return static_cast<int>(nds.size()) - 1;
    };

    const int root         = new_node(-1, -1);
    int       active_node  = root;
    int       active_edge  = 0;
    int       active_len   = 0;
    int       remaining    = 0;
    int       leaf_end     = -1;
    int       need_link    = -1;
    auto      add_link     = [&](int node) {
        if (need_link != -1) nds[need_link].suffix_link = node;
        need_link = node;
    };
    auto edge_len = [&](int node) -> int {
        const int e = nds[node].end == -1 ? leaf_end : nds[node].end;
        return e - nds[node].start + 1;
    };

    for (int pos = 0; pos <= n; ++pos) { // position n is the virtual terminal
        leaf_end  = pos;
        remaining += 1;
        need_link = -1;
        while (remaining > 0) {
            if (active_len == 0) active_edge = pos;
            const int c  = char_at(active_edge);
            auto      it = nds[active_node].children.find(c);
            if (it == nds[active_node].children.end()) {
                const int leaf = new_node(pos, -1);          // rule 2: new leaf edge
                nds[active_node].children[c] = leaf;
                add_link(active_node);
            } else {
                const int nxt = it->second;
                if (active_len >= edge_len(nxt)) {           // walk down: hop to the next node
                    active_edge += edge_len(nxt);
                    active_len -= edge_len(nxt);
                    active_node = nxt;
                    continue;
                }
                if (char_at(nds[nxt].start + active_len) == char_at(pos)) { // rule 3: already present
                    active_len += 1;
                    add_link(active_node);
                    break;
                }
                // Split the edge at active_len and hang a new leaf.
                const int split = new_node(nds[nxt].start, nds[nxt].start + active_len - 1);
                nds[active_node].children[c] = split;
                const int leaf = new_node(pos, -1);
                nds[split].children[char_at(pos)] = leaf;
                nds[nxt].start += active_len;
                nds[split].children[char_at(nds[nxt].start)] = nxt;
                add_link(split);
            }
            remaining -= 1;
            if (active_node == root && active_len > 0) {
                active_len -= 1;
                active_edge = pos - remaining + 1;
            } else if (active_node != root) {
                active_node = nds[active_node].suffix_link != -1 ? nds[active_node].suffix_link : root;
            }
        }
    }
    return tree;
}

bool suffix_tree_contains(const SuffixTree& tree, const std::string& pattern) {
    if (pattern.empty()) return true;
    const int n       = static_cast<int>(tree.text.size());
    auto      char_at = [&](int i) -> int { return i < n ? static_cast<int>(static_cast<unsigned char>(tree.text[i])) : 256; };
    int         node = 0;
    std::size_t ip   = 0;
    while (ip < pattern.size()) {
        const int c  = static_cast<int>(static_cast<unsigned char>(pattern[ip]));
        auto      it = tree.nodes[static_cast<std::size_t>(node)].children.find(c);
        if (it == tree.nodes[static_cast<std::size_t>(node)].children.end()) return false;
        const int e    = it->second;
        int       k    = tree.nodes[static_cast<std::size_t>(e)].start;
        const int eend = tree.nodes[static_cast<std::size_t>(e)].end == -1 ? n : tree.nodes[static_cast<std::size_t>(e)].end;
        while (k <= eend && ip < pattern.size()) {
            if (char_at(k) != static_cast<int>(static_cast<unsigned char>(pattern[ip]))) return false;
            ++k;
            ++ip;
        }
        node = e;
    }
    return true;
}

std::size_t distinct_substring_count(const SuffixTree& tree) {
    const int     n      = static_cast<int>(tree.text.size());
    long long     sum    = 0;
    long long     leaves = 0;
    for (std::size_t idx = 1; idx < tree.nodes.size(); ++idx) { // skip the root
        const SuffixTreeNode& nd = tree.nodes[idx];
        const int             e  = nd.end == -1 ? n : nd.end;
        sum += (e - nd.start + 1);
        if (nd.children.empty()) ++leaves;
    }
    return static_cast<std::size_t>(sum - leaves);
}

} // namespace datamunge::algorithms
