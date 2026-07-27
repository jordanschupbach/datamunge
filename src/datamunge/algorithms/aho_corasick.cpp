#include <datamunge/algorithms/aho_corasick.hpp>

#include <queue>
#include <unordered_map>

namespace datamunge::algorithms {

namespace {

// A single automaton state. The goto function is `next` (the trie edges); `fail` is the failure
// link; `dict` is the dictionary-suffix (output) link -- the nearest state reachable by failure
// links that completes a pattern, or 0 (the root) when there is none. `output` holds the indices of
// every pattern that ends *exactly* at this state (a vector, so duplicate patterns coexist).
struct Node {
    std::unordered_map<char, std::size_t> next;
    std::size_t                           fail{0};
    std::size_t                           dict{0};
    std::vector<std::size_t>              output;
};

} // namespace

std::vector<AhoCorasickMatch> aho_corasick_search(const std::string&              text,
                                                  const std::vector<std::string>& patterns) {
    // ---- Build the trie of patterns (the goto function). Node 0 is the root. --------------------
    std::vector<Node> nodes(1); // root
    for (std::size_t pi = 0; pi < patterns.size(); ++pi) {
        const std::string& p = patterns[pi];
        if (p.empty()) continue; // empty patterns match nowhere, by convention
        std::size_t cur = 0;
        for (const char ch : p) {
            auto it = nodes[cur].next.find(ch);
            if (it == nodes[cur].next.end()) {
                const std::size_t created = nodes.size();
                nodes.emplace_back();           // may reallocate; only indices are kept, so it is safe
                nodes[cur].next[ch] = created;  // re-index `cur` after the potential reallocation
                cur = created;
            } else {
                cur = it->second;
            }
        }
        nodes[cur].output.push_back(pi);
    }

    // ---- Compute failure and dictionary-suffix links by BFS over the trie, in order of depth. ----
    // For an edge u --ch--> v, the failure link of v is the goto target of ch from fail[u] (walking
    // up failure links until such a target exists), which is the longest proper suffix of v's
    // matched prefix that is a trie node. The dictionary link of v is fail[v] itself if fail[v]
    // completes a pattern, otherwise fail[v]'s own dictionary link.
    std::queue<std::size_t> bfs;
    bfs.push(0);
    while (!bfs.empty()) {
        const std::size_t u = bfs.front();
        bfs.pop();
        for (const auto& [ch, v] : nodes[u].next) {
            std::size_t f = nodes[u].fail;
            while (f != 0 && nodes[f].next.find(ch) == nodes[f].next.end()) f = nodes[f].fail;
            std::size_t candidate = 0;
            if (const auto it = nodes[f].next.find(ch); it != nodes[f].next.end() && it->second != v)
                candidate = it->second;
            nodes[v].fail = candidate;
            nodes[v].dict = nodes[candidate].output.empty() ? nodes[candidate].dict : candidate;
            bfs.push(v);
        }
    }

    // ---- Scan the text once, following goto / failure transitions and emitting every match. ------
    std::vector<AhoCorasickMatch> matches;
    std::size_t                   state = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        while (state != 0 && nodes[state].next.find(ch) == nodes[state].next.end())
            state = nodes[state].fail;
        if (const auto it = nodes[state].next.find(ch); it != nodes[state].next.end())
            state = it->second;
        // else: no edge from the root either -> stay at the root.

        // Report the pattern(s) ending at `state`, then walk the dictionary-suffix links so that
        // every shorter pattern that is a suffix of the current match is reported too.
        for (std::size_t u = state; u != 0; u = nodes[u].dict)
            for (const std::size_t pi : nodes[u].output)
                matches.push_back(AhoCorasickMatch{i - patterns[pi].size() + 1, pi});
    }
    return matches;
}

} // namespace datamunge::algorithms
