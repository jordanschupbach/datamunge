#pragma once

/// \file dfa_minimization.hpp
/// \brief DFA minimization by the Moore, Hopcroft, and Brzozowski algorithms.
///
/// Every regular language has a *unique* minimal DFA (fewest states). Three classic
/// algorithms compute it:
///
/// - **Moore** (partition refinement): start by separating accepting from non-accepting
///   states, then repeatedly split any block whose members disagree on which block a symbol
///   leads to, until stable. \f$O(kn^2)\f$.
/// - **Hopcroft** (worklist refinement): the same refinement driven by a worklist of splitter
///   sets and predecessor images, giving \f$O(kn\log n)\f$ -- asymptotically optimal.
/// - **Brzozowski**: reverse the automaton, determinize, reverse, determinize. Two rounds of
///   reverse-then-subset-construct yield the minimal DFA directly. Simple but worst-case
///   exponential.
///
/// All three return the same minimal (complete) DFA. This module implements them on the
/// shared automaton types.

#include <datamunge/algorithms/automaton.hpp>
#include <datamunge/algorithms/powerset_construction.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace datamunge::algorithms {

/// \brief Reverse a DFA into an NFA (flip edges; swap start/accept roles).
inline NFA reverse_dfa(const DFA& d) {
    NFA r;
    r.num_states  = d.num_states;
    r.num_symbols = d.num_symbols;
    r.delta.assign(d.num_states, std::vector<std::vector<int>>(d.num_symbols));
    for (int s = 0; s < d.num_states; ++s)
        for (int sym = 0; sym < d.num_symbols; ++sym) {
            int t = d.delta[s][sym];
            if (t >= 0) r.delta[t][sym].push_back(s);
        }
    for (int s = 0; s < d.num_states; ++s)
        if (d.accept[s]) r.start_states.push_back(s);   // accepting states become starts
    r.accept.assign(d.num_states, 0);
    if (d.start >= 0) r.accept[d.start] = 1;            // old start becomes the only accept
    return r;
}

/// \brief Make a DFA's transition function total by routing every \c -1 to a fresh dead state.
inline DFA complete_dfa(const DFA& d) {
    bool need = d.start < 0;
    for (const auto& row : d.delta)
        for (int t : row)
            if (t < 0) need = true;
    if (!need) return d;

    DFA c    = d;
    int dead = c.num_states++;
    for (auto& row : c.delta)
        for (int& t : row)
            if (t < 0) t = dead;
    c.delta.emplace_back(c.num_symbols, dead);   // dead state loops to itself
    c.accept.push_back(0);
    if (c.start < 0) c.start = dead;
    return c;
}

namespace detail {
/// Build the quotient DFA from a state->block labeling, keeping only reachable blocks.
inline DFA build_from_blocks(const DFA& d, const std::vector<int>& block, int numblocks) {
    std::vector<int> rep(numblocks, -1);
    for (int s = 0; s < d.num_states; ++s)
        if (rep[block[s]] < 0) rep[block[s]] = s;

    std::vector<std::vector<int>> bdelta(numblocks, std::vector<int>(d.num_symbols));
    std::vector<char>             bacc(numblocks, 0);
    for (int b = 0; b < numblocks; ++b) {
        bacc[b] = d.accept[rep[b]];
        for (int c = 0; c < d.num_symbols; ++c) bdelta[b][c] = block[d.delta[rep[b]][c]];
    }

    int              bstart = block[d.start];
    std::vector<int> newid(numblocks, -1), order;
    std::queue<int>  q;
    q.push(bstart);
    newid[bstart] = 0;
    order.push_back(bstart);
    int cnt = 1;
    while (!q.empty()) {
        int b = q.front();
        q.pop();
        for (int c = 0; c < d.num_symbols; ++c) {
            int nb = bdelta[b][c];
            if (newid[nb] < 0) { newid[nb] = cnt++; order.push_back(nb); q.push(nb); }
        }
    }

    DFA m;
    m.num_symbols = d.num_symbols;
    m.num_states  = cnt;
    m.start       = 0;
    m.delta.assign(cnt, std::vector<int>(d.num_symbols));
    m.accept.assign(cnt, 0);
    for (int i = 0; i < cnt; ++i) {
        int b       = order[i];
        m.accept[i] = bacc[b];
        for (int c = 0; c < d.num_symbols; ++c) m.delta[i][c] = newid[bdelta[b][c]];
    }
    return m;
}
}  // namespace detail

/// \brief Minimize a DFA by Moore's partition-refinement algorithm.
inline DFA moore_minimize(const DFA& din) {
    DFA d = complete_dfa(din);
    int n = d.num_states;

    std::vector<int> block(n);
    for (int i = 0; i < n; ++i) block[i] = d.accept[i] ? 1 : 0;
    int cur = (std::count(block.begin(), block.end(), 1) && std::count(block.begin(), block.end(), 0)) ? 2 : 1;

    while (true) {
        std::map<std::vector<int>, int> sig;
        std::vector<int>                nb(n);
        for (int s = 0; s < n; ++s) {
            std::vector<int> key{block[s]};
            for (int c = 0; c < d.num_symbols; ++c) key.push_back(block[d.delta[s][c]]);
            auto it = sig.find(key);
            if (it == sig.end()) { int id = static_cast<int>(sig.size()); sig[key] = id; nb[s] = id; }
            else nb[s] = it->second;
        }
        int newnum = static_cast<int>(sig.size());
        block      = nb;
        if (newnum == cur) break;
        cur = newnum;
    }
    return detail::build_from_blocks(d, block, cur);
}

/// \brief Minimize a DFA by Hopcroft's worklist algorithm.
inline DFA hopcroft_minimize(const DFA& din) {
    DFA d = complete_dfa(din);
    int n = d.num_states, K = d.num_symbols;

    std::vector<std::vector<std::vector<int>>> pred(K, std::vector<std::vector<int>>(n));
    for (int s = 0; s < n; ++s)
        for (int c = 0; c < K; ++c) pred[c][d.delta[s][c]].push_back(s);

    std::set<int> F, NF;
    for (int s = 0; s < n; ++s) (d.accept[s] ? F : NF).insert(s);

    std::vector<std::set<int>> P, W;
    if (!F.empty())  { P.push_back(F);  W.push_back(F); }
    if (!NF.empty()) { P.push_back(NF); W.push_back(NF); }

    while (!W.empty()) {
        std::set<int> A = W.back();
        W.pop_back();
        for (int c = 0; c < K; ++c) {
            std::set<int> X;
            for (int t : A)
                for (int s : pred[c][t]) X.insert(s);
            if (X.empty()) continue;

            std::vector<std::set<int>> newP;
            for (auto& Y : P) {
                std::set<int> inter, diff;
                for (int y : Y) (X.count(y) ? inter : diff).insert(y);
                if (!inter.empty() && !diff.empty()) {
                    newP.push_back(inter);
                    newP.push_back(diff);
                    auto wit = std::find(W.begin(), W.end(), Y);
                    if (wit != W.end()) { *wit = inter; W.push_back(diff); }
                    else W.push_back(inter.size() <= diff.size() ? inter : diff);
                } else {
                    newP.push_back(Y);
                }
            }
            P = std::move(newP);
        }
    }

    std::vector<int> block(n);
    for (int b = 0; b < static_cast<int>(P.size()); ++b)
        for (int s : P[b]) block[s] = b;
    return detail::build_from_blocks(d, block, static_cast<int>(P.size()));
}

/// \brief Minimize a DFA by Brzozowski's algorithm: reverse, determinize, reverse, determinize.
inline DFA brzozowski_minimize(const DFA& d) {
    DFA t = powerset_construction(reverse_dfa(d), /*complete=*/true);
    return powerset_construction(reverse_dfa(t), /*complete=*/true);
}

}  // namespace datamunge::algorithms
