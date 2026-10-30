/// @file bakery.cpp
/// @brief Lamport's bakery lock, an exhaustive model checker for the protocol and a random simulator.

#include <datamunge/algorithms/bakery.hpp>

#include <algorithm>
#include <deque>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace datamunge::algorithms {

BakeryLock::BakeryLock(std::size_t threads)
    : n_(threads),
      choosing_(threads ? new std::atomic<bool>[threads] : nullptr),
      number_(threads ? new std::atomic<std::uint64_t>[threads] : nullptr) {
    if (threads == 0) throw std::invalid_argument("BakeryLock: need at least one thread");
    for (std::size_t i = 0; i < n_; ++i) {
        choosing_[i].store(false);
        number_[i].store(0);
    }
}

void BakeryLock::lock(std::size_t id) {
    choosing_[id].store(true);
    std::uint64_t m = 0;
    for (std::size_t j = 0; j < n_; ++j) m = std::max(m, number_[j].load());
    number_[id].store(m + 1);
    std::uint64_t seen = max_ticket_.load();
    while (m + 1 > seen && !max_ticket_.compare_exchange_weak(seen, m + 1)) {
    }
    choosing_[id].store(false);
    for (std::size_t j = 0; j < n_; ++j) {
        if (j == id) continue;
        while (choosing_[j].load()) {
        }
        for (;;) {
            const std::uint64_t nj = number_[j].load(), ni = number_[id].load();
            if (nj == 0 || nj > ni || (nj == ni && j > id)) break;
        }
    }
}

void BakeryLock::unlock(std::size_t id) { number_[id].store(0); }

/// @brief The protocol as a state machine shared by the checker and the simulator.
namespace {

/// @brief One process's program counter, loop index, running maximum, flag and ticket.
struct Proc {
    std::uint8_t  pc       = 0;  ///< 0 idle, 1 read max, 2 write ticket, 3 clear flag, 4 wait flag, 5 wait ticket, 6 critical.
    std::uint8_t  j        = 0;  ///< loop index for pcs 1, 4, 5.
    std::uint32_t tmp      = 0;  ///< running maximum in pc 1.
    bool          choosing = false;  ///< the shared choosing flag.
    std::uint32_t number   = 0;  ///< the shared ticket.
};

/// @brief Whole-system state.
using State = std::vector<Proc>;

/// @brief Phase of a program counter.
BakeryPhase phase_of(std::uint8_t pc) {
    if (pc == 0) return BakeryPhase::Idle;
    if (pc <= 3) return BakeryPhase::Doorway;
    if (pc <= 5) return BakeryPhase::Waiting;
    return BakeryPhase::Critical;
}

/// @brief Skips the process's own index in the wait loop, and the flag wait when it is disabled.
void normalize(State& s, std::size_t i, bool use_choosing) {
    Proc& p = s[i];
    if (p.pc != 4) return;
    if (p.j == i) ++p.j;
    if (p.j >= s.size()) {
        p.pc = 6;
        return;
    }
    if (!use_choosing) p.pc = 5;
}

/// @brief Applies one atomic step of process @p i; returns false if it is a pure wait (no change).
///        Sets @p drew to the ticket written when the step is the ticket write.
bool step(State& s, std::size_t i, bool use_choosing, std::string* action, std::uint32_t* drew) {
    const std::size_t n = s.size();
    Proc&             p = s[i];
    auto say = [&](const std::string& a) {
        if (action) *action = a;
    };
    const std::string me = std::to_string(i);
    switch (p.pc) {
        case 0:
            p.pc  = 1;
            p.j   = 0;
            p.tmp = 0;
            if (use_choosing) {
                p.choosing = true;
                say("choosing[" + me + "] = true");
            } else {
                say("start doorway");
            }
            return true;
        case 1: {
            const std::uint32_t v = s[p.j].number;
            say("read number[" + std::to_string(p.j) + "] = " + std::to_string(v));
            p.tmp = std::max(p.tmp, v);
            if (++p.j == n) p.pc = 2;
            return true;
        }
        case 2:
            p.number = p.tmp + 1;
            if (drew) *drew = p.number;
            say("number[" + me + "] = " + std::to_string(p.number));
            if (use_choosing) {
                p.pc = 3;
            } else {
                p.pc = 4;
                p.j  = 0;
                normalize(s, i, use_choosing);
            }
            return true;
        case 3:
            p.choosing = false;
            p.pc       = 4;
            p.j        = 0;
            say("choosing[" + me + "] = false");
            normalize(s, i, use_choosing);
            return true;
        case 4:
            if (s[p.j].choosing) return false;  // keep waiting
            say("read choosing[" + std::to_string(p.j) + "] = false");
            p.pc = 5;
            return true;
        case 5: {
            const Proc& q = s[p.j];
            if (q.number != 0 && (q.number < p.number || (q.number == p.number && p.j < i))) return false;
            say("read number[" + std::to_string(p.j) + "] = " + std::to_string(q.number) + ", pass");
            ++p.j;
            p.pc = 4;
            normalize(s, i, use_choosing);
            if (p.pc == 6 && action) *action += ", enter critical section";
            return true;
        }
        default:
            p.pc     = 0;
            p.number = 0;
            say("number[" + me + "] = 0, leave critical section");
            return true;
    }
}

/// @brief Byte encoding of a state for hashing.
std::string key_of(const State& s) {
    std::string k;
    for (const auto& p : s) {
        k.push_back(static_cast<char>(p.pc));
        k.push_back(static_cast<char>(p.j));
        k.push_back(static_cast<char>(p.tmp));
        k.push_back(static_cast<char>(p.choosing));
        k.push_back(static_cast<char>(p.number));
    }
    return k;
}

/// @brief Processes in the critical section.
std::size_t critical_count(const State& s) {
    std::size_t c = 0;
    for (const auto& p : s) c += p.pc == 6;
    return c;
}

}  // namespace

BakeryCheck bakery_model_check(const BakeryModelOptions& options) {
    const std::size_t n = options.processes;
    if (n == 0 || n > 4) throw std::invalid_argument("bakery_model_check: processes must be 1 to 4");
    if (options.max_ticket == 0 || options.max_ticket > 200) throw std::invalid_argument("bakery_model_check: max_ticket must be 1 to 200");
    BakeryCheck out;
    struct Parent {
        std::string from;
        BakeryStep  step;
    };
    std::unordered_map<std::string, Parent> parent;
    std::deque<State> queue;
    const State start(n);
    const std::string start_key = key_of(start);
    parent.emplace(start_key, Parent{"", {0, ""}});
    queue.push_back(start);
    std::string violation;
    while (!queue.empty()) {
        const State s = queue.front();
        queue.pop_front();
        const std::string sk = key_of(s);
        const std::size_t crit = critical_count(s);
        out.max_critical = std::max(out.max_critical, crit);
        if (crit > 1 && violation.empty()) {
            out.mutual_exclusion = false;
            violation            = sk;
        }
        for (std::size_t i = 0; i < n; ++i) {
            State         t = s;
            std::string   action;
            std::uint32_t drew = 0;
            if (!step(t, i, options.use_choosing, &action, &drew)) continue;
            ++out.transitions;
            if (drew > options.max_ticket) {
                ++out.cut_by_bound;
                continue;
            }
            std::string tk = key_of(t);
            if (parent.emplace(tk, Parent{sk, {i, action}}).second) queue.push_back(std::move(t));
        }
    }
    out.states = parent.size();
    if (!violation.empty()) {
        for (std::string k = violation; k != start_key; k = parent.at(k).from) out.counterexample.push_back(parent.at(k).step);
        std::reverse(out.counterexample.begin(), out.counterexample.end());
    }
    return out;
}

BakerySimulation bakery_simulate(const BakeryModelOptions& options, std::size_t steps, std::uint64_t seed) {
    const std::size_t n = options.processes;
    if (n == 0 || n > 64) throw std::invalid_argument("bakery_simulate: processes must be 1 to 64");
    BakerySimulation out;
    out.entries.assign(n, 0);
    State s(n);
    for (auto& p : s) p.j = 0;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    for (std::size_t t = 0; t < steps; ++t) {
        const std::size_t i      = pick(rng);
        const bool        before = s[i].pc == 6;
        step(s, i, options.use_choosing, nullptr, nullptr);
        if (!before && s[i].pc == 6) {
            ++out.entries[i];
            out.order.push_back(i);
        }
        out.violations += critical_count(s) > 1;
        std::vector<BakeryPhase>   ph(n);
        std::vector<std::uint32_t> num(n);
        for (std::size_t k = 0; k < n; ++k) {
            ph[k]  = phase_of(s[k].pc);
            num[k] = s[k].number;
        }
        out.phase.push_back(std::move(ph));
        out.number.push_back(std::move(num));
    }
    return out;
}

}  // namespace datamunge::algorithms
