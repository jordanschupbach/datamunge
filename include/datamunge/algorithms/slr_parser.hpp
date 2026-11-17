#pragma once

/// \file slr_parser.hpp
/// \brief A Simple LR (SLR(1)) bottom-up shift-reduce parser.
///
/// LR parsers work *bottom-up*: instead of predicting productions from the top like LL, they
/// shift input onto a stack and *reduce* a matched right-hand side to its nonterminal once the
/// whole thing is on top. The engine is a DFA over *LR(0) items* -- productions with a dot,
/// \f$A\to\alpha\bullet\beta\f$ -- whose states are built by *closure* (a dot before a
/// nonterminal \f$B\f$ pulls in every \f$B\to\bullet\gamma\f$) and *goto* (advancing the dot
/// over a symbol). From this automaton we build an ACTION/GOTO table:
/// - a dot before a terminal \f$a\f$ with \f$\mathrm{goto}=j\f$ gives *shift j*;
/// - a completed item \f$A\to\alpha\bullet\f$ gives *reduce* \f$A\to\alpha\f$.
/// *SLR* resolves when to reduce with the simplest rule: reduce on lookahead \f$a\f$ only if
/// \f$a\in\mathrm{FOLLOW}(A)\f$. A grammar is *SLR(1)* exactly when the resulting table has no
/// shift/reduce or reduce/reduce conflicts. LR parsers accept a strictly larger class of
/// grammars than LL(1) -- including left-recursive ones, which top-down parsers cannot handle.
///
/// This implementation augments the grammar, builds the LR(0) automaton and SLR table, reports
/// whether the grammar is SLR(1), and recognises input strings.

#include <cstddef>
#include <datamunge/algorithms/context_free_grammar.hpp>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {

/// @brief Complete grammar/construction and actual parser limits; zero disables each policy.
/// @par Example
/// @code
/// const dm::algorithms::SLROptions value{};
/// std::print("{}\n",value.max_states); // 10000
/// @endcode
struct SLROptions {
  std::size_t max_grammar_bytes{1000000}; ///< Complete production/start byte geometry.
  std::size_t max_states{10000};          ///< Actual stored LR states.
  std::size_t max_items{
      100000}; ///< Actual items in a temporary closure and total stored states separately.
  std::size_t max_table_entries{1000000}; ///< Actual ACTION plus GOTO keys.
  std::size_t max_construction_work{
      10000000}; ///< Actual selected closure/rule/state visits, not every primitive string comparison.
  std::size_t max_input_tokens{100000}; ///< Complete token count.
  std::size_t max_actions{1000000}; ///< Actual parse ACTION lookups including final accept/reject.
  std::size_t max_history_values{
      1000000}; ///< Actual copied stack entries, string bytes and records.
};
/// @brief One actual LR state item as production index and dot offset.
/// @par Example
/// @code
/// const dm::algorithms::SLRItemRecord value{};
/// std::print("{}\n",value.production); // 0
/// @endcode
struct SLRItemRecord {
  int production{}; ///< Actual augmented production index; zero is synthetic start.
  int dot{};        ///< Actual recognized RHS prefix length.
};
/// @brief Actual constructed table action.
/// @par Example
/// @code
/// const dm::algorithms::SLRActionRecord value{};
/// std::print("{}\n",value.state); // 0
/// @endcode
struct SLRActionRecord {
  int         state{};  ///< Actual source state.
  std::string terminal; ///< Actual lookahead key.
  std::string action;   ///< Actual shift, reduce or accept kind.
  int         value{};  ///< Actual shift state or reduction production index; ignored for accept.
};
/// @brief Owned snapshot of actual completed LR construction.
/// @par Example
/// @code
/// const dm::algorithms::SLRConstruction value{};
/// std::print("{}\n",value.conflicts); // 0
/// @endcode
struct SLRConstruction {
  std::vector<Production> productions; ///< Actual augmented production order.
  std::map<std::string, std::set<std::string>>
      first; ///< Actual fixed-point FIRST sets; empty string denotes epsilon.
  std::map<std::string, std::set<std::string>>
      follow; ///< Actual fixed-point FOLLOW sets, including synthetic start.
  std::vector<std::vector<SLRItemRecord>>
      states; ///< Actual stored LR item sets in automaton order.
  std::vector<SLRActionRecord>
      actions; ///< Actual retained ACTION table; conflicted entries are not safe parsing decisions.
  std::size_t conflicts{}; ///< Actual attempts to install different actions into an existing key.
  std::size_t selected_work{}; ///< Actual selected construction visit count.
};
/// @brief One actual parser action with the real post-action state stack.
/// @par Example
/// @code
/// const dm::algorithms::SLRParseStep value{};
/// std::print("{}\n",value.input_offset); // 0
/// @endcode
struct SLRParseStep {
  std::size_t      input_offset{}; ///< Actual post-action consumed token count.
  std::string      lookahead;      ///< Actual token at the lookup before this action.
  std::string      action;         ///< Actual shift, reduce, accept or reject.
  int              value{};        ///< Actual shift-state/reduction index, otherwise zero.
  std::vector<int> states; ///< Actual state stack after the action (unchanged on accept/reject).
};
/// @brief Owned actual parser outcome, work and optional histories.
/// @par Example
/// @code
/// const dm::algorithms::SLRParseRun value{};
/// std::print("{}\n",value.accepted); // false
/// @endcode
struct SLRParseRun {
  bool        accepted{}; ///< Actual explicit accept action reached after all supplied tokens.
  std::size_t consumed{}; ///< Actual shifted input tokens.
  std::size_t actions{};  ///< Actual ACTION lookups including accept/reject.
  std::vector<int>
      depths; ///< Optional actual initial depth plus each shift/reduce post-depth; legacy trace convention.
  std::vector<SLRParseStep>
      steps; ///< Optional actual action/state snapshots, including accept/reject.
};
/// \brief A Simple LR (SLR(1)) parser: LR(0) automaton, SLR table, and recognition.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};
/// std::print("{}\n",dm::algorithms::SLRParser(g).is_slr()); // true
/// @endcode
class SLRParser {
 public:
  /// @brief Empty-string marker for nullable FIRST sets; not a supplied grammar symbol.
  static constexpr const char* kEpsilon = "";
  /// @brief Reserved appended end marker; forbidden as grammar/input token.
  static constexpr const char* kEndMarker = "$";
  /// @brief Reserved synthetic start nonterminal.
  static constexpr const char* kAugStart = "__START__";

  /// @brief Build a checked SLR(1) LR(0)/FOLLOW automaton from a complete CFG.
  /// Owns augmented productions and actual tables. Empty RHS denotes epsilon; empty symbol strings, reserved $ and __START__ symbols and undefined start are rejected. Nonterminals are exactly production LHS symbols. Conflicts are retained as diagnostic counts and is_slr becomes false; parsing such a grammar throws. Complete grammar bytes and actual states/items/table/work are bounded. Construction can be exponential in grammar size; this implementation uses explicit ordered sets and full fixed-point passes. Const completed-parser queries use no mutable shared state.
  /// @param grammar Complete borrowed original grammar.
  /// @param options Per-instance complete construction and actual parsing policies; defaults shown in SLROptions.
  /// @throws std::invalid_argument Malformed/reserved symbols or undefined start.
  /// @throws std::length_error Construction geometry, actual work or counter/container capacity exceeded.
  /// @par Example
  /// @code
  /// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};
  /// std::print("{}\n",dm::algorithms::SLRParser(g).is_slr()); // true
  /// @endcode
  /// @par References
  /// - F. L. DeRemer, Practical Translators for LR(k) Languages, MIT PhD thesis1969; SLR FOLLOW-based reduction tables.
  /// @see inspection(), parse_trace()
  explicit SLRParser(const ContextFreeGrammar& grammar, SLROptions options = {})
      : options_(options) {
    validate_grammar(grammar);
    // Augmented grammar: production 0 is __START__ -> start.
    aug_start_prod_ = {kAugStart, {grammar.start}};
    prods_.push_back(aug_start_prod_);
    for (const auto& p : grammar.productions)
      prods_.push_back(p);
    start_ = grammar.start;

    collect_symbols();
    compute_first();
    compute_follow();
    build_automaton();
    build_table();
  }

  /// \brief True if the grammar is SLR(1) (no table conflicts).
  /// @return Whether the completed table has no conflicting action attempts.
  /// @par Example
  /// @code
  /// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};dm::algorithms::SLRParser parser(g);
  /// std::print("{}\n",parser.is_slr()); // true
  /// @endcode
  bool is_slr() const { return !has_conflict_; }
  /// \brief Number of LR(0) states in the automaton.
  /// @return Actual number of stored LR item sets.
  /// @par Example
  /// @code
  /// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};dm::algorithms::SLRParser parser(g);
  /// std::print("{}\n",parser.state_count()); // 3
  /// @endcode
  std::size_t state_count() const { return states_.size(); }

  /// @brief Recognize complete tokens through actual SLR ACTION/GOTO execution.
  /// Requires a conflict-free constructed table; ambiguous/non-SLR grammars throw instead of silently using the first conflicting action. Complete token bound checked first; reserved end marker in supplied tokens rejects. No caller token is mutated. Unknown terminals reject. Lookups/reductions are bounded to terminate nullable cycles or unproductive grammars explicitly. State stack is invocation-local and independent const calls share no mutable parse state. Captures are actual post-action states, not replay. Runtime is proportional to actual actions times ordered-map lookup cost, plus captured stack/string copies; no unconditional linear-time promise for arbitrary grammar reductions.
  /// @param tokens Complete borrowed terminal token sequence; omit synthetic end marker.
  /// @return Owned accepted flag, consumed count, actual lookup count and optional depths/states.
  /// @throws std::logic_error Grammar has conflicting table actions or internal stack invariant failed.
  /// @throws std::length_error Complete input, actual action/stack/capture or counter capacity exceeded.
  /// @par Example
  /// @code
  /// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};
  /// auto r=dm::algorithms::SLRParser(g).parse_run({"id"});
  /// std::print("{}\n",r.accepted); // true
  /// @endcode
  /// @see inspection()
  SLRParseRun parse_run(const std::vector<std::string>& tokens) const {
    return parse(tokens, false);
  }
  /// @brief Recognize complete tokens through actual SLR ACTION/GOTO execution with captured stacks.
  /// Requires a conflict-free constructed table; ambiguous/non-SLR grammars throw instead of silently using the first conflicting action. Complete token bound checked first; reserved end marker in supplied tokens rejects. No caller token is mutated. Unknown terminals reject. Lookups/reductions are bounded to terminate nullable cycles or unproductive grammars explicitly. State stack is invocation-local and independent const calls share no mutable parse state. Captures are actual post-action states, not replay. Runtime is proportional to actual actions times ordered-map lookup cost, plus captured stack/string copies; no unconditional linear-time promise for arbitrary grammar reductions.
  /// @param tokens Complete borrowed terminal token sequence; omit synthetic end marker.
  /// @return Owned accepted flag, consumed count, actual lookup count and optional depths/states.
  /// @throws std::logic_error Grammar has conflicting table actions or internal stack invariant failed.
  /// @throws std::length_error Complete input, actual action/stack/capture or counter capacity exceeded.
  /// @par Example
  /// @code
  /// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};
  /// auto r=dm::algorithms::SLRParser(g).parse_trace({"id"});
  /// std::print("{}\n",r.accepted); // true
  /// @endcode
  /// @see inspection()
  SLRParseRun parse_trace(const std::vector<std::string>& tokens) const {
    return parse(tokens, true);
  }
  /// @brief Recognize complete tokens through the same actual native parse producer.
  /// Uses default per-instance controls and does not retain optional states. Const independent calls use local parse storage.
  /// @param tokens Complete borrowed terminals, without synthetic end marker.
  /// @return Whether the complete token sequence was accepted.
  /// @throws std::logic_error Conflicted grammar or internal invariant failure.
  /// @throws std::length_error Input/actions/counter policies exceeded.
  /// @par Example
  /// @code
  /// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};
  /// std::print("{}\n",dm::algorithms::SLRParser(g).recognize({"id"})); // true
  /// @endcode
  /// @see parse_run()
  bool recognize(const std::vector<std::string>& tokens) const {
    return parse_run(tokens).accepted;
  }
  /// @brief Return actual accepted parse stack depths through the preserved API.
  /// Actual initial depth then post-shift/reduce depths; accept adds no extra depth. Returns empty on rejection. Uses the same actual capturing producer.
  /// @param tokens Complete borrowed terminals.
  /// @return Owned actual depths or empty on rejection.
  /// @throws std::logic_error Conflicted grammar/invariant failure.
  /// @throws std::length_error Input/actions/capture bound exceeded.
  /// @par Example
  /// @code
  /// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};
  /// std::print("{}\n",dm::algorithms::SLRParser(g).stack_depth_trace({"id"}).front()); // 1
  /// @endcode
  /// @see parse_trace()
  std::vector<int> stack_depth_trace(const std::vector<std::string>& tokens) const {
    auto r = parse_trace(tokens);
    return r.accepted ? std::move(r.depths) : std::vector<int>{};
  }
  /// @brief Copy the actual completed LR construction for inspection.
  /// Includes augmented productions, fixed-point sets, actual states and retained actions. Copies own all strings/items, so no parser lifetime dependency. O(construction size), no mutation/shared state.
  /// @return Owned snapshot of actual constructed structures.
  /// @par Example
  /// @code
  /// dm::algorithms::ContextFreeGrammar g{{{"S",{"id"}}},"S"};
  /// std::print("{}\n",dm::algorithms::SLRParser(g).inspection().productions.size()); // 2
  /// @endcode
  /// @see parse_trace()
  SLRConstruction inspection() const {
    SLRConstruction r;
    r.productions   = prods_;
    r.first         = first_;
    r.follow        = follow_;
    r.conflicts     = conflicts_;
    r.selected_work = construction_work_;
    for (const auto& state : states_) {
      std::vector<SLRItemRecord> items;
      for (const auto& i : state)
        items.push_back({i.prod, i.dot});
      r.states.push_back(std::move(items));
    }
    for (const auto& [key, a] : action_)
      r.actions.push_back(
          {key.first,
           key.second,
           a.type == Action::Shift    ? "shift"
           : a.type == Action::Reduce ? "reduce"
                                      : "accept",
           a.value});
    return r;
  }

 private:
  /// @brief Private immutable LR production/dot identity.
  struct Item {
    int prod; ///< Actual augmented production index.
    int dot;  ///< Actual RHS prefix length.

    /// @brief Order LR item identities by production then dot.
    /// @param o Other item.
    /// @return Strict identity ordering.
    bool operator<(const Item& o) const { return prod < o.prod || (prod == o.prod && dot < o.dot); }
    /// @brief Compare LR item identities.
    /// @param o Other item.
    /// @return Same production and dot.
    bool operator==(const Item& o) const { return prod == o.prod && dot == o.dot; }
  };
  /// @brief Private installed table action and its actual target.
  struct Action {
    /// @brief Kind of installed LR action.
    enum Type {
      Shift,  ///< Consume token and push destination state.
      Reduce, ///< Pop RHS and push LHS goto.
      Accept  ///< Complete augmented start at end marker.
    };
    Type type;  ///< Actual installed action kind.
    int  value; ///< Actual shift destination or reduction production index.
  };

  std::vector<Production> prods_;          ///< Actual owned augmented production order.
  Production              aug_start_prod_; ///< Actual owned synthetic start rule.
  std::string             start_;          ///< Original owned start symbol.
  std::set<std::string>   terminals_;      ///< Actual terminal symbols including end marker.
  std::set<std::string>   nonterminals_;   ///< Actual augmented LHS symbols.
  std::map<std::string, std::set<std::string>> first_;  ///< Actual fixed-point FIRST sets.
  std::map<std::string, std::set<std::string>> follow_; ///< Actual fixed-point FOLLOW sets.
  std::vector<std::set<Item>> states_; ///< Actual owned LR item sets in discovery order.
  std::map<std::pair<int, std::string>, int>    transitions_; ///< Actual LR symbol transitions.
  std::map<std::pair<int, std::string>, Action> action_; ///< Actual retained lookahead actions.
  std::map<std::pair<int, std::string>, int>    goto_;   ///< Actual nonterminal state transitions.
  bool       has_conflict_ = false; ///< At least one actual different-action collision.
  SLROptions options_;              ///< Immutable invocation policies after construction.
  mutable std::size_t
      construction_work_{};    ///< Mutated only during construction, never by const public queries.
  std::size_t stored_items_{}; ///< Actual aggregate item count across stored states.
  std::size_t conflicts_{};    ///< Actual different-action insertion conflicts.

  /// @brief Count selected construction visits before counter or policy overflow.
  void visit() const {
    if (construction_work_ == std::numeric_limits<std::size_t>::max() ||
        (options_.max_construction_work && construction_work_ >= options_.max_construction_work))
      throw std::length_error("SLR: construction work");
    ++construction_work_;
  }
  /// @brief Validate complete grammar and reserved symbol geometry.
  /// @param g Borrowed original grammar.
  void validate_grammar(const ContextFreeGrammar& g) {
    std::size_t bytes = 0;
    /// @brief Count complete grammar bytes before overflow.
    /// @param n Additional complete byte count.
    auto add = [&](std::size_t n) {
      if (n > std::numeric_limits<std::size_t>::max() - bytes)
        throw std::length_error("SLR: grammar capacity");
      bytes += n;
      if (options_.max_grammar_bytes && bytes > options_.max_grammar_bytes)
        throw std::length_error("SLR: grammar bytes");
    };
    /// @brief Validate one complete grammar symbol and count its bytes.
    /// @param x Actual complete symbol.
    auto symbol = [&](const std::string& x) {
      if (x.empty() || x == kEndMarker || x == kAugStart)
        throw std::invalid_argument("SLR: reserved/empty grammar symbol");
      add(x.size());
    };
    symbol(g.start);
    bool start = false;
    if (g.productions.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::length_error("SLR: production indices");
    for (const auto& p : g.productions) {
      symbol(p.lhs);
      if (p.lhs == g.start)
        start = true;
      if (p.rhs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("SLR: dot index");
      for (const auto& x : p.rhs)
        symbol(x);
    }
    if (!start)
      throw std::invalid_argument("SLR: undefined start");
  }
  /// @brief Bound one actual stored LR state and aggregate item geometry.
  void store_state() {
    const auto n = states_.back().size();
    if (states_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        (options_.max_states && states_.size() > options_.max_states) ||
        n > std::numeric_limits<std::size_t>::max() - stored_items_)
      throw std::length_error("SLR: states/items");
    stored_items_ += n;
    if (options_.max_items && stored_items_ > options_.max_items)
      throw std::length_error("SLR: stored items");
  }
  /// @brief Execute one actual checked parse; all runtime storage is local.
  /// @param tokens Complete supplied terminals.
  /// @param capture Retain actual states/depths.
  /// @return Actual accepted/rejected result.
  SLRParseRun parse(const std::vector<std::string>& tokens, bool capture) const {
    if (has_conflict_)
      throw std::logic_error("SLR: conflicted table");
    if (options_.max_input_tokens && tokens.size() > options_.max_input_tokens)
      throw std::length_error("SLR: input tokens");
    SLRParseRun r;
    for (const auto& t : tokens)
      if (t == kEndMarker)
        return r;
    std::vector<int> stack{0};
    if (capture)
      r.depths.push_back(1);
    std::size_t retained = capture ? 1 : 0;
    /// @brief Retain the actual current producing transition with checked copy geometry.
    auto record = [&](const std::string& look, const std::string& action, int value) {
      if (capture) {
        for (const auto n : {std::size_t{1}, look.size(), action.size(), stack.size()}) {
          if (n > std::numeric_limits<std::size_t>::max() - retained)
            throw std::length_error("SLR: history capacity");
          retained += n;
          if (options_.max_history_values && retained > options_.max_history_values)
            throw std::length_error("SLR: history");
        }
        r.steps.push_back({r.consumed, look, action, value, stack});
      }
    };
    for (;;) {
      if (r.actions == std::numeric_limits<std::size_t>::max() ||
          (options_.max_actions && r.actions >= options_.max_actions))
        throw std::length_error("SLR: parse actions");
      ++r.actions;
      const std::string look = r.consumed < tokens.size() ? tokens[r.consumed] : kEndMarker;
      const auto        it   = action_.find({stack.back(), look});
      if (it == action_.end()) {
        record(look, "reject", 0);
        return r;
      }
      const auto act = it->second;
      if (act.type == Action::Accept) {
        r.accepted = r.consumed == tokens.size();
        record(look, "accept", 0);
        return r;
      }
      if (act.type == Action::Shift) {
        stack.push_back(act.value);
        ++r.consumed;
        record(look, "shift", act.value);
      } else {
        const auto& p = prods_.at(act.value);
        if (p.rhs.size() >= stack.size())
          throw std::logic_error("SLR: reduction stack");
        stack.resize(stack.size() - p.rhs.size());
        const auto g = goto_.find({stack.back(), p.lhs});
        if (g == goto_.end())
          throw std::logic_error("SLR: missing goto");
        stack.push_back(g->second);
        record(look, "reduce", act.value);
      }
      if (capture) {
        if (stack.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::length_error("SLR: depth index");
        r.depths.push_back(static_cast<int>(stack.size()));
      }
    }
  }

  /// @brief Check constructed nonterminal membership.
  /// @param s Actual grammar symbol.
  /// @return Whether s appears as an augmented LHS.
  bool is_nt(const std::string& s) const { return nonterminals_.count(s) > 0; }

  /// @brief Collect actual augmented terminal/nonterminal sets.
  void collect_symbols() {
    for (const auto& p : prods_)
      nonterminals_.insert(p.lhs);
    for (const auto& p : prods_)
      for (const auto& s : p.rhs)
        if (!nonterminals_.count(s))
          terminals_.insert(s);
    terminals_.insert(kEndMarker);
  }

  /// @brief Reach the actual monotone nullable FIRST fixed point.
  void compute_first() {
    for (const auto& t : terminals_)
      first_[t] = {t};
    for (const auto& nt : nonterminals_)
      first_[nt];
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& p : prods_) {
        visit();
        std::set<std::string> add;
        if (p.rhs.empty()) {
          add.insert(kEpsilon);
        } else {
          bool all_nullable = true;
          for (const auto& sym : p.rhs) {
            for (const auto& t : first_[sym])
              if (t != kEpsilon)
                add.insert(t);
            if (!first_[sym].count(kEpsilon)) {
              all_nullable = false;
              break;
            }
          }
          if (all_nullable)
            add.insert(kEpsilon);
        }
        for (const auto& t : add)
          if (first_[p.lhs].insert(t).second)
            changed = true;
      }
    }
  }

  /// @brief Reach the actual monotone FOLLOW fixed point with end marker.
  void compute_follow() {
    for (const auto& nt : nonterminals_)
      follow_[nt];
    follow_[kAugStart].insert(kEndMarker);
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& p : prods_) {
        visit();
        for (std::size_t i = 0; i < p.rhs.size(); ++i) {
          const std::string& B = p.rhs[i];
          if (!is_nt(B))
            continue;
          bool all_nullable = true;
          for (std::size_t j = i + 1; j < p.rhs.size(); ++j) {
            for (const auto& t : first_[p.rhs[j]])
              if (t != kEpsilon)
                if (follow_[B].insert(t).second)
                  changed = true;
            if (!first_[p.rhs[j]].count(kEpsilon)) {
              all_nullable = false;
              break;
            }
          }
          if (all_nullable) {
            for (const auto& t : follow_[p.lhs])
              if (follow_[B].insert(t).second)
                changed = true;
          }
        }
      }
    }
  }

  /// @brief Construct actual LR item closure with checked selected visits.
  /// @param items Owned seed item set.
  /// @return Owned actual closed item set.
  std::set<Item> closure(std::set<Item> items) const {
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& it : std::set<Item>(items)) {
        visit();
        if (options_.max_items && items.size() > options_.max_items)
          throw std::length_error("SLR: closure items");
        const Production& p = prods_[it.prod];
        if (it.dot < static_cast<int>(p.rhs.size())) {
          const std::string& B = p.rhs[it.dot];
          if (nonterminals_.count(B)) {
            for (int i = 0; i < static_cast<int>(prods_.size()); ++i)
              if (prods_[i].lhs == B)
                if (items.insert({i, 0}).second)
                  changed = true;
          }
        }
      }
    }
    return items;
  }

  /// @brief Advance actual items over one symbol and close.
  /// @param I Borrowed actual source item set.
  /// @param X Borrowed symbol to cross.
  /// @return Owned actual closed destination set.
  std::set<Item> goto_set(const std::set<Item>& I, const std::string& X) const {
    std::set<Item> moved;
    for (const auto& it : I) {
      visit();
      const Production& p = prods_[it.prod];
      if (it.dot < static_cast<int>(p.rhs.size()) && p.rhs[it.dot] == X)
        moved.insert({it.prod, it.dot + 1});
    }
    return closure(moved);
  }

  /// @brief Search stored actual states by exact item identity.
  /// @param s Borrowed actual closed item set.
  /// @return Existing int state index, or -1.
  int find_state(const std::set<Item>& s) const {
    for (int i = 0; i < static_cast<int>(states_.size()); ++i)
      if ((visit(), states_[i] == s))
        return i;
    return -1;
  }

  /// @brief Construct actual LR states/transitions in deterministic symbol order.
  void build_automaton() {
    states_.push_back(closure({{0, 0}}));
    store_state(); // start: __START__ -> . start
    for (std::size_t i = 0; i < states_.size(); ++i) {
      // Gather symbols appearing right of a dot.
      std::set<std::string> syms;
      for (const auto& it : states_[i]) {
        const Production& p = prods_[it.prod];
        if (it.dot < static_cast<int>(p.rhs.size()))
          syms.insert(p.rhs[it.dot]);
      }
      for (const auto& X : syms) {
        auto j_set = goto_set(states_[i], X);
        if (j_set.empty())
          continue;
        int j = find_state(j_set);
        if (j == -1) {
          states_.push_back(j_set);
          store_state();
          j = static_cast<int>(states_.size()) - 1;
        }
        transitions_[{static_cast<int>(i), X}] = j;
      }
    }
  }

  /// @brief Install actual FOLLOW-filtered reductions and terminal shifts.
  void build_table() {
    for (int i = 0; i < static_cast<int>(states_.size()); ++i) {
      for (const auto& it : states_[i]) {
        const Production& p = prods_[it.prod];
        if (it.dot < static_cast<int>(p.rhs.size())) {
          const std::string& X  = p.rhs[it.dot];
          auto               tr = transitions_.find({i, X});
          if (tr == transitions_.end())
            continue;
          if (terminals_.count(X)) {
            set_action(i, X, {Action::Shift, tr->second});
          } else {
            if (options_.max_table_entries &&
                action_.size() + goto_.size() >= options_.max_table_entries &&
                !goto_.contains({i, X}))
              throw std::length_error("SLR: table");
            goto_[{i, X}] = tr->second;
          }
        } else {
          // completed item
          if (p.lhs == kAugStart) {
            set_action(i, kEndMarker, {Action::Accept, 0});
          } else {
            for (const auto& a : follow_[p.lhs])
              set_action(i, a, {Action::Reduce, it.prod});
          }
        }
      }
    }
  }

  /// @brief Install an actual action or record a conflicting attempt.
  /// @param state Actual source index.
  /// @param a Actual lookahead terminal.
  /// @param act Actual proposed action.
  void set_action(int state, const std::string& a, Action act) {
    visit();
    auto key = std::make_pair(state, a);
    auto it  = action_.find(key);
    if (it != action_.end()) {
      // Conflict unless it is the identical action.
      if (it->second.type != act.type || it->second.value != act.value) {
        has_conflict_ = true;
        ++conflicts_;
      }
    } else {
      if (options_.max_table_entries && action_.size() + goto_.size() >= options_.max_table_entries)
        throw std::length_error("SLR: table");
      action_[key] = act;
    }
  }
};

} // namespace datamunge::algorithms
