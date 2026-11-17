/// @file ll1_parser.cpp
/// @brief Checked actual FIRST/FOLLOW construction and conflict witnesses with one shared bounded predictive-stack producer.
#include <algorithm>
#include <datamunge/algorithms/ll1_parser.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete grammar/token and actual count/capture checks.
namespace ll1_detail {
/// @brief Add selected logical work without size_t wrap.
/// @param count Mutable actual work/count.
/// @param amount Additional logical count, default1.
/// @throws std::length_error On actual count overflow.
void add(std::size_t& count, std::size_t amount = 1) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("LL1 count geometry");
  count += amount;
}
/// @brief Validate every original grammar symbol and start/index shape before fixed-point construction.
/// @param grammar Borrowed complete original production list/start.
/// @throws std::invalid_argument On undefined/empty/reserved start/lhs/rhs symbols.
/// @throws std::length_error On production-index or complete symbol geometry.
void grammar(const ContextFreeGrammar& grammar) {
  if (grammar.start.empty() || grammar.start == "$" || grammar.productions.empty() ||
      !grammar.is_nonterminal(grammar.start))
    throw std::invalid_argument("LL1 defined nonempty start required");
  if (grammar.productions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("LL1 production index geometry");
  std::size_t symbols = 0;
  for (auto& p : grammar.productions) {
    if (p.lhs.empty() || p.lhs == "$")
      throw std::invalid_argument("LL1 reserved/empty lhs");
    add(symbols, p.rhs.size());
    for (auto& s : p.rhs)
      if (s.empty() || s == "$")
        throw std::invalid_argument("LL1 epsilon must be empty rhs; reserved end marker");
  }
}
/// @brief Validate original tokens and exact finite action/append geometry before copies.
/// @param tokens Borrowed complete original tokens.
/// @param max_steps Actual finite action budget.
/// @throws std::invalid_argument On literal empty/$ tokens.
/// @throws std::length_error On append/action geometry.
void input(const std::vector<std::string>& tokens, std::size_t max_steps) {
  if (tokens.size() == tokens.max_size() || max_steps == std::numeric_limits<std::size_t>::max())
    throw std::length_error("LL1 input/action geometry");
  for (auto& t : tokens)
    if (t.empty() || t == "$")
      throw std::invalid_argument("LL1 reserved/empty input token");
}
/// @brief Admit known actual logical token/stack/action records before copying fields.
/// @param trace Optional mutable complete parse capture.
/// @param amount Complete next logical record geometry.
/// @param cap Actual optional cap,0 off.
/// @throws std::length_error On actual count/quota overflow.
void admit(LL1ParseTrace* trace, std::size_t amount, std::size_t cap) {
  if (!trace)
    return;
  if (cap && (trace->captured_entries > cap || amount > cap - trace->captured_entries))
    throw std::length_error("LL1 actual capture quota");
  add(trace->captured_entries, amount);
}
} // namespace ll1_detail
LL1Parser::LL1Parser(ContextFreeGrammar grammar) : g_(std::move(grammar)) {
  ll1_detail::grammar(g_);
  nonterminals_ = g_.nonterminals();
  compute_first();
  compute_follow();
  build_table();
}
bool LL1Parser::insert_first(const std::string& symbol, const std::string& terminal) {
  const bool added = first_[symbol].insert(terminal).second;
  if (added)
    ll1_detail::add(build_work_.first_insertions);
  return added;
}
bool LL1Parser::insert_follow(const std::string& symbol, const std::string& terminal) {
  const bool added = follow_[symbol].insert(terminal).second;
  if (added)
    ll1_detail::add(build_work_.follow_insertions);
  return added;
}
std::set<std::string> LL1Parser::first_of_seq(const std::vector<std::string>& seq) const {
  std::set<std::string> result;
  bool                  all_nullable = true;
  for (const auto& sym : seq) {
    const auto& fs = first_.at(sym);
    for (const auto& t : fs)
      if (t != kEpsilon)
        result.insert(t);
    if (!fs.count(kEpsilon)) {
      all_nullable = false;
      break;
    }
  }
  if (all_nullable)
    result.insert(kEpsilon);
  return result;
}
void LL1Parser::compute_first() {
  // terminals: FIRST = {self}; nonterminals: start empty.
  for (const auto& t : g_.terminals())
    insert_first(t, t);
  for (const auto& nt : g_.nonterminals())
    first_[nt]; // empty set
  bool changed = true;
  while (changed) {
    ll1_detail::add(build_work_.first_passes);
    changed = false;
    for (const auto& p : g_.productions) {
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
        if (insert_first(p.lhs, t))
          changed = true;
    }
  }
}
void LL1Parser::compute_follow() {
  for (const auto& nt : g_.nonterminals())
    follow_[nt];
  insert_follow(g_.start, kEndMarker);
  bool changed = true;
  while (changed) {
    ll1_detail::add(build_work_.follow_passes);
    changed = false;
    for (const auto& p : g_.productions) {
      for (std::size_t i = 0; i < p.rhs.size(); ++i) {
        const std::string& B = p.rhs[i];
        if (!is_nt(B))
          continue;
        std::vector<std::string> beta(p.rhs.begin() + i + 1, p.rhs.end());
        auto                     fb = first_of_seq(beta);
        for (const auto& t : fb)
          if (t != kEpsilon)
            if (insert_follow(B, t))
              changed = true;
        if (beta.empty() || fb.count(kEpsilon)) {
          for (const auto& t : follow_[p.lhs])
            if (insert_follow(B, t))
              changed = true;
        }
      }
    }
  }
}
void LL1Parser::build_table() {
  for (int i = 0; i < static_cast<int>(g_.productions.size()); ++i) {
    const auto& p  = g_.productions[i];
    auto        fa = first_of_seq(p.rhs);
    for (const auto& a : fa) {
      if (a == kEpsilon)
        continue;
      insert_entry(p.lhs, a, i);
    }
    if (fa.count(kEpsilon)) {
      for (const auto& b : follow_[p.lhs])
        insert_entry(p.lhs, b, i);
    }
  }
}
void LL1Parser::insert_entry(
    const std::string& nonterminal,
    const std::string& lookahead,
    int                production) {
  ll1_detail::add(build_work_.table_attempts);
  auto key   = std::make_pair(nonterminal, lookahead);
  auto found = table_.find(key);
  if (found != table_.end() && found->second != production) {
    ll1_detail::add(build_work_.conflicts);
    conflicts_.push_back({nonterminal, lookahead, found->second, production});
  } else
    table_[key] = production;
}
LL1ParseAnalysis LL1Parser::parse(
    const std::vector<std::string>& tokens,
    std::size_t                     max_steps,
    LL1ParseTrace*                  capture,
    std::size_t                     max_entries) const {
  std::vector<std::string> input = tokens;
  input.push_back(kEndMarker);
  std::vector<std::string> stack{kEndMarker, g_.start};
  if (capture)
    capture->initial_stack = stack;
  LL1ParseAnalysis out;
  std::size_t      ip = 0;
  while (out.steps < max_steps) {
    const auto&    top        = stack.back();
    const auto&    look       = input[ip];
    LL1ParseAction action     = LL1ParseAction::Reject;
    LL1ParseStatus status     = LL1ParseStatus::StepLimit;
    int            production = -1;
    bool           finished   = false;
    std::size_t    after_size = stack.size();
    if (top == kEndMarker) {
      finished = true;
      if (look == kEndMarker && ip == tokens.size()) {
        action = LL1ParseAction::Accept;
        status = LL1ParseStatus::Accepted;
      } else
        status = LL1ParseStatus::ExtraInput;
    } else if (!is_nt(top)) {
      if (top == look) {
        action = LL1ParseAction::Match;
        --after_size;
      } else {
        status   = LL1ParseStatus::TerminalMismatch;
        finished = true;
      }
    } else {
      auto found = table_.find({top, look});
      if (found == table_.end()) {
        status   = LL1ParseStatus::NoEntry;
        finished = true;
      } else {
        production      = found->second;
        action          = LL1ParseAction::Expand;
        const auto size = g_.productions[production].rhs.size();
        --after_size;
        if (size > stack.max_size() - after_size)
          throw std::length_error("LL1 actual stack geometry");
        after_size += size;
      }
    }
    LL1ParseEvent event;
    if (capture) {
      std::size_t records = 1;
      ll1_detail::add(records, stack.size());
      ll1_detail::add(records, after_size);
      ll1_detail::admit(capture, records, max_entries);
      event.step         = out.steps;
      event.input_offset = ip;
      event.top          = top;
      event.lookahead    = look;
      event.action       = action;
      event.production   = production;
      event.before       = stack;
    }
    if (action == LL1ParseAction::Match) {
      stack.pop_back();
      ++ip;
    } else if (action == LL1ParseAction::Expand) {
      stack.pop_back();
      const auto& rhs = g_.productions[production].rhs;
      for (auto it = rhs.rbegin(); it != rhs.rend(); ++it)
        stack.push_back(*it);
    }
    ++out.steps;
    out.consumed        = ip;
    out.remaining_stack = stack.size();
    out.maximum_stack   = std::max(out.maximum_stack, stack.size());
    if (capture) {
      event.after = stack;
      capture->events.push_back(std::move(event));
    }
    if (finished) {
      out.status   = status;
      out.accepted = status == LL1ParseStatus::Accepted;
      return out;
    }
  }
  out.status          = LL1ParseStatus::StepLimit;
  out.consumed        = ip;
  out.remaining_stack = stack.size();
  return out;
}
LL1ParseAnalysis LL1Parser::analyze(const std::vector<std::string>& tokens, std::size_t max_steps)
    const {
  if (!is_ll1())
    throw std::logic_error("LL1 parsing refuses conflicting table");
  ll1_detail::input(tokens, max_steps);
  return parse(tokens, max_steps, nullptr, 0);
}
bool LL1Parser::recognize(const std::vector<std::string>& tokens) const {
  auto out = analyze(tokens);
  if (out.status == LL1ParseStatus::StepLimit)
    throw std::length_error("LL1 recognition action budget exhausted");
  return out.accepted;
}
LL1ParseTrace LL1Parser::trace(
    const std::vector<std::string>& tokens,
    std::size_t                     max_steps,
    std::size_t                     max_entries) const {
  if (!is_ll1())
    throw std::logic_error("LL1 tracing refuses conflicting table");
  ll1_detail::input(tokens, max_steps);
  LL1ParseTrace out;
  std::size_t   records = tokens.size();
  ll1_detail::add(records, 2);
  ll1_detail::admit(&out, records, max_entries);
  out.tokens   = tokens;
  out.analysis = parse(tokens, max_steps, &out, max_entries);
  return out;
}
ContextFreeGrammar ll1_expression_grammar(bool left_recursive) {
  ContextFreeGrammar g;
  g.start = "E";
  if (left_recursive)
    g.productions = {
        {"E", {"E", "+", "T"}},
        {"E", {"T"}},
        {"T", {"T", "*", "F"}},
        {"T", {"F"}},
        {"F", {"(", "E", ")"}},
        {"F", {"id"}}};
  else
    g.productions = {
        {"E", {"T", "E'"}},
        {"E'", {"+", "T", "E'"}},
        {"E'", {}},
        {"T", {"F", "T'"}},
        {"T'", {"*", "F", "T'"}},
        {"T'", {}},
        {"F", {"(", "E", ")"}},
        {"F", {"id"}}};
  return g;
}
} // namespace datamunge::algorithms
