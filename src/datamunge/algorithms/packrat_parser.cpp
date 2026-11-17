/// @file packrat_parser.cpp
/// @brief Checked owned byte PEG interpretation, actual rule memo/choice records and bounded work.
#include <algorithm>
#include <datamunge/algorithms/packrat_parser.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
PegExpr peg_lit(std::string s) {
  return {PegOp::Literal, std::move(s), {}};
}
PegExpr peg_ref(std::string name) {
  return {PegOp::Ref, std::move(name), {}};
}
PegExpr peg_seq(std::vector<PegExpr> es) {
  return {PegOp::Seq, "", std::move(es)};
}
PegExpr peg_choice(std::vector<PegExpr> es) {
  return {PegOp::Choice, "", std::move(es)};
}
PegExpr peg_star(PegExpr e) {
  return {PegOp::Star, "", {std::move(e)}};
}
PegExpr peg_plus(PegExpr e) {
  return {PegOp::Plus, "", {std::move(e)}};
}
PegExpr peg_opt(PegExpr e) {
  return {PegOp::Opt, "", {std::move(e)}};
}
PegExpr peg_and(PegExpr e) {
  return {PegOp::And, "", {std::move(e)}};
}
PegExpr peg_not(PegExpr e) {
  return {PegOp::Not, "", {std::move(e)}};
}
/// @brief Private checked counters and exception-safe actual frame accounting.
namespace {
/// @brief Add actual nonnegative work/history counts without overflow.
/// @param a Existing count.
/// @param b Added count.
/// @return Checked sum.
/// @throws std::length_error If addition overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("PEG: count geometry");
  return a + b;
}
/// @brief Scope one actual rule/expression frame and restore depth on every exit.
class Frame {
  std::size_t& depth_; ///< Actual caller-owned current-depth counter.
 public:
  /// @brief Admit one actual frame before entering its expression/rule work.
  /// @param depth Actual current depth, updated only on successful admission.
  /// @param maximum Actual high-water mark, updated on successful entry.
  /// @param cap Optional actual frame cap,0 disables.
  /// @throws std::length_error If depth geometry or selected policy fails.
  Frame(std::size_t& depth, std::size_t& maximum, std::size_t cap) : depth_(depth) {
    auto next = add(depth, 1);
    if (cap && next > cap)
      throw std::length_error("PEG: actual recursion depth quota");
    depth   = next;
    maximum = std::max(maximum, depth);
  }
  /// @brief Restore the actual caller depth during normal return or exception unwind.
  ~Frame() { --depth_; }
};
/// @brief Increment one actual work counter under optional policy and legacy-long geometry.
/// @param value Actual counter to increment on success.
/// @param cap Optional actual-work cap,0 disables.
/// @param legacy_long True additionally requires representability by old long accessors.
/// @throws std::length_error If count geometry or selected policy fails.
void increment(std::size_t& value, std::size_t cap, bool legacy_long = false) {
  auto next = add(value, 1);
  if ((cap && next > cap) ||
      (legacy_long && next > static_cast<unsigned long>(std::numeric_limits<long>::max())))
    throw std::length_error("PEG: actual work quota/count geometry");
  value = next;
}
} // namespace
PackratParser::PackratParser(
    std::map<std::string, PegExpr> rules,
    std::string                    start,
    PackratOptions                 options)
    : rules_(std::move(rules)), start_(std::move(start)), options_(options) {
  if (start_.empty() || !rules_.contains(start_))
    throw std::invalid_argument("PEG: defined nonempty start rule required");
  std::vector<const PegExpr*> pending;
  for (auto& [name, node] : rules_) {
    if (name.empty())
      throw std::invalid_argument("PEG: nonempty rule names required");
    pending.push_back(&node);
  }
  std::size_t nodes = 0;
  while (!pending.empty()) {
    auto* node = pending.back();
    pending.pop_back();
    increment(nodes, options_.max_grammar_nodes);
    switch (node->op) {
      case PegOp::Literal:
        if (!node->children.empty())
          throw std::invalid_argument("PEG: literal has no children");
        break;
      case PegOp::Ref:
        if (!node->children.empty() || node->text.empty() || !rules_.contains(node->text))
          throw std::invalid_argument("PEG: reference must name a defined rule without children");
        break;
      case PegOp::Seq:
      case PegOp::Choice:
        if (!node->text.empty())
          throw std::invalid_argument("PEG: structural sequence/choice has no text");
        break;
      case PegOp::Star:
      case PegOp::Plus:
      case PegOp::Opt:
      case PegOp::And:
      case PegOp::Not:
        if (!node->text.empty() || node->children.size() != 1)
          throw std::invalid_argument("PEG: unary operator needs exactly one child and no text");
        break;
      default:
        throw std::invalid_argument("PEG: unknown operator");
    }
    for (auto& child : node->children)
      pending.push_back(&child);
  }
}
void PackratParser::emit(PackratTrace* history, PackratEvent event) {
  if (!history)
    return;
  auto next = add(history->captured_entries, add(1, event.rule.size()));
  if (options_.max_trace_entries && next > options_.max_trace_entries)
    throw std::length_error("PEG: actual event history quota");
  history->captured_entries = next;
  history->events.push_back(std::move(event));
}
PackratParser::Match
PackratParser::parse_rule(const std::string& name, std::size_t position, PackratTrace* history) {
  Frame frame(depth_, last_.max_depth, options_.max_recursion_depth);
  auto  key = std::make_pair(name, position);
  auto  it  = memo_.find(key);
  if (it != memo_.end()) {
    if (it->second.active)
      throw std::invalid_argument("PEG: active-key left recursion unsupported");
    increment(last_.memo_hits, 0, true);
    auto result = it->second.result;
    if (history)
      emit(
          history,
          {PackratEventKind::memo_hit,
           name,
           PegOp::Ref,
           position,
           result.matched,
           result.end,
           0,
           0,
           depth_});
    return result;
  }
  increment(last_.rule_evaluations, options_.max_rule_evaluations, true);
  memo_[key] = {true, {}};
  if (history)
    emit(
        history,
        {PackratEventKind::rule_enter, name, PegOp::Ref, position, false, 0, 0, 0, depth_});
  auto result   = parse_expr(rules_.at(name), position, name, history);
  memo_.at(key) = {false, result};
  if (history)
    emit(
        history,
        {PackratEventKind::rule_exit,
         name,
         PegOp::Ref,
         position,
         result.matched,
         result.end,
         0,
         0,
         depth_});
  return result;
}
PackratParser::Match PackratParser::parse_expr(
    const PegExpr&     e,
    std::size_t        position,
    const std::string& owner,
    PackratTrace*      history) {
  Frame frame(depth_, last_.max_depth, options_.max_recursion_depth);
  increment(last_.expression_evaluations, options_.max_expression_evaluations);
  auto        expression = last_.expression_evaluations;
  Match       result;
  std::size_t alternative = 0;
  switch (e.op) {
    case PegOp::Literal:
      if (position <= input_.size() && e.text.size() <= input_.size() - position &&
          input_.compare(position, e.text.size(), e.text) == 0)
        result = {true, position + e.text.size()};
      break;
    case PegOp::Ref:
      result = parse_rule(e.text, position, history);
      break;
    case PegOp::Seq: {
      auto p = position;
      result = {true, p};
      for (auto& child : e.children) {
        result = parse_expr(child, p, owner, history);
        if (!result.matched)
          break;
        p = result.end;
      }
      break;
    }
    case PegOp::Choice: {
      for (std::size_t i = 0; i < e.children.size(); ++i) {
        alternative = i;
        if (history)
          emit(
              history,
              {PackratEventKind::choice_try,
               owner,
               PegOp::Choice,
               position,
               false,
               0,
               i,
               expression,
               depth_});
        auto candidate = parse_expr(e.children[i], position, owner, history);
        if (candidate.matched) {
          result = candidate;
          if (history)
            emit(
                history,
                {PackratEventKind::choice_accept,
                 owner,
                 PegOp::Choice,
                 position,
                 true,
                 result.end,
                 i,
                 expression,
                 depth_});
          break;
        }
      }
      break;
    }
    case PegOp::Star: {
      auto p = position;
      for (;;) {
        auto q = parse_expr(e.children[0], p, owner, history);
        if (!q.matched || q.end == p)
          break;
        p = q.end;
      }
      result = {true, p};
      break;
    }
    case PegOp::Plus: {
      auto first = parse_expr(e.children[0], position, owner, history);
      if (!first.matched)
        break;
      auto p = first.end;
      for (;;) {
        auto q = parse_expr(e.children[0], p, owner, history);
        if (!q.matched || q.end == p)
          break;
        p = q.end;
      }
      result = {true, p};
      break;
    }
    case PegOp::Opt: {
      auto child = parse_expr(e.children[0], position, owner, history);
      result     = child.matched ? child : Match{true, position};
      break;
    }
    case PegOp::And: {
      auto child = parse_expr(e.children[0], position, owner, history);
      if (child.matched)
        result = {true, position};
      break;
    }
    case PegOp::Not: {
      auto child = parse_expr(e.children[0], position, owner, history);
      if (!child.matched)
        result = {true, position};
      break;
    }
  }
  if (history)
    emit(
        history,
        {PackratEventKind::expression_result,
         owner,
         e.op,
         position,
         result.matched,
         result.end,
         alternative,
         expression,
         depth_});
  return result;
}
PackratParser::Match PackratParser::execute(const std::string& input, PackratTrace* history) {
  memo_.clear();
  input_.clear();
  last_  = {};
  depth_ = 0;
  try {
    if (options_.max_input_bytes && input.size() > options_.max_input_bytes)
      throw std::length_error("PEG: complete input-byte quota");
    if (history) {
      auto count = add(input.size(), 3);
      if (options_.max_trace_entries && count > options_.max_trace_entries)
        throw std::length_error("PEG: complete input/decision capture quota");
      history->captured_entries = count;
      history->input            = input;
    }
    input_               = input;
    auto result          = parse_rule(start_, 0, history);
    last_.matched_prefix = result.matched;
    last_.end            = result.end;
    last_.accepted       = result.matched && result.end == input_.size();
    if (history) {
      for (auto& [key, value] : memo_) {
        auto count = add(history->captured_entries, add(1, key.first.size()));
        if (options_.max_trace_entries && count > options_.max_trace_entries)
          throw std::length_error("PEG: final memo history quota");
        history->captured_entries = count;
        history->memo.push_back({key.first, key.second, value.result.matched, value.result.end});
      }
    }
    last_.completed = true;
    return result;
  } catch (...) {
    memo_.clear();
    last_.completed = false;
    throw;
  }
}
bool PackratParser::parse(const std::string& input) {
  (void)execute(input, nullptr);
  return last_.accepted;
}
PackratTrace PackratParser::parse_trace(const std::string& input) {
  PackratTrace t;
  (void)execute(input, &t);
  t.run = last_;
  return t;
}
long PackratParser::rule_calls() const noexcept {
  return static_cast<long>(last_.rule_evaluations);
}
long PackratParser::memo_hits() const noexcept {
  return static_cast<long>(last_.memo_hits);
}
PackratRun PackratParser::last_run() const noexcept {
  return last_;
}
std::map<std::string, PegExpr> packrat_anbn_rules() {
  return {
      {"S",
       peg_choice(
           {peg_seq({peg_lit("a"), peg_ref("S"), peg_lit("b")}),
            peg_seq({peg_lit("a"), peg_lit("b")})})}};
}
std::map<std::string, PegExpr> packrat_negative_lookahead_rules() {
  return {{"Start", peg_seq({peg_plus(peg_lit("a")), peg_not(peg_lit("c"))})}};
}
std::map<std::string, PegExpr> packrat_reuse_rules() {
  return {
      {"S",
       peg_choice({peg_seq({peg_ref("A"), peg_lit("b")}), peg_seq({peg_ref("A"), peg_lit("c")})})},
      {"A", peg_plus(peg_lit("a"))}};
}
std::map<std::string, PegExpr> packrat_quadratic_rules() {
  return {
      {"S",
       peg_seq({peg_star(peg_seq({peg_and(peg_ref("A")), peg_lit("a")})), peg_not(peg_lit("a"))})},
      {"A", peg_star(peg_lit("a"))}};
}
std::string packrat_balanced_input(std::size_t n, std::size_t max_bytes) {
  if (n > std::string().max_size() / 2 || (max_bytes && n > max_bytes / 2))
    throw std::length_error("PEG fixture: complete2n byte quota");
  return std::string(n, 'a') + std::string(n, 'b');
}
} // namespace datamunge::algorithms
