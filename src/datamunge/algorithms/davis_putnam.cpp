/// @file
/// @brief Canonical checked finite-CNF Davis-Putnam kernel and owned actual snapshots.
#include <datamunge/algorithms/davis_putnam.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Private sparse ordered-clause elimination and checked snapshot helpers.
namespace {
/// @brief Sorted distinct signed literals, never containing zero or INT_MIN.
using Clause=std::set<int>;
/// @brief Lexicographically sorted distinct clauses; no subsumption is implied.
using Formula=std::set<Clause>;
/// @brief Add size counts or throw before wraparound.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Exact sum.
/// @throws std::length_error Sum exceeds size_t.
std::size_t add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a) throw std::length_error("Davis-Putnam count overflow");
  return a+b;
}
/// @brief Enforce one optional zero-disabled count ceiling.
/// @param value Checked actual or prospective count.
/// @param limit Zero to disable, otherwise inclusive ceiling.
/// @throws std::length_error value exceeds nonzero limit.
void cap(std::size_t value,std::size_t limit) {
  if(limit && value>limit) throw std::length_error("Davis-Putnam resource limit");
}
/// @brief Test whether a validated clause contains both signs of some variable.
/// @param c Ordered literals excluding INT_MIN, so negation is safe.
/// @return True if the clause is always satisfied.
bool tautology(const Clause& c) {
  for(int l:c) if(c.count(-l)) return true;
  return false;
}
/// @brief Copy a complete ordered formula after charging all retained entries.
/// @param f Current canonical formula.
/// @param used Cumulative snapshot entries, updated before allocation.
/// @param limit Optional inclusive entry cap, zero disables.
/// @return Owned vector representation preserving canonical order.
/// @throws std::length_error Arithmetic, cap or container capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
std::vector<std::vector<int>> snapshot(const Formula& f,std::size_t& used,std::size_t limit) {
  auto cost=add(1,f.size());for(const auto& c:f) cost=add(cost,c.size());
  used=add(used,cost);cap(used,limit);
  std::vector<std::vector<int>> out;out.reserve(f.size());
  for(const auto& c:f) out.emplace_back(c.begin(),c.end());
  return out;
}
/// @brief Execute all rules once, optionally retaining actual transitions.
/// @param n Nonnegative maximum variable id, checked before all verdicts.
/// @param input Borrowed complete CNF, unchanged.
/// @param trace Optional output history; null avoids all snapshots.
/// @param clause_limit Optional maximum stored normalized clause-set size.
/// @param pair_limit Optional cumulative resolution-pair ceiling.
/// @param entry_limit Optional cumulative snapshot-entry ceiling.
/// @return Exact SAT verdict on successful completion.
/// @throws std::invalid_argument Invalid variable bound or literal.
/// @throws std::length_error Arithmetic, optional limit or capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
bool solve(int n,const std::vector<std::vector<int>>& input,DpTrace* trace,
           std::size_t clause_limit,std::size_t pair_limit,std::size_t entry_limit) {
  if(n<0) throw std::invalid_argument("Davis-Putnam negative variable bound");
  for(const auto& c:input) for(int l:c)
    if(l==0 || l==INT_MIN || l>n || l< -n) throw std::invalid_argument("Davis-Putnam literal outside domain");
  Formula f;
  for(const auto& c:input) {Clause s(c.begin(),c.end());if(!tautology(s)) {f.insert(std::move(s));cap(f.size(),clause_limit);}}
  DpWork work;work.peak_clauses=f.size();std::size_t entries=0;
  if(trace) {trace->num_vars=n;trace->initial=snapshot(f,entries,entry_limit);}
  for(;;) {
    if(f.empty() || f.begin()->empty()) {
      const bool sat=f.empty();if(trace) {trace->satisfiable=sat;trace->work=work;}return sat;
    }
    DpStep step;step.before_clauses=f.size();int literal=0;
    for(const auto& c:f) if(c.size()==1) {literal=*c.begin();break;}
    if(literal) step.rule=DpRule::unit;
    else {
      Clause lits;for(const auto& c:f) lits.insert(c.begin(),c.end());
      for(int l:lits) if(!lits.count(-l)) {literal=l;break;}
      if(literal) step.rule=DpRule::pure;
      else {step.rule=DpRule::resolution;literal=INT_MAX;for(int l:lits) literal=std::min(literal,l<0?-l:l);}
    }
    step.literal=literal;Formula next;
    if(step.rule!=DpRule::resolution) {
      for(auto c:f) if(!c.count(literal)) {c.erase(-literal);next.insert(std::move(c));cap(next.size(),clause_limit);}
      if(step.rule==DpRule::unit) work.unit_steps=add(work.unit_steps,1);
      else work.pure_steps=add(work.pure_steps,1);
    } else {
      std::vector<const Clause*> pos,neg;
      for(const auto& c:f) {
        if(c.count(literal)) pos.push_back(&c);
        else if(c.count(-literal)) neg.push_back(&c);
        else next.insert(c);
      }
      step.positive_clauses=pos.size();step.negative_clauses=neg.size();
      if(pos.size() && neg.size()>std::numeric_limits<std::size_t>::max()/pos.size()) throw std::length_error("Davis-Putnam pair product overflow");
      step.pairs=pos.size()*neg.size();const auto total=add(work.resolution_pairs,step.pairs);cap(total,pair_limit);
      for(const auto* a:pos) for(const auto* b:neg) {
        Clause r=*a;r.erase(literal);for(int l:*b) if(l!=-literal) r.insert(l);
        if(tautology(r)) step.tautologies=add(step.tautologies,1);
        else {if(!next.insert(std::move(r)).second) step.duplicates=add(step.duplicates,1);cap(next.size(),clause_limit);}
      }
      work.resolution_steps=add(work.resolution_steps,1);work.resolution_pairs=total;
      work.tautological_resolvents=add(work.tautological_resolvents,step.tautologies);
      work.duplicate_resolvents=add(work.duplicate_resolvents,step.duplicates);
    }
    work.peak_clauses=std::max(work.peak_clauses,next.size());
    if(trace) {step.after=snapshot(next,entries,entry_limit);trace->steps.push_back(std::move(step));}
    f=std::move(next);
  }
}
} // namespace
bool davis_putnam(int num_vars,const std::vector<std::vector<int>>& clauses_in) {
  return solve(num_vars,clauses_in,nullptr,0,0,0);
}
DpTrace davis_putnam_trace(int num_vars,const std::vector<std::vector<int>>& clauses_in,
                          std::size_t max_clauses,std::size_t max_pairs,std::size_t max_trace_entries) {
  DpTrace t;solve(num_vars,clauses_in,&t,max_clauses,max_pairs,max_trace_entries);return t;
}
std::vector<std::vector<int>> davis_putnam_grid_formula(std::size_t order,std::size_t max_clauses) {
  if(!order) throw std::invalid_argument("Davis-Putnam grid order must be positive");
  if(order>static_cast<std::size_t>((INT_MAX-1)/2)) throw std::length_error("Davis-Putnam grid variable ids overflow");
  const auto count=add(add(order,order),2);cap(count,max_clauses);(void)add(add(order,order),add(add(order,order),add(order,order)));
  std::vector<std::vector<int>> f;f.reserve(count);std::vector<int> a,b;a.reserve(order);b.reserve(order);
  for(std::size_t i=0;i<order;++i) {
    const int x=2+static_cast<int>(i),y=2+static_cast<int>(order+i);
    f.push_back({1,x});f.push_back({-1,y});a.push_back(-x);b.push_back(-y);
  }
  f.push_back(std::move(a));f.push_back(std::move(b));return f;
}
} // namespace datamunge::algorithms
