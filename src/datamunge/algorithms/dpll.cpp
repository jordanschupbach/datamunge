/// @file dpll.cpp
/// @brief Validated raw-CNF deterministic DPLL with explicit heap frames and actual event capture.
#include <datamunge/algorithms/dpll.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private checked counters, raw-clause scans and iterative search frames.
namespace {
/// @brief Dense signed partial assignment, index0 unused.
using Assign=std::vector<int>;
/// @brief Checked addition for dense sizes, event entries and logical work.
/// @param a Current count.
/// @param b Increment.
/// @return Exact sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("DPLL count overflow");
  return a+b;
}
/// @brief Increment one logical counter with overflow checking.
/// @param counter Mutable actual counter.
/// @throws std::length_error On size_t overflow.
void increment(std::size_t& counter) {counter=add(counter,1);}
/// @brief Check a nonzero optional entry limit.
/// @param entries Requested count.
/// @param limit Zero disables only optional cap.
/// @throws std::length_error If a nonzero limit is exceeded.
void cap(std::size_t entries,std::size_t limit) {
  if(limit && entries>limit)throw std::length_error("DPLL capture/generator cap exceeded");
}
/// @brief Obtain a validated literal's variable without abs(INT_MIN).
/// @param literal Nonzero validated signed id, never INT_MIN.
/// @return Positive variable id.
std::size_t variable(int literal) {return static_cast<std::size_t>(literal<0?-literal:literal);}
/// @brief Validate all literals before dense allocation or any SAT verdict.
/// @param n Nonnegative dense variable count.
/// @param clauses Raw original CNF.
/// @return Raw input capture cost1+clause count+literal count.
/// @throws std::invalid_argument For a negative bound or any invalid literal, including zero/INT_MIN.
/// @throws std::length_error On entry-count overflow.
std::size_t validate(int n,const std::vector<std::vector<int>>& clauses) {
  if(n<0)throw std::invalid_argument("DPLL variable count must be nonnegative");
  std::size_t entries=add(1,clauses.size());
  for(const auto& c:clauses)for(int l:c) {
    if(l==0 || l==std::numeric_limits<int>::min() || l>n || l< -n)
      throw std::invalid_argument("DPLL literal outside variable domain");
    entries=add(entries,1);
  }
  return entries;
}
/// @brief Test a validated literal under a partial assignment.
/// @param literal Signed literal.
/// @param a Partial dense assignment.
/// @return True when assigned to the literal's satisfying polarity.
bool true_literal(int literal,const Assign& a) {return a[variable(literal)]==(literal>0?1:-1);}
/// @brief Inspect the full formula with actual early-exit work and first conflict identity.
/// @param clauses Original raw clauses.
/// @param a Partial dense assignment.
/// @param w Accumulated work, mutated on each inspected occurrence.
/// @param reason Receives first conflicting original clause, or clause-count sentinel.
/// @return One if all clauses true, minus one for a false clause, zero otherwise.
/// @throws std::length_error On work-counter overflow.
int status(const std::vector<std::vector<int>>& clauses,const Assign& a,DpllWork& w,std::size_t& reason) {
  bool all=true;reason=clauses.size();
  for(std::size_t i=0;i<clauses.size();++i) {
    increment(w.clause_visits);bool satisfied=false,unassigned=false;
    for(int l:clauses[i]) {
      increment(w.literal_visits);const auto v=variable(l);
      if(!a[v])unassigned=true;
      else if(true_literal(l,a)){satisfied=true;break;}
    }
    if(satisfied)continue;
    if(!unassigned){reason=i;return -1;}
    all=false;
  }
  return all?1:0;
}
/// @brief Find the first unsatisfied raw clause with one unassigned occurrence.
/// @param clauses Original CNF, preserving multiplicity.
/// @param a Partial dense assignment.
/// @param w Actual scan counters.
/// @param reason Receives original unit clause index or clause-count sentinel.
/// @return Forced signed literal, or zero if none recognized.
/// @throws std::length_error On work-counter overflow.
int find_unit(const std::vector<std::vector<int>>& clauses,const Assign& a,DpllWork& w,std::size_t& reason) {
  reason=clauses.size();
  for(std::size_t i=0;i<clauses.size();++i) {
    increment(w.clause_visits);int only=0,count=0;bool satisfied=false;
    for(int l:clauses[i]) {
      increment(w.literal_visits);
      if(true_literal(l,a)){satisfied=true;break;}
      if(!a[variable(l)]){only=l;count=std::min(2,count+1);}
    }
    if(!satisfied && count==1){reason=i;return only;}
  }
  return 0;
}
/// @brief Find the smallest unassigned variable occurring in only one unsatisfied polarity.
/// @param clauses Original raw CNF.
/// @param a Partial dense assignment.
/// @param w Actual work counters, including both passes over each unsatisfied clause.
/// @return Signed pure literal, or zero if none.
/// @throws std::length_error On work-counter overflow; allocation can also fail.
int find_pure(const std::vector<std::vector<int>>& clauses,const Assign& a,DpllWork& w) {
  std::vector<unsigned char> polarity(a.size(),0);
  for(const auto& c:clauses) {
    increment(w.clause_visits);bool satisfied=false;
    for(int l:c){increment(w.literal_visits);if(true_literal(l,a)){satisfied=true;break;}}
    if(satisfied)continue;
    for(int l:c){increment(w.literal_visits);const auto v=variable(l);if(!a[v])polarity[v]|=l>0?1:2;}
  }
  for(std::size_t v=1;v<a.size();++v)if(!a[v]) {
    if(polarity[v]==1)return static_cast<int>(v);
    if(polarity[v]==2)return -static_cast<int>(v);
  }
  return 0;
}
/// @brief One owned suspended decision node; child mutations cannot alter its restored state.
struct Frame {
  Assign assignment; ///< Current propagated partial state, held fixed while children run.
  std::size_t node=0; ///< Unique entry-order id.
  std::size_t parent=std::numeric_limits<std::size_t>::max(); ///< Parent or root sentinel.
  std::size_t depth=0; ///< Decision depth.
  int incoming=0; ///< Signed branch literal that entered this node, zero at root.
  int split=0; ///< Selected branch variable, zero before branching.
  bool false_tried=false; ///< Whether the negative child has already been entered.
};
/// @brief Capture one actual partial state after checking complete event cost.
/// @param t Result/capture accumulator.
/// @param history Whether capture is enabled.
/// @param limit Optional logical entry cap.
/// @param f Current actual frame.
/// @param kind Event kind.
/// @param literal Assigned/entered/failed-child literal, zero otherwise.
/// @param clause Original unit/conflict index or clause-count sentinel.
/// @throws std::length_error Before capture if count/cap exceeds limits; allocation can fail.
void emit(DpllTrace& t,bool history,std::size_t limit,const Frame& f,DpllEventKind kind,int literal,std::size_t clause) {
  if(!history)return;
  const auto next=add(t.captured_entries,add(1,f.assignment.size()));cap(next,limit);
  t.events.push_back({kind,f.node,f.parent,f.depth,literal,clause,f.assignment});t.captured_entries=next;
}
/// @brief Enter one true/false child from the parent's fixed state without native recursion.
/// @param stack Live explicit search frames.
/// @param literal Signed child decision.
/// @param t Result/work/capture accumulator.
/// @param history Enable actual snapshots.
/// @param limit Optional entry cap.
/// @param clause_count Original clause-count sentinel.
/// @throws std::length_error On counter/entry overflow; allocation can fail.
void child(std::vector<Frame>& stack,int literal,DpllTrace& t,bool history,std::size_t limit,std::size_t clause_count) {
  const auto& p=stack.back();Frame c;c.assignment=p.assignment;c.assignment[variable(literal)]=literal>0?1:-1;
  c.node=t.solve.work.nodes;c.parent=p.node;c.depth=add(p.depth,1);c.incoming=literal;
  increment(t.solve.work.nodes);increment(t.solve.work.branches);
  t.solve.work.max_depth=std::max(t.solve.work.max_depth,c.depth);
  emit(t,history,limit,c,DpllEventKind::enter,literal,clause_count);stack.push_back(std::move(c));
}
/// @brief Return from failed nodes, restoring parents and trying an unvisited false child.
/// @param stack Nonempty live stack whose top failed.
/// @param t Actual result/work/capture accumulator.
/// @param history Enable snapshots.
/// @param limit Optional cap.
/// @param clause_count Original clause-count sentinel.
/// @return True after entering a false child; false when root failure completes UNSAT.
/// @throws std::length_error On count/cap overflow; allocation can fail.
bool failed(std::vector<Frame>& stack,DpllTrace& t,bool history,std::size_t limit,std::size_t clause_count) {
  for(;;) {
    const int incoming=stack.back().incoming;stack.pop_back();if(stack.empty())return false;
    auto& p=stack.back();increment(t.solve.work.backtracks);
    emit(t,history,limit,p,DpllEventKind::backtrack,incoming,clause_count);
    if(!p.false_tried){p.false_tried=true;const int literal=-p.split;child(stack,literal,t,history,limit,clause_count);return true;}
    emit(t,history,limit,p,DpllEventKind::exhausted,0,clause_count);
  }
}
/// @brief Shared deterministic iterative DPLL, optionally retaining original input and events.
/// @param n Variable count.
/// @param clauses Borrowed original CNF.
/// @param history Enable actual owned capture.
/// @param limit Optional entry cap, zero disables.
/// @return Result/work and, when enabled, original input and all events.
/// @throws std::invalid_argument For complete input-domain failures.
/// @throws std::length_error For counter/cap/container overflow; allocation can fail.
DpllTrace run(int n,const std::vector<std::vector<int>>& clauses,bool history,std::size_t limit) {
  const auto input_entries=validate(n,clauses);const auto size=add(static_cast<std::size_t>(n),1);
  if(size>Assign{}.max_size())throw std::length_error("DPLL dense assignment capacity exceeded");
  DpllTrace t;t.num_vars=n;
  if(history){cap(add(input_entries,add(size,1)),limit);t.clauses=clauses;t.captured_entries=input_entries;}
  Frame root;root.assignment.resize(size,0);std::vector<Frame> stack;stack.push_back(std::move(root));
  t.solve.work.nodes=1;emit(t,history,limit,stack.back(),DpllEventKind::enter,0,clauses.size());
  for(;;) {
    auto& f=stack.back();std::size_t reason=clauses.size();const int st=status(clauses,f.assignment,t.solve.work,reason);
    if(st==1) {
      emit(t,history,limit,f,DpllEventKind::satisfied,0,clauses.size());
      t.solve.result.satisfiable=true;t.solve.result.assignment=f.assignment;
      for(std::size_t v=1;v<size;++v)if(!t.solve.result.assignment[v])t.solve.result.assignment[v]=1;
      return t;
    }
    if(st==-1) {
      increment(t.solve.work.conflicts);emit(t,history,limit,f,DpllEventKind::conflict,0,reason);
      if(!failed(stack,t,history,limit,clauses.size()))return t;
      continue;
    }
    if(const int u=find_unit(clauses,f.assignment,t.solve.work,reason)) {
      f.assignment[variable(u)]=u>0?1:-1;increment(t.solve.work.unit_steps);
      emit(t,history,limit,f,DpllEventKind::unit,u,reason);continue;
    }
    if(const int p=find_pure(clauses,f.assignment,t.solve.work)) {
      f.assignment[variable(p)]=p>0?1:-1;increment(t.solve.work.pure_steps);
      emit(t,history,limit,f,DpllEventKind::pure,p,clauses.size());continue;
    }
    std::size_t v=1;while(v<size && f.assignment[v])++v;
    f.split=static_cast<int>(v);increment(t.solve.work.decisions);
    child(stack,f.split,t,history,limit,clauses.size());
  }
}
} // namespace
SatResult dpll_sat(int n,const std::vector<std::vector<int>>& clauses) {return run(n,clauses,false,0).solve.result;}
DpllMeasurement dpll_sat_work(int n,const std::vector<std::vector<int>>& clauses) {return run(n,clauses,false,0).solve;}
DpllTrace dpll_sat_trace(int n,const std::vector<std::vector<int>>& clauses,std::size_t limit) {return run(n,clauses,true,limit);}
bool cnf_assignment_satisfies(int n,const std::vector<std::vector<int>>& clauses,const std::vector<int>& a) {
  validate(n,clauses);
  if(a.size()!=static_cast<std::size_t>(n)+1)throw std::invalid_argument("CNF total model size mismatch");
  for(std::size_t v=1;v<a.size();++v)if(a[v]!=1 && a[v]!=-1)throw std::invalid_argument("CNF model must be total +/-1");
  for(const auto& c:clauses) {bool sat=false;for(int l:c)if(true_literal(l,a)){sat=true;break;}if(!sat)return false;}
  return true;
}
std::vector<std::vector<int>> random_3sat_formula(int n,std::size_t m,std::mt19937& engine,std::size_t max_literals) {
  if(n<3)throw std::invalid_argument("random3-SAT requires at least three variables");
  if(m>std::numeric_limits<std::size_t>::max()/3 || m>std::vector<std::vector<int>>{}.max_size())throw std::length_error("random3-SAT literal count overflow");
  cap(3*m,max_literals);std::vector<std::vector<int>> out;out.reserve(m);
  for(std::size_t i=0;i<m;++i) {
    std::vector<int> c;c.reserve(3);
    while(c.size()<3) {
      const int v=1+static_cast<int>(engine()%static_cast<unsigned>(n));bool duplicate=false;
      for(int l:c)if(variable(l)==static_cast<std::size_t>(v)){duplicate=true;break;}
      if(!duplicate)c.push_back(engine()&1?v:-v);
    }
    out.push_back(std::move(c));
  }
  return out;
}
} // namespace datamunge::algorithms
