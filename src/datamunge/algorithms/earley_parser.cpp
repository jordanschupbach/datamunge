/// @file earley_parser.cpp
/// @brief Indexed Earley closure with both arrival orders for nullable completion and actual first causes.
#include <datamunge/algorithms/earley_parser.hpp>
#include <format>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
namespace datamunge::algorithms {
/// @brief Checked compilation, sparse chart indexes and insertion-ordered agendas.
namespace {
/// @brief Add a logical count without wrapping size_t.
/// @param a Existing count.
/// @param b Nonnegative increment.
/// @return Checked sum.
/// @throws std::length_error For overflow.
std::size_t sum_size(std::size_t a,std::size_t b) {
 if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("Earley capture arithmetic overflow");return a+b;
}
/// @brief Increment a work count before its associated operation.
/// @param value Mutable counter.
/// @throws std::length_error Before counter wrap.
void count(std::uint64_t& value) {
 if(value==std::numeric_limits<std::uint64_t>::max())throw std::length_error("Earley work counter overflow");++value;
}
/// @brief Intern an opaque symbol in first-appearance order.
/// @param names Mutable private symbol map.
/// @param name Borrowed raw string, including empty/control-byte names.
/// @return Stable numeric id.
/// @throws std::bad_alloc For map/string storage.
std::size_t intern(std::unordered_map<std::string,std::size_t>& names,const std::string& name) {
 const auto found=names.find(name);if(found!=names.end())return found->second;
 const auto id=names.size();names.emplace(name,id);return id;
}
/// @brief Compiled original rule with integer symbol ids.
struct Rule {
 std::size_t lhs; ///< Nonterminal id from the LHS-only first pass.
 std::vector<std::size_t> rhs; ///< Original RHS symbol order; empty is epsilon.
};
/// @brief Owned integer grammar/token view used only during one recognition.
struct Prepared {
 std::size_t nonterminals=0; ///< LHS-only id prefix length.
 std::size_t start=earley_no_item; ///< Start nonterminal id, or absent for no start production.
 std::vector<Rule> rules; ///< Original production order, including duplicates.
 std::vector<std::vector<int>> groups; ///< Original production ids for each LHS.
 std::vector<std::size_t> tokens; ///< Interned atomic input names.

 /// @brief Compile already shape-checked strings while preserving every original rule id.
 /// @param grammar Borrowed raw grammar.
 /// @param input Borrowed tokens.
 /// @throws std::bad_alloc For private compilation storage.
 /// @throws std::length_error For container bounds.
 Prepared(const ContextFreeGrammar& grammar,const std::vector<std::string>& input) {
  std::unordered_map<std::string,std::size_t> names;
  for(const auto& p:grammar.productions)(void)intern(names,p.lhs);
  nonterminals=names.size();groups.resize(nonterminals);
  const auto selected=names.find(grammar.start);if(selected!=names.end())start=selected->second;
  for(const auto& p:grammar.productions){Rule r{names.at(p.lhs),{}};
   for(const auto& symbol:p.rhs)r.rhs.push_back(intern(names,symbol));
   groups[r.lhs].push_back(static_cast<int>(rules.size()));rules.push_back(std::move(r));}
  for(const auto& token:input)tokens.push_back(intern(names,token));
 }
};
/// @brief Hash valid scalar item coordinates; collisions are resolved by equality.
struct ItemHash {
 /// @brief Combine three int hashes without signed arithmetic overflow.
 /// @param item Valid private item value.
 /// @return Unsigned hash, with no ordering meaning.
 std::size_t operator()(const EarleyItem& item) const noexcept {
  std::size_t h=std::hash<int>{}(item.prod);
  for(int v:{item.dot,item.origin})h^=std::hash<int>{}(v)+std::size_t(0x9e3779b9U)+(h<<6)+(h>>2);
  return h;
 }
};
/// @brief One lazily allocated active column with stable numeric references.
struct Column {
 std::vector<EarleyItem> items; ///< Unique insertion order; copied before processing.
 std::unordered_set<EarleyItem,ItemHash> seen; ///< Membership only; never drives iteration order.
 std::unordered_map<std::size_t,std::vector<std::size_t>> waiting; ///< Processed waiter ordinals by expected nonterminal.
 std::unordered_map<std::size_t,std::unordered_map<int,std::size_t>> complete; ///< First processed completed ordinal by LHS/origin.
 std::unordered_set<std::size_t> predicted; ///< Nonterminals whose productions were already attempted here.
};
/// @brief One private iterative recognition with optional local owned capture.
class Run {
 const Prepared& prepared_; ///< Borrowed immutable compiled symbols, lasting through finish().
 std::vector<std::unique_ptr<Column>> columns_; ///< Sparse active columns; outer size never changes.
 EarleyTrace* trace_; ///< Local optional capture, never exposed until successful return.
 std::size_t limit_; ///< Optional complete capture cap; zero disables only quota.
 EarleyWork work_; ///< Checked actual logical counts.

 /// @brief Insert a unique item and its first reason, checking quota before writes.
 /// @param k Target boundary, at most token count.
 /// @param item Valid original production/dot/origin coordinates, copied by value.
 /// @param reason First attempted deduction, with earlier numeric parent locations.
 /// @throws std::length_error For item/counter/capture bounds.
 /// @throws std::bad_alloc For item/index/reason/active-column storage.
 void add(std::size_t k,EarleyItem item,EarleyReason reason) {
  count(work_.insert_attempts);auto& pointer=columns_[k];
  if(pointer&&pointer->seen.find(item)!=pointer->seen.end()){count(work_.duplicate_items);return;}
  if(pointer&&pointer->items.size()>=static_cast<std::size_t>(std::numeric_limits<int>::max()))throw std::length_error("Earley column exceeds int item count");
  if(trace_){const auto next=sum_size(trace_->captured_entries,2);if(limit_&&next>limit_)throw std::length_error("Earley unique-item capture cap exceeded");trace_->captured_entries=next;}
  if(!pointer)pointer=std::make_unique<Column>();
  pointer->seen.insert(item);pointer->items.push_back(item);
  if(trace_)trace_->reasons[k].push_back(reason);count(work_.unique_items);
 }
 /// @brief Join one existing waiter and one first completed span exactly once.
 /// @param k Target/end column.
 /// @param waiter Copied waiter coordinates, before the completed nonterminal.
 /// @param from Stable waiter location.
 /// @param completed Stable completed-item location.
 /// @throws std::length_error For work/item/capture limits.
 /// @throws std::bad_alloc For a new target item.
 void join(std::size_t k,EarleyItem waiter,EarleyLocation from,EarleyLocation completed) {
  count(work_.completion_joins);add(k,{waiter.prod,waiter.dot+1,waiter.origin},{EarleyAction::complete,from,completed});
 }
public:
 /// @brief Allocate only the outer column references, never all runtime indexes eagerly.
 /// @param prepared Borrowed compiled grammar/tokens.
 /// @param trace Optional local trace with base count and column metadata allocated.
 /// @param limit Optional capture quota.
 /// @throws std::bad_alloc For outer reference storage.
 /// @throws std::length_error For container bounds.
 Run(const Prepared& prepared,EarleyTrace* trace,std::size_t limit)
  :prepared_(prepared),columns_(prepared.tokens.size()+1),trace_(trace),limit_(limit) {}
 /// @brief Close every column in order, then prepare summary and move captured item buffers.
 /// @return Owned verdict, N+1 int sizes and logical work.
 /// @throws std::length_error For item/work/capture/container bounds.
 /// @throws std::bad_alloc For runtime and output storage.
 EarleyResult finish() {
  const auto& p=prepared_;
  if(p.start!=earley_no_item)for(int id:p.groups[p.start])add(0,{id,0,0},{});
  for(std::size_t k=0;k<columns_.size();++k){if(!columns_[k])continue;auto& column=*columns_[k];
   for(std::size_t index=0;index<column.items.size();++index){
    const EarleyItem item=column.items[index];const EarleyLocation here{k,index};
    const auto& rule=p.rules[item.prod];count(work_.items_processed);
    if(static_cast<std::size_t>(item.dot)<rule.rhs.size()){
     const auto symbol=rule.rhs[item.dot];
     if(symbol<p.nonterminals){
      column.waiting[symbol].push_back(index);
      if(column.predicted.insert(symbol).second){count(work_.prediction_batches);
       for(int id:p.groups[symbol]){count(work_.predicted_productions);add(k,{id,0,static_cast<int>(k)},{EarleyAction::predict,here,{}});}}
      const auto published=column.complete.find(symbol);
      if(published!=column.complete.end()){
       const auto empty=published->second.find(static_cast<int>(k));
       if(empty!=published->second.end()){const auto completed=empty->second;count(work_.nullable_joins);join(k,item,here,{k,completed});}}
     }else{count(work_.scan_checks);if(k<p.tokens.size()&&p.tokens[k]==symbol){count(work_.scan_matches);add(k+1,{item.prod,item.dot+1,item.origin},{EarleyAction::scan,here,{}});}}
    }else{
     auto& origins=column.complete[rule.lhs];
     if(!origins.emplace(item.origin,index).second)continue;
     count(work_.completed_spans);auto& origin=*columns_[item.origin];
     const auto waiting=origin.waiting.find(rule.lhs);
     if(waiting!=origin.waiting.end())for(std::size_t wi=0;wi<waiting->second.size();++wi){
      const auto ordinal=waiting->second[wi];const EarleyItem waiter=origin.items[ordinal];
      join(k,waiter,{static_cast<std::size_t>(item.origin),ordinal},here);}
    }
   }
  }
  EarleyResult result;result.chart_sizes.reserve(columns_.size());
  for(std::size_t k=0;k<columns_.size();++k){auto& c=columns_[k];result.chart_sizes.push_back(c?static_cast<int>(c->items.size()):0);
   if(k==p.tokens.size()&&c)for(const auto& item:c->items)
    if(p.rules[item.prod].lhs==p.start&&item.origin==0&&static_cast<std::size_t>(item.dot)==p.rules[item.prod].rhs.size())result.accepted=true;
   if(trace_&&c)trace_->chart[k]=std::move(c->items);
  }
  result.work=work_;return result;
 }
};
} // namespace
std::size_t earley_trace_base_entries(const ContextFreeGrammar& grammar,std::size_t n) {
 const auto bound=static_cast<std::size_t>(std::numeric_limits<int>::max());
 if(n>bound||grammar.productions.size()>bound)throw std::length_error("Earley input or production indices exceed int");
 std::size_t rhs=0;for(const auto& p:grammar.productions){if(p.rhs.size()>bound)throw std::length_error("Earley dot exceeds int");rhs=sum_size(rhs,p.rhs.size());}
 return sum_size(sum_size(sum_size(sum_size(n,n),grammar.productions.size()),rhs),2);
}
EarleyResult earley_recognize_work(const ContextFreeGrammar& grammar,const std::vector<std::string>& tokens) {
 (void)earley_trace_base_entries(grammar,tokens.size());const Prepared p(grammar,tokens);return Run(p,nullptr,0).finish();
}
EarleyTrace earley_trace(const ContextFreeGrammar& grammar,const std::vector<std::string>& tokens,std::size_t max_entries) {
 const auto base=earley_trace_base_entries(grammar,tokens.size());if(max_entries&&base>max_entries)throw std::length_error("Earley base capture cap exceeded");
 EarleyTrace t;t.grammar=grammar;t.tokens=tokens;t.captured_entries=base;t.chart.resize(tokens.size()+1);t.reasons.resize(tokens.size()+1);
 const Prepared p(t.grammar,t.tokens);auto result=Run(p,&t,max_entries).finish();t.accepted=result.accepted;t.work=result.work;return t;
}
std::string earley_item_text(const EarleyTrace& t,EarleyLocation where) {
 if(where.column>=t.chart.size()||where.column>t.tokens.size()||where.index>=t.chart[where.column].size())throw std::invalid_argument("Earley item location outside chart");
 const auto item=t.chart[where.column][where.index];
 if(item.prod<0||static_cast<std::size_t>(item.prod)>=t.grammar.productions.size()||item.dot<0||item.origin<0||static_cast<std::size_t>(item.origin)>where.column)throw std::invalid_argument("Earley item coordinates invalid");
 const auto& rule=t.grammar.productions[item.prod];if(static_cast<std::size_t>(item.dot)>rule.rhs.size())throw std::invalid_argument("Earley dot outside production");
 auto text=std::format("p{} {:?} ->",item.prod,rule.lhs);
 for(std::size_t i=0;i<=rule.rhs.size();++i){if(i==static_cast<std::size_t>(item.dot))text+=" .";if(i<rule.rhs.size())text+=std::format(" {:?}",rule.rhs[i]);}
 text+=std::format(" ({})",item.origin);return text;
}
} // namespace datamunge::algorithms
