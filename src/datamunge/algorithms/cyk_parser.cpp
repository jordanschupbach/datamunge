/// @file
/// @brief Checked compressed-id CYK kernel, complete actual charts and iterative first-witness trees.
#include <datamunge/algorithms/cyk_parser.hpp>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Checked capacities, grammar compression and first-choice parsing internals.
namespace cyk_detail {
/// @brief Add representable size counts.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Exact sum.
/// @throws std::length_error size_t overflow.
std::size_t add(std::size_t a,std::size_t b) {
  if(b>std::numeric_limits<std::size_t>::max()-a) throw std::length_error("CYK: size sum overflow");return a+b;
}
/// @brief Multiply representable size counts.
/// @param a First nonnegative factor.
/// @param b Second nonnegative factor.
/// @return Exact product.
/// @throws std::length_error size_t overflow.
std::size_t mul(std::size_t a,std::size_t b) {
  if(a && b>std::numeric_limits<std::size_t>::max()/a) throw std::length_error("CYK: size product overflow");return a*b;
}
/// @brief Compute the triangular count after cancelling its factor of two.
/// @param n Number of positions.
/// @return n(n+1)/2.
/// @throws std::length_error Count cannot fit size_t.
std::size_t triangle(std::size_t n) {
  if(!n) return 0;auto a=n,b=add(n,1);if(a%2==0) a/=2;else b/=2;return mul(a,b);
}
/// @brief Increment one actual work counter without wrapping.
/// @param count Actual counter to increment.
/// @throws std::overflow_error Count is already SIZE_MAX.
void tick(std::size_t& count) {
  if(count==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("CYK: work count overflow");++count;
}
/// @brief Owned normalized id vocabulary and raw ordered compiled productions.
struct Grammar {
  std::vector<int> ids; ///< Sorted distinct original ids, including the start.
  std::array<std::vector<std::size_t>,256> terminals; ///< Raw terminal candidates indexed by unsigned byte.
  std::vector<std::array<std::size_t,3>> binary; ///< Raw ordered productions mapped to dense ids.
  std::size_t start{0}; ///< Dense start id.
};
/// @brief Compress arbitrary original integer ids without using their magnitude.
/// @param g Public read-only grammar.
/// @return Owned compiled grammar, preserving duplicate/raw production order.
/// @throws std::length_error Grammar container capacity exceeded.
/// @throws std::bad_alloc Grammar allocation failure.
Grammar prepare(const CNFGrammar& g) {
  Grammar p;p.ids.push_back(g.start);
  for(const auto& row:g.unary) for(int id:row.second) p.ids.push_back(id);
  for(const auto& rule:g.binary) for(int id:rule) p.ids.push_back(id);
  std::sort(p.ids.begin(),p.ids.end());p.ids.erase(std::unique(p.ids.begin(),p.ids.end()),p.ids.end());
  /// @brief Map one already-collected original id to its dense sorted index.
  /// @param id Original nonterminal already present in p.ids.
  /// @return Zero-based index in the sorted distinct vocabulary.
  const auto index=[&p](int id) {return static_cast<std::size_t>(std::lower_bound(p.ids.begin(),p.ids.end(),id)-p.ids.begin());};
  p.start=index(g.start);
  for(const auto& row:g.unary) for(int id:row.second) p.terminals[static_cast<unsigned char>(row.first)].push_back(index(id));
  p.binary.reserve(g.binary.size());for(const auto& rule:g.binary) p.binary.push_back({index(rule[0]),index(rule[1]),index(rule[2])});
  return p;
}
/// @brief Compact first-choice data for one potential dense membership slot.
struct Choice {
  std::size_t split{0}; ///< Zero for a terminal witness, otherwise positive left length.
  std::size_t rule{std::numeric_limits<std::size_t>::max()}; ///< Original binary rule index, SIZE_MAX for a terminal.
};
/// @brief Run the actual dense membership recurrence, optionally retaining first choices and work.
/// @param g Public grammar for original witness ids/rules.
/// @param s Input byte string.
/// @param p Prepared dense grammar.
/// @param shape Checked geometry and worst-case counts.
/// @param out Optional complete analysis destination; null avoids capture/counter work.
/// @return Whether the start occurs in the full nonempty span.
/// @throws std::length_error Container capacity exceeded.
/// @throws std::overflow_error Actual captured counter overflow.
/// @throws std::bad_alloc Dense table, offsets, choices or output allocation failure.
bool run(const CNFGrammar& g,const std::string& s,const Grammar& p,const CykShape& shape,CykAnalysis* out) {
  const auto n=s.size(),m=p.ids.size();if(!n) return false;
  if(shape.state_slots>std::vector<std::uint8_t>().max_size() || shape.cells>std::vector<std::size_t>().max_size() || (out && (shape.state_slots>std::vector<Choice>().max_size() || shape.cells>std::vector<CykCell>().max_size()))) throw std::length_error("CYK: chart capacity exceeded");
  std::vector<std::uint8_t> member(shape.state_slots,0);std::vector<std::size_t> count(shape.cells,0),offset(n+1,0);
  std::size_t next=0;for(std::size_t len=1;len<=n;++len) {offset[len]=next;next+=n-len+1;}
  std::vector<Choice> choices;if(out) {choices.resize(shape.state_slots);out->input=s;out->cells.resize(shape.cells);}
  /// @brief Capture one completed cell by scanning dense ids in sorted original-id order.
  /// @param begin Starting input byte offset of the completed span.
  /// @param len Positive completed span length.
  /// @throws std::length_error Public entry-vector capacity exceeded.
  /// @throws std::bad_alloc Public entry allocation failure.
  const auto capture=[&](std::size_t begin,std::size_t len) {
    if(!out) return;const auto c=offset[len]+begin;auto& cell=out->cells[c];cell.begin=begin;cell.length=len;cell.entries.reserve(count[c]);
    for(std::size_t id=0;id<m;++id) if(member[c*m+id]) {
      const auto choice=choices[c*m+id];CykEntry e;e.nonterminal=p.ids[id];e.left_length=choice.split;e.binary_rule=choice.rule;
      if(choice.split) {e.left=g.binary[choice.rule][1];e.right=g.binary[choice.rule][2];}else e.terminal=s[begin];
      cell.entries.push_back(e);
    }
  };
  for(std::size_t i=0;i<n;++i) {
    if(out) tick(out->work.terminal_lookups);
    for(auto id:p.terminals[static_cast<unsigned char>(s[i])]) {
      if(out) tick(out->work.terminal_rule_tests);const auto slot=i*m+id;
      if(!member[slot]) {member[slot]=1;++count[i];if(out) tick(out->work.new_states);}
    }
    capture(i,1);
  }
  for(std::size_t len=2;len<=n;++len) for(std::size_t begin=0;begin<=n-len;++begin) {
    const auto cell=offset[len]+begin;
    for(std::size_t split=1;split<len;++split) {
      if(out) tick(out->work.split_points);const auto left=offset[split]+begin,right=offset[len-split]+begin+split;
      if(!count[left] || !count[right]) {if(out) tick(out->work.skipped_empty_splits);continue;}
      for(std::size_t r=0;r<p.binary.size();++r) {
        if(out) tick(out->work.binary_rule_tests);const auto& rule=p.binary[r];
        if(member[left*m+rule[1]] && member[right*m+rule[2]]) {
          if(out) tick(out->work.matching_rules);const auto slot=cell*m+rule[0];
          if(!member[slot]) {member[slot]=1;++count[cell];if(out) {choices[slot]={split,r};tick(out->work.new_states);}}
        }
      }
    }
    capture(begin,len);
  }
  return member[offset[n]*m+p.start]!=0;
}
/// @brief Find a unique witness for one id without assuming edited entries are sorted.
/// @param cell Checked coordinate cell.
/// @param id Original nonterminal id.
/// @return Borrowed entry pointer, or null when absent.
/// @throws std::invalid_argument Duplicate entries for the requested id.
const CykEntry* find(const CykCell& cell,int id) {
  const CykEntry* entry=nullptr;for(const auto& e:cell.entries) if(e.nonterminal==id) {if(entry) throw std::invalid_argument("CYK: duplicate witness");entry=&e;}return entry;
}
/// @brief Pending preorder task with decreasing child lengths.
struct Task {
  int id{0}; ///< Original nonterminal to recover.
  std::size_t begin{0}; ///< Child span start.
  std::size_t length{0}; ///< Child span positive length.
  std::size_t parent{std::numeric_limits<std::size_t>::max()}; ///< Parent output index, SIZE_MAX for root.
  bool right{false}; ///< Attach as a right child when true, otherwise left.
};

bool recognize(const CNFGrammar& g,const std::string& s) {
  if(s.empty()) return false;const auto p=prepare(g);return run(g,s,p,cyk_shape(s.size(),p.ids.size(),p.binary.size()),nullptr);
}
} // namespace cyk_detail

CykShape cyk_shape(std::size_t n,std::size_t m,std::size_t r) {
  if(!m) throw std::invalid_argument("CYK: nonterminal count must include start");
  CykShape p;p.input_size=n;p.nonterminal_count=m;p.binary_rule_count=r;p.cells=cyk_detail::triangle(n);
  if(n>=2) {std::array<std::size_t,3> f{n-1,n,cyk_detail::add(n,1)};
    for(auto divisor:{std::size_t{2},std::size_t{3}}) for(auto& factor:f) if(factor%divisor==0) {factor/=divisor;break;}
    p.splits=cyk_detail::mul(cyk_detail::mul(f[0],f[1]),f[2]);
  }
  p.state_slots=cyk_detail::mul(p.cells,m);p.max_rule_tests=cyk_detail::mul(p.splits,r);return p;
}
CykAnalysis cyk_analyze(const CNFGrammar& g,const std::string& s,std::size_t max_states,std::size_t max_tests) {
  auto p=cyk_detail::prepare(g);const auto shape=cyk_shape(s.size(),p.ids.size(),p.binary.size());
  if((max_states && shape.state_slots>max_states)||(max_tests && shape.max_rule_tests>max_tests)) throw std::length_error("CYK: complete potential capacity cap exceeded");
  CykAnalysis a;a.start_symbol=g.start;a.shape=shape;a.accepted=cyk_detail::run(g,s,p,shape,&a);a.nonterminals=std::move(p.ids);return a;
}
const CykCell& cyk_cell(const CykAnalysis& a,std::size_t begin,std::size_t len) {
  const auto n=a.input.size();if(!len || begin>=n || len>n-begin) throw std::out_of_range("CYK: invalid cell span");
  const auto total=cyk_detail::triangle(n);if(a.cells.size()!=total) throw std::invalid_argument("CYK: malformed chart size");
  const auto index=total-cyk_detail::triangle(n-len+1)+begin;const auto& cell=a.cells[index];
  if(cell.begin!=begin || cell.length!=len) throw std::invalid_argument("CYK: malformed cell coordinates");return cell;
}
std::vector<CykParseNode> cyk_first_parse(const CykAnalysis& a,std::size_t max_nodes) {
  const auto n=a.input.size();if(a.cells.size()!=cyk_detail::triangle(n)) throw std::invalid_argument("CYK: malformed chart size");
  if(!n) {if(a.accepted) throw std::invalid_argument("CYK: empty input cannot accept");return {};}
  const auto* root=cyk_detail::find(cyk_cell(a,0,n),a.start_symbol);
  if(a.accepted!=(root!=nullptr)) throw std::invalid_argument("CYK: inconsistent root acceptance");if(!root) return {};
  const auto needed=cyk_detail::add(n,n-1);if(max_nodes && needed>max_nodes) throw std::length_error("CYK: complete parse-node cap exceeded");
  std::vector<CykParseNode> nodes;nodes.reserve(needed);std::vector<cyk_detail::Task> todo{{a.start_symbol,0,n}};
  while(!todo.empty()) {
    const auto task=todo.back();todo.pop_back();const auto* e=cyk_detail::find(cyk_cell(a,task.begin,task.length),task.id);
    if(!e) throw std::invalid_argument("CYK: missing child witness");
    const auto index=nodes.size();if(index>=needed) throw std::invalid_argument("CYK: oversized witness tree");
    CykParseNode node;node.nonterminal=task.id;node.begin=task.begin;node.length=task.length;
    if(!e->left_length) {
      if(task.length!=1 || e->terminal!=a.input[task.begin] || e->binary_rule!=std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("CYK: malformed terminal witness");node.terminal=e->terminal;
    }else {
      if(e->left_length>=task.length || e->binary_rule>=a.shape.binary_rule_count) throw std::invalid_argument("CYK: malformed binary witness");
      todo.push_back({e->right,task.begin+e->left_length,task.length-e->left_length,index,true});
      todo.push_back({e->left,task.begin,e->left_length,index,false});
    }
    nodes.push_back(node);
    if(task.parent!=std::numeric_limits<std::size_t>::max()) {if(task.right) nodes[task.parent].right_child=index;else nodes[task.parent].left_child=index;}
  }
  if(nodes.size()!=needed) throw std::invalid_argument("CYK: incomplete witness tree");return nodes;
}
} // namespace datamunge::algorithms
