/// @file
/// @brief Shared checked Lowrance–Wagner recurrence, local OSA comparison and executable byte scripts.
#include <datamunge/algorithms/damerau_levenshtein.hpp>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Checked dimensions/candidates and actual unrestricted-distance operations.
namespace damerau_detail {
/// @brief Add representable nonnegative size counts.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error Sum cannot fit size_t.
std::size_t add(std::size_t a,std::size_t b) {if(b>std::numeric_limits<std::size_t>::max()-a) throw std::length_error("Damerau: size sum overflow");return a+b;}
/// @brief Multiply representable nonnegative capacities.
/// @param a First factor.
/// @param b Second factor.
/// @return Exact product.
/// @throws std::length_error Product cannot fit size_t.
std::size_t mul(std::size_t a,std::size_t b) {if(a && b>std::numeric_limits<std::size_t>::max()/a) throw std::length_error("Damerau: cell product overflow");return a*b;}
/// @brief Add candidate costs, clipping safely at the finite sentinel.
/// @param a Nonnegative stored candidate/base.
/// @param b Nonnegative additional cost.
/// @param bound Checked n+m sentinel.
/// @return min(a+b,bound), without forming an overflowing sum.
std::size_t capped(std::size_t a,std::size_t b,std::size_t bound) {return a>=bound || b>=bound-a?bound:a+b;}
/// @brief Increment an actual aggregate work counter without wrapping.
/// @param count Counter to increment.
/// @throws std::overflow_error Count is SIZE_MAX.
void tick(std::size_t& count) {if(count==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("Damerau: work overflow");++count;}
/// @brief Execute the canonical full-table unit-cost Lowrance–Wagner recurrence once.
/// @param a Source bytes.
/// @param b Target bytes.
/// @param p Checked table shape.
/// @param out Optional actual table/cell/work destination; null avoids capture.
/// @return Exact unrestricted distance.
/// @throws std::length_error Table/record capacity exceeded.
/// @throws std::overflow_error Captured counter overflow.
/// @throws std::bad_alloc Table/record allocation failure.
std::size_t run(const std::string& a,const std::string& b,const DamerauShape& p,DamerauTrace* out) {
  std::vector<std::size_t> local;auto& h=out?out->table:local;h.assign(p.cells,0);if(out) out->cells.reserve(p.interior_cells);
  const auto n=a.size(),m=b.size(),cols=p.columns,inf=p.bound;h[0]=inf;
  for(std::size_t i=0;i<=n;++i) {h[(i+1)*cols]=inf;h[(i+1)*cols+1]=i;}
  for(std::size_t j=0;j<=m;++j) {h[j+1]=inf;h[cols+j+1]=j;}
  std::array<std::size_t,256> last{};
  for(std::size_t i=1;i<=n;++i) {
    std::size_t column=0;
    for(std::size_t j=1;j<=m;++j) {
      DamerauCell c;c.source_prefix=i;c.target_prefix=j;c.last_source=last[static_cast<unsigned char>(b[j-1])];c.last_target=column;
      const bool equal=a[i-1]==b[j-1];if(equal) column=j;
      c.diagonal=capped(h[i*cols+j],equal?0:1,inf);c.insertion=capped(h[(i+1)*cols+j],1,inf);c.deletion=capped(h[i*cols+j+1],1,inf);c.transposition=inf;
      if(c.last_source && c.last_target) {
        c.transposition=capped(h[c.last_source*cols+c.last_target],i-c.last_source-1,inf);
        c.transposition=capped(c.transposition,1,inf);c.transposition=capped(c.transposition,j-c.last_target-1,inf);
        if(out) tick(out->work.eligible_transpositions);
      }
      c.value=c.diagonal;c.choice=equal?DamerauMove::match:DamerauMove::substitute;
      if(c.insertion<c.value) {c.value=c.insertion;c.choice=DamerauMove::insert;}
      if(c.deletion<c.value) {c.value=c.deletion;c.choice=DamerauMove::erase;}
      if(c.transposition<c.value) {c.value=c.transposition;c.choice=DamerauMove::transpose;}
      h[(i+1)*cols+j+1]=c.value;
      if(out) {tick(out->work.cells_evaluated);tick(out->work.byte_comparisons);if(equal) tick(out->work.byte_matches);if(c.choice==DamerauMove::transpose) tick(out->work.selected_transpositions);out->cells.push_back(c);}
    }
    last[static_cast<unsigned char>(a[i-1])]=i;if(out) tick(out->work.last_row_updates);
  }
  return h[(n+1)*cols+m+1];
}
/// @brief One locally validated prefix traceback block, later expanded into unit edits.
struct Block {
  std::size_t source_begin{0}; ///< Source prefix before this block.
  std::size_t target_begin{0}; ///< Target prefix before this block.
  std::size_t source_length{0}; ///< Source bytes consumed.
  std::size_t target_length{0}; ///< Target bytes produced.
  DamerauMove move{DamerauMove::match}; ///< Selected move; bridge blocks can span several bytes.
};
} // namespace damerau_detail

DamerauShape damerau_shape(std::size_t n,std::size_t m) {
  DamerauShape p;p.rows=damerau_detail::add(n,2);p.columns=damerau_detail::add(m,2);p.cells=damerau_detail::mul(p.rows,p.columns);p.interior_cells=damerau_detail::mul(n,m);p.bound=damerau_detail::add(n,m);return p;
}
std::size_t damerau_levenshtein_distance(const std::string& a,const std::string& b) {return damerau_detail::run(a,b,damerau_shape(a.size(),b.size()),nullptr);}
DamerauTrace damerau_levenshtein_trace(const std::string& a,const std::string& b,std::size_t max_cells) {
  const auto p=damerau_shape(a.size(),b.size());if(max_cells && p.cells>max_cells) throw std::length_error("Damerau: complete cell cap exceeded");
  DamerauTrace t;t.source=a;t.target=b;t.shape=p;t.distance=damerau_detail::run(a,b,p,&t);return t;
}
std::size_t damerau_prefix_distance(const DamerauTrace& t,std::size_t i,std::size_t j) {
  if(i>t.source.size() || j>t.target.size()) throw std::out_of_range("Damerau: prefix outside input");
  const auto p=damerau_shape(t.source.size(),t.target.size());if(t.table.size()!=p.cells) throw std::invalid_argument("Damerau: malformed table size");return t.table[(i+1)*p.columns+j+1];
}
std::vector<ByteEdit> damerau_edit_script(const DamerauTrace& t,std::size_t max_edits) {
  const auto n=t.source.size(),m=t.target.size();const auto p=damerau_shape(n,m);
  if(t.table.size()!=p.cells || t.cells.size()!=p.interior_cells || t.distance>std::max(n,m) || t.table[p.columns+1]!=0 || t.table[(n+1)*p.columns+m+1]!=t.distance) throw std::invalid_argument("Damerau: malformed trace shape/distance");
  if(max_edits && t.distance>max_edits) throw std::length_error("Damerau: complete edit cap exceeded");
  std::vector<damerau_detail::Block> blocks;std::size_t i=n,j=m,cost=0;
  while(i || j) {
    const auto current=t.table[(i+1)*p.columns+j+1];std::size_t pi=i,pj=j,step=0;DamerauMove move;
    if(!i) {move=DamerauMove::insert;--pj;step=1;if(current!=j) throw std::invalid_argument("Damerau: malformed insertion border");}
    else if(!j) {move=DamerauMove::erase;--pi;step=1;if(current!=i) throw std::invalid_argument("Damerau: malformed deletion border");}
    else {
      const auto& c=t.cells[(i-1)*m+j-1];if(c.source_prefix!=i || c.target_prefix!=j || c.value!=current || current>std::max(i,j)) throw std::invalid_argument("Damerau: malformed visited cell");move=c.choice;
      switch(move) {
        case DamerauMove::match:case DamerauMove::substitute:
          if((t.source[i-1]==t.target[j-1])!=(move==DamerauMove::match)) throw std::invalid_argument("Damerau: invalid diagonal witness");--pi;--pj;step=move==DamerauMove::match?0:1;break;
        case DamerauMove::insert:--pj;step=1;break;
        case DamerauMove::erase:--pi;step=1;break;
        case DamerauMove::transpose:
          if(!c.last_source || !c.last_target || c.last_source>=i || c.last_target>=j || t.source[c.last_source-1]!=t.target[j-1] || t.source[i-1]!=t.target[c.last_target-1]) throw std::invalid_argument("Damerau: invalid crossed-byte witness");
          pi=c.last_source-1;pj=c.last_target-1;step=damerau_detail::add(i-c.last_source-1,j-c.last_target-1);step=damerau_detail::add(step,1);break;
        default:throw std::invalid_argument("Damerau: invalid witness move");
      }
    }
    const auto previous=t.table[(pi+1)*p.columns+pj+1];if(previous>p.bound || step>p.bound-previous || previous+step!=current) throw std::invalid_argument("Damerau: inconsistent witness recurrence");
    cost=damerau_detail::add(cost,step);blocks.push_back({pi,pj,i-pi,j-pj,move});i=pi;j=pj;
  }
  if(cost!=t.distance) throw std::invalid_argument("Damerau: inconsistent total script cost");
  std::reverse(blocks.begin(),blocks.end());std::vector<ByteEdit> edits;edits.reserve(cost);
  for(const auto& block:blocks) {
    const auto pos=block.target_begin;
    switch(block.move) {
      case DamerauMove::match:break;
      case DamerauMove::substitute:edits.push_back({block.move,pos,t.source.substr(block.source_begin,1),t.target.substr(pos,1)});break;
      case DamerauMove::insert:edits.push_back({block.move,pos,"",t.target.substr(pos,1)});break;
      case DamerauMove::erase:edits.push_back({block.move,pos,t.source.substr(block.source_begin,1),""});break;
      case DamerauMove::transpose: {
        for(std::size_t k=1;k+1<block.source_length;++k) edits.push_back({DamerauMove::erase,pos+1,t.source.substr(block.source_begin+k,1),""});
        std::string before;before.push_back(t.source[block.source_begin]);before.push_back(t.source[block.source_begin+block.source_length-1]);auto after=before;std::swap(after[0],after[1]);edits.push_back({DamerauMove::transpose,pos,before,after});
        for(std::size_t k=1;k+1<block.target_length;++k) edits.push_back({DamerauMove::insert,pos+k,"",t.target.substr(pos+k,1)});break;
      }
    }
  }
  return edits;
}
std::string apply_byte_edits(std::string source,const std::vector<ByteEdit>& edits) {
  for(const auto& e:edits) {
    bool valid=false;
    switch(e.kind) {
      case DamerauMove::match:valid=e.before.size()==1 && e.after==e.before;break;
      case DamerauMove::substitute:valid=e.before.size()==1 && e.after.size()==1 && e.before!=e.after;break;
      case DamerauMove::insert:valid=e.before.empty() && e.after.size()==1;break;
      case DamerauMove::erase:valid=e.before.size()==1 && e.after.empty();break;
      case DamerauMove::transpose:valid=e.before.size()==2 && e.after.size()==2 && e.before[0]==e.after[1] && e.before[1]==e.after[0];break;
    }
    if(!valid) throw std::invalid_argument("ByteEdit: invalid kind/payload");
    if(e.position>source.size() || e.before.size()>source.size()-e.position) throw std::out_of_range("ByteEdit: current position/span outside string");
    if(source.compare(e.position,e.before.size(),e.before)!=0) throw std::invalid_argument("ByteEdit: expected bytes differ");
    source.replace(e.position,e.before.size(),e.after);
  }
  return source;
}
std::size_t optimal_string_alignment_distance(const std::string& a,const std::string& b) {
  const auto& s=a.size()<=b.size()?a:b;const auto& t=a.size()<=b.size()?b:a;const auto columns=damerau_detail::add(s.size(),1),bound=damerau_detail::add(a.size(),b.size());
  std::vector<std::size_t> prev2(columns),prev(columns),curr(columns);for(std::size_t j=0;j<columns;++j) prev[j]=j;
  for(std::size_t i=1;i<=t.size();++i) {
    curr[0]=i;
    for(std::size_t j=1;j<columns;++j) {
      curr[j]=std::min({damerau_detail::capped(prev[j-1],s[j-1]==t[i-1]?0:1,bound),damerau_detail::capped(prev[j],1,bound),damerau_detail::capped(curr[j-1],1,bound)});
      if(i>1 && j>1 && t[i-1]==s[j-2] && t[i-2]==s[j-1]) curr[j]=std::min(curr[j],damerau_detail::capped(prev2[j-2],1,bound));
    }
    prev2.swap(prev);prev.swap(curr);
  }
  return prev[s.size()];
}
} // namespace datamunge::algorithms
