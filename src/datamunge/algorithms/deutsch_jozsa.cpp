/// @file deutsch_jozsa.cpp
/// @brief Shared executing Deutsch–Jozsa circuit and sequential promised comparison.
#include <datamunge/algorithms/deutsch_jozsa.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Local checked dimensions and shared gate kernel; no shared mutable state.
namespace {
/// @brief Check domain and callable before shifts, allocation or callback evaluation.
/// @param n Nonnegative qubits below size_t's value-bit count.
/// @param f Required callback, not evaluated here.
/// @return Representable 2^n without allocation.
/// @throws std::invalid_argument For invalid n or empty f.
std::size_t dimension(int n,const std::function<int(std::size_t)>& f) {
  if(n<0 || n>=std::numeric_limits<std::size_t>::digits)
    throw std::invalid_argument("deutsch_jozsa: invalid qubit count");
  if(!f)throw std::invalid_argument("deutsch_jozsa: empty callback");
  return std::size_t{1}<<n;
}
/// @brief Execute legacy gate order, optionally snapshotting the actual boundaries.
/// @param n Validated qubit count.
/// @param size Checked domain size 2^n.
/// @param f Callback evaluated once per ascending x; low bit used.
/// @param trace Null for ordinary calls; otherwise receives owned states and counts.
/// @return Unclipped numerical result with the original >0.5 threshold.
/// @throws std::bad_alloc For allocation failure; callback exceptions propagate.
DeutschJozsaResult circuit(int n,std::size_t size,const std::function<int(std::size_t)>& f,DeutschJozsaTrace* trace) {
  QuantumState state=qstate_zero(n);
  for(int q=0;q<n;++q)h_gate(state,q);
  if(trace) {
    trace->n=n;trace->before_oracle=state;trace->truth_table.resize(size);
    trace->work.oracle_applications=1;
    trace->work.hadamard_gates=static_cast<std::size_t>(n);
    trace->work.butterflies=static_cast<std::size_t>(n)*(size/2);
  }
  for(std::size_t x=0;x<size;++x) {
    const int bit=f(x)&1;
    if(bit)state[x]=-state[x];
    if(trace) {
      trace->truth_table[x]=bit;trace->ones+=static_cast<std::size_t>(bit);
      ++trace->work.callback_calls;trace->work.phase_flips+=static_cast<std::size_t>(bit);
    }
  }
  if(trace)trace->after_oracle=state;
  for(int q=0;q<n;++q) {
    h_gate(state,q);
    if(trace){++trace->work.hadamard_gates;trace->work.butterflies+=size/2;}
  }
  const double p0=std::norm(state[0]);const DeutschJozsaResult result{p0>0.5,p0};
  if(trace) {
    trace->result=result;trace->final_state=std::move(state);
    trace->mean_phase=1.0-2.0*(static_cast<double>(trace->ones)/static_cast<double>(size));
    trace->promise=(trace->ones==0 || trace->ones==size)?DeutschJozsaPromise::constant:
      (size>1 && trace->ones==size/2)?DeutschJozsaPromise::balanced:DeutschJozsaPromise::neither;
  }
  return result;
}
} // namespace
DeutschJozsaResult deutsch_jozsa(int n,const std::function<int(std::size_t)>& f) {
  const auto size=dimension(n,f);
  if(size>QuantumState{}.max_size())throw std::length_error("deutsch_jozsa: state capacity");
  return circuit(n,size,f,nullptr);
}
DeutschJozsaTrace deutsch_jozsa_trace(int n,const std::function<int(std::size_t)>& f,std::size_t max_basis_states) {
  const auto size=dimension(n,f);
  if((max_basis_states && size>max_basis_states) || size>QuantumState{}.max_size() ||
     size>std::vector<int>{}.max_size() ||
     (n>0 && size>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(n)))
    throw std::length_error("deutsch_jozsa_trace: state cap, capacity or work overflow");
  DeutschJozsaTrace trace;circuit(n,size,f,&trace);return trace;
}
DeutschJozsaClassicalResult deutsch_jozsa_classical(int n,const std::function<int(std::size_t)>& f) {
  const auto size=dimension(n,f);const auto limit=size==1?std::size_t{1}:size/2+1;
  const int first=f(0)&1;
  for(std::size_t x=1;x<limit;++x)if((f(x)&1)!=first)return {false,x+1};
  return {true,limit};
}
} // namespace datamunge::algorithms
