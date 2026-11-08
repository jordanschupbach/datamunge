/// @file exp_golomb.cpp
/// @brief Portable full-range Exp-Golomb emission and bounded-prefix parsing with shared actual spans.
#include <datamunge/algorithms/exp_golomb.hpp>
#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Checked geometry and shared actual coding kernels; no explanatory replay.
namespace exp_golomb_detail {
/// @brief Validate the native order before any shifts, including on empty input.
/// @param k Requested common order.
/// @throws std::invalid_argument Outside [0,63].
void order(int k){if(k<0||k>63)throw std::invalid_argument("Exp-Golomb order must be in [0,63]");}
/// @brief Checked logical count addition.
/// @param a Existing count.
/// @param b Increment.
/// @return Nonwrapping sum.
/// @throws std::length_error On overflow.
std::size_t add(std::size_t a,std::size_t b){if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("Exp-Golomb count overflow");return a+b;}
/// @brief Checked logical count multiplication.
/// @param a First count.
/// @param b Second count.
/// @return Nonwrapping product.
/// @throws std::length_error On overflow.
std::size_t multiply(std::size_t a,std::size_t b){if(a&&b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("Exp-Golomb count product overflow");return a*b;}
/// @brief Mathematical floor(log2(q+1)) without forming 2^64.
/// @param q Native unsigned quotient.
/// @return Leading zero count, in [0,64].
std::size_t zeros(std::uint64_t q){return q==std::numeric_limits<std::uint64_t>::max()?64:std::bit_width(q+1)-1;}
/// @brief Smallest quotient with the given zero-run length.
/// @param m Valid zero count, in [0,64].
/// @return Mathematical 2^m-1 represented without a shift by64.
std::uint64_t base(std::size_t m){return m==64?std::numeric_limits<std::uint64_t>::max():(std::uint64_t{1}<<m)-1;}
/// @brief Preflight complete coded size, using only scalar storage.
/// @param values Borrowed native values.
/// @param k Valid common order.
/// @return Complete logical output size.
/// @throws std::length_error On counter/container limits.
std::size_t geometry(const std::vector<std::uint64_t>& values,int k){
    std::size_t b=0;for(auto n:values)b=add(b,2*zeros(n>>k)+1+k);
    if(b>std::vector<bool>{}.max_size())throw std::length_error("Exp-Golomb bit container limit");return b;
}
/// @brief Reserve defined logical capture entries before copies or appends.
/// @param trace Owned capture under construction.
/// @param n Additional entries.
/// @param cap Quota; zero disables it.
/// @throws std::length_error On quota or arithmetic limits.
void capture(ExpGolombTrace& trace,std::size_t n,std::size_t cap){auto c=add(trace.captured_entries,n);if(cap&&c>cap)throw std::length_error("Exp-Golomb capture limit");trace.captured_entries=c;}
/// @brief Count actual positions in one completed emitted or parsed word.
/// @param w Accumulated work.
/// @param m Actual leading zeros.
/// @param k Actual low-bit suffix length.
/// @throws std::length_error On counter overflow.
void record(ExpGolombWork& w,std::size_t m,int k){
    w.values=add(w.values,1);w.zero_prefix_bits=add(w.zero_prefix_bits,m);
    w.quotient_bits=add(w.quotient_bits,m+1);w.suffix_bits=add(w.suffix_bits,k);
    w.bit_visits=add(w.bit_visits,2*m+1+k);
}
/// @brief Emit one word into pre-reserved output and return its actual indices.
/// @param n Native value.
/// @param out Bit container with enough capacity.
/// @param k Valid common order.
/// @return Actual half-open boundaries and original value.
ExpGolombWord emit_word(std::uint64_t n,std::vector<bool>& out,int k){
    const auto q=n>>k;const auto m=zeros(q);const auto tail=q-base(m);const auto begin=out.size();
    for(std::size_t j=0;j<m;++j)out.push_back(false);
    const auto payload=out.size();out.push_back(true);
    for(auto j=m;j>0;--j)out.push_back((tail>>(j-1))&1);
    const auto suffix=out.size();for(int j=k;j>0;--j)out.push_back((n>>(j-1))&1);
    return {n,begin,payload,suffix,out.size()};
}
/// @brief Shared sequence emitter with optional actual span capture.
/// @param values Borrowed native values.
/// @param k Valid common order.
/// @param b Preflight output geometry.
/// @param trace Optional capture with full quota/word capacity reserved.
/// @return Owned bits and actual work.
/// @throws std::bad_alloc On output allocation.
/// @throws std::length_error On counter limits.
ExpGolombEncoded emit(const std::vector<std::uint64_t>& values,int k,std::size_t b,ExpGolombTrace* trace){
    ExpGolombEncoded out;out.bits.reserve(b);
    for(auto n:values){auto word=emit_word(n,out.bits,k);record(out.work,word.payload_begin-word.begin,k);if(trace)trace->words.push_back(word);}return out;
}
/// @brief Parse only the requested complete-word prefix, recording actual spans online.
/// @param bits Borrowed complete input, possibly containing an unread tail.
/// @param count Nonnegative upper limit, never an allocation request.
/// @param k Valid common order.
/// @param trace Optional capture already owning and accounting for all input bits.
/// @param cap Optional capture quota.
/// @return Complete values, exact consumed boundary and actual work.
/// @throws std::invalid_argument On incomplete attempted words.
/// @throws std::overflow_error On values outside the native range.
/// @throws std::length_error On actual output/counter/capture limits.
/// @throws std::bad_alloc On output or span allocation.
ExpGolombDecoded parse(const std::vector<bool>& bits,int count,int k,ExpGolombTrace* trace,std::size_t cap){
    ExpGolombDecoded out;std::size_t pos=0;
    while(pos<bits.size()&&out.values.size()<static_cast<std::size_t>(count)){
        const auto begin=pos;std::size_t m=0;
        while(pos<bits.size()&&!bits[pos]){++pos;if(++m>static_cast<std::size_t>(64-k))throw std::overflow_error("Exp-Golomb prefix exceeds native range");}
        if(pos==bits.size())throw std::invalid_argument("incomplete Exp-Golomb zero run");
        const auto payload=pos++;
        if(m+static_cast<std::size_t>(k)>bits.size()-pos)throw std::invalid_argument("incomplete Exp-Golomb quotient or suffix");
        std::uint64_t tail=0;for(std::size_t j=0;j<m;++j)tail=(tail<<1)|static_cast<std::uint64_t>(bits[pos++]);
        const auto b=base(m);if(tail>std::numeric_limits<std::uint64_t>::max()-b)throw std::overflow_error("Exp-Golomb quotient overflow");
        const auto q=b+tail;const auto suffix=pos;
        std::uint64_t low=0;for(int j=0;j<k;++j)low=(low<<1)|static_cast<std::uint64_t>(bits[pos++]);
        if(q>(std::numeric_limits<std::uint64_t>::max()>>k))throw std::overflow_error("Exp-Golomb value overflow");
        const auto value=(q<<k)|low;
        if(out.values.size()==out.values.max_size()||(trace&&trace->words.size()==trace->words.max_size()))throw std::length_error("Exp-Golomb output container limit");
        if(trace)capture(*trace,2,cap);
        out.values.push_back(value);if(trace)trace->words.push_back({value,begin,payload,suffix,pos});record(out.work,m,k);
    }
    out.consumed_bits=pos;return out;
}
/// @brief Validate all decoder scalar parameters before input inspection/copy.
/// @param count Requested nonnegative upper limit.
/// @param k Common native order.
/// @throws std::invalid_argument On invalid count/order.
void parameters(int count,int k){order(k);if(count<0)throw std::invalid_argument("Exp-Golomb count must be nonnegative");}
} // namespace exp_golomb_detail
std::size_t exp_golomb_code_length(std::uint64_t n,int k){exp_golomb_detail::order(k);return 2*exp_golomb_detail::zeros(n>>k)+1+k;}
ExpGolombSize exp_golomb_size(const std::vector<std::uint64_t>& values,int k){
    exp_golomb_detail::order(k);ExpGolombSize out;out.values=values.size();out.order=k;
    for(auto n:values){out.bits=exp_golomb_detail::add(out.bits,exp_golomb_code_length(n,k));out.maximum=std::max(out.maximum,n);}
    out.fixed_width=values.empty()?0:std::max<std::size_t>(1,std::bit_width(out.maximum));
    out.fixed_bits=exp_golomb_detail::multiply(out.values,out.fixed_width);return out;
}
std::string exp_golomb_bit_string(const std::vector<bool>& bits){
    if(bits.size()>std::string{}.max_size())throw std::length_error("Exp-Golomb debug text limit");
    std::string text;text.reserve(bits.size());for(bool bit:bits)text.push_back(bit?'1':'0');return text;
}
ExpGolombWork exp_golomb_append_work(std::uint64_t n,std::vector<bool>& out,int k){
    const auto length=exp_golomb_code_length(n,k);const auto b=exp_golomb_detail::add(out.size(),length);
    if(b>out.max_size())throw std::length_error("Exp-Golomb append limit");out.reserve(b);
    const auto word=exp_golomb_detail::emit_word(n,out,k);ExpGolombWork w;exp_golomb_detail::record(w,word.payload_begin-word.begin,k);return w;
}
ExpGolombEncoded exp_golomb_encode_work(const std::vector<std::uint64_t>& values,int k){
    exp_golomb_detail::order(k);return exp_golomb_detail::emit(values,k,exp_golomb_detail::geometry(values,k),nullptr);
}
ExpGolombDecoded exp_golomb_decode_work(const std::vector<bool>& bits,int count,int k){exp_golomb_detail::parameters(count,k);return exp_golomb_detail::parse(bits,count,k,nullptr,0);}
ExpGolombTrace exp_golomb_encode_trace(const std::vector<std::uint64_t>& values,int k,std::size_t max_entries){
    exp_golomb_detail::order(k);const auto b=exp_golomb_detail::geometry(values,k);ExpGolombTrace t;t.order=k;
    if(values.size()>t.words.max_size())throw std::length_error("Exp-Golomb word history limit");
    exp_golomb_detail::capture(t,exp_golomb_detail::add(b,exp_golomb_detail::multiply(values.size(),2)),max_entries);
    t.values=values;t.words.reserve(values.size());auto out=exp_golomb_detail::emit(t.values,k,b,&t);
    t.bits=std::move(out.bits);t.work=out.work;t.consumed_bits=b;return t;
}
ExpGolombTrace exp_golomb_decode_trace(const std::vector<bool>& bits,int count,int k,std::size_t max_entries){
    exp_golomb_detail::parameters(count,k);ExpGolombTrace t;t.order=k;exp_golomb_detail::capture(t,bits.size(),max_entries);t.bits=bits;
    auto out=exp_golomb_detail::parse(t.bits,count,k,&t,max_entries);t.values=std::move(out.values);t.work=out.work;t.consumed_bits=out.consumed_bits;return t;
}
} // namespace datamunge::algorithms
