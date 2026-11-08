/// @file elias_gamma.cpp
/// @brief Checked uint64 gamma emission, strict complete parsing and actual span capture.
#include <datamunge/algorithms/elias_gamma.hpp>
#include <bit>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Pure size preflight and shared actual encoder/parser helpers.
namespace gamma_detail {
/// @brief Add logical bit/record counts without wrapping.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a,std::size_t b){
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("gamma count overflow");return a+b;
}
/// @brief Multiply counts without wrapping.
/// @param a First factor.
/// @param b Second factor.
/// @return Checked product.
/// @throws std::length_error On size_t overflow.
std::size_t multiply(std::size_t a,std::size_t b){
    if(a&&b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("gamma count product overflow");return a*b;
}
/// @brief Scalar encoding geometry, requiring no encoded string or per-value cache.
struct Plan {
    std::size_t bits=0; ///< Complete gamma payload size.
    std::uint64_t maximum=0; ///< Maximum positive value, or zero for empty input.
    std::size_t width=0; ///< Maximum binary width, or zero for empty input.
};
/// @brief Validate positive values and compute exact bit geometry before allocation.
/// @param values Raw positive values.
/// @return Bit count and maximum/width, all zero for empty input.
/// @throws std::invalid_argument On zero.
/// @throws std::length_error On bit-count overflow.
Plan plan(const std::vector<std::uint64_t>& values){
    Plan p;
    for(auto n:values){const auto length=elias_gamma_code_length(n);p.bits=add(p.bits,length);
        if(n>p.maximum){p.maximum=n;p.width=(length+1)/2;}}
    return p;
}
/// @brief Check output-string geometry without allocating it.
/// @param bits Complete output length.
/// @throws std::length_error On string size limits.
void string_geometry(std::size_t bits){if(bits>std::string{}.max_size())throw std::length_error("gamma bit string exceeds container limits");}
/// @brief Validate every input byte before parsing, including unread tails and NULs.
/// @param bits Borrowed bit characters.
/// @throws std::invalid_argument On anything other than literal zero or one.
void validate_bits(const std::string& bits){
    for(char c:bits)if(c!='0'&&c!='1')throw std::invalid_argument("gamma bit string must contain only zero and one");
}
/// @brief Reserve complete retained logical records before input copying or append.
/// @param trace Owned capture in progress.
/// @param count Additional entries.
/// @param cap Optional quota; zero disables it.
/// @throws std::length_error On quota or arithmetic overflow.
void capture(GammaTrace& trace,std::size_t count,std::size_t cap){
    const auto total=add(trace.captured_entries,count);
    if(cap&&total>cap)throw std::length_error("gamma capture limit exceeded");trace.captured_entries=total;
}
/// @brief Count one complete word actually emitted or consumed by the kernel.
/// @param work Existing checked counters.
/// @param prefix Actual zero-prefix length.
/// @param payload Actual binary length including its leading one.
/// @throws std::length_error On logical count overflow.
void record_work(GammaWork& work,std::size_t prefix,std::size_t payload){
    work.values=add(work.values,1);work.zero_prefix_bits=add(work.zero_prefix_bits,prefix);
    work.payload_bits=add(work.payload_bits,payload);work.bit_visits=add(work.bit_visits,add(prefix,payload));
}
/// @brief Actual emitter, optionally appending spans while writing the ordinary output.
/// @param values Valid positive values, unchanged.
/// @param p Preflight bit geometry fitting the output string.
/// @param trace Optional owned capture with complete quota already reserved.
/// @return Owned emitted characters and actual counters.
/// @throws std::length_error On counter/container limits.
GammaEncoded emit(const std::vector<std::uint64_t>& values,const Plan& p,GammaTrace* trace){
    GammaEncoded out;out.bits.reserve(p.bits);
    for(auto n:values){const auto width=static_cast<std::size_t>(std::bit_width(n)),prefix=width-1,begin=out.bits.size();
        out.bits.append(prefix,'0');const auto payload_begin=out.bits.size();
        for(auto b=width;b>0;--b)out.bits.push_back((n>>(b-1))&1?'1':'0');
        if(trace)trace->words.push_back({n,begin,payload_begin,out.bits.size()});
        record_work(out.work,prefix,width);
    }
    return out;
}
/// @brief Actual strict parser, optionally recording spans before each value append.
/// @param bits Validated binary alphabet; structural completeness checked here.
/// @param trace Optional owned capture, already reserving the entire input string.
/// @param cap Optional retained-history limit.
/// @return Parsed values and successful consumed-bit counts.
/// @throws std::invalid_argument On truncated words.
/// @throws std::overflow_error On a prefix wider than a uint64 value can represent.
/// @throws std::length_error On quota, counters or container limits.
GammaDecoded parse(const std::string& bits,GammaTrace* trace,std::size_t cap){
    GammaDecoded out;std::size_t i=0;
    while(i<bits.size()){
        const auto begin=i;std::size_t zeros=0;
        while(i<bits.size()&&bits[i]=='0'){
            ++i;if(++zeros>63)throw std::overflow_error("gamma prefix requires more than 64 value bits");
        }
        if(i==bits.size())throw std::invalid_argument("truncated gamma zero prefix");
        const auto payload_begin=i++; // The validated nonzero character is the leading one.
        if(zeros>bits.size()-i)throw std::invalid_argument("truncated gamma binary payload");
        std::uint64_t value=1;
        for(std::size_t b=0;b<zeros;++b)value=(value<<1)|static_cast<std::uint64_t>(bits[i++]=='1');
        if(trace)capture(*trace,2,cap);
        out.values.push_back(value);
        if(trace)trace->words.push_back({value,begin,payload_begin,i});
        record_work(out.work,zeros,zeros+1);
    }
    return out;
}
} // namespace gamma_detail

std::size_t elias_gamma_code_length(std::uint64_t n){
    if(!n)throw std::invalid_argument("gamma coding requires positive integers");
    return 2*static_cast<std::size_t>(std::bit_width(n))-1;
}
GammaSizeSummary elias_gamma_size_summary(const std::vector<std::uint64_t>& values){
    const auto p=gamma_detail::plan(values);
    return {values.size(),p.maximum,p.bits,p.width,gamma_detail::multiply(values.size(),p.width)};
}
GammaEncoded elias_gamma_encode_work(const std::vector<std::uint64_t>& values){
    const auto p=gamma_detail::plan(values);gamma_detail::string_geometry(p.bits);return gamma_detail::emit(values,p,nullptr);
}
GammaDecoded elias_gamma_decode_work(const std::string& bits){
    gamma_detail::validate_bits(bits);return gamma_detail::parse(bits,nullptr,0);
}
GammaTrace elias_gamma_encode_trace(const std::vector<std::uint64_t>& values,std::size_t max_entries){
    const auto p=gamma_detail::plan(values);gamma_detail::string_geometry(p.bits);
    if(values.size()>std::vector<GammaWord>{}.max_size())throw std::length_error("gamma word history exceeds container limits");
    GammaTrace trace;gamma_detail::capture(trace,gamma_detail::add(p.bits,gamma_detail::multiply(values.size(),2)),max_entries);
    trace.encoding=true;trace.values=values;trace.words.reserve(values.size());
    auto out=gamma_detail::emit(trace.values,p,&trace);trace.bits=std::move(out.bits);trace.work=out.work;return trace;
}
GammaTrace elias_gamma_decode_trace(const std::string& bits,std::size_t max_entries){
    gamma_detail::validate_bits(bits);GammaTrace trace;gamma_detail::capture(trace,bits.size(),max_entries);trace.bits=bits;
    auto out=gamma_detail::parse(trace.bits,&trace,max_entries);trace.values=std::move(out.values);trace.work=out.work;return trace;
}
} // namespace datamunge::algorithms
