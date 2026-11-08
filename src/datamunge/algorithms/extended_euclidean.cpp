/// @file extended_euclidean.cpp
/// @brief Shared signed128-bit Euclid execution, checked legacy narrowing and actual chronological row certificates.
#include <datamunge/algorithms/extended_euclidean.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Checked native-domain arithmetic and shared ordinary/traced Euclidean execution.
namespace euclid_detail {
/// @brief Compiler-supported intermediate holding all native signed64 quotient/coefficient products.
using Wide=__int128;
/// @brief Require a widened final coefficient to fit the public signed64 field.
/// @param value Actual final normalized coefficient.
/// @return Exact signed64 value.
/// @throws std::overflow_error If the checked coefficient cannot be represented.
std::int64_t coefficient(Wide value){
    if(value<std::numeric_limits<std::int64_t>::min()||value>std::numeric_limits<std::int64_t>::max())throw std::overflow_error("extended Euclid: final coefficient exceeds int64");return static_cast<std::int64_t>(value);
}
/// @brief Admit whole actual row records before their allocation or arithmetic.
/// @param trace Optional local owned trace; null retains no history.
/// @param additional Initial two records or one next derived record.
/// @param cap Optional logical row quota, zero disables it.
/// @throws std::length_error On quota, count or container geometry limits.
void admit(EuclidTrace* trace,std::size_t additional,std::size_t cap){
    if(!trace)return;
    if(additional>std::numeric_limits<std::size_t>::max()-trace->captured_entries)throw std::length_error("Euclid row count overflow");
    const auto count=trace->captured_entries+additional;
    if((cap&&count>cap)||count>trace->rows.max_size())throw std::length_error("Euclid row capture limit exceeded");trace->captured_entries=count;
}
/// @brief Execute the original signed-truncating recurrence with optional actual raw rows.
/// All native input/remainder magnitudes are at most2^63. Canonical running
/// coefficient magnitudes, including terminal zero rows, are at most2^63;
/// loose quotient products are at most2^126, safely inside signed128 updates.
/// Actual division counts are bounded by native word width, so work counts fit.
/// @param a Arbitrary signed64 first input.
/// @param b Arbitrary signed64 second input.
/// @param trace Optional local capture, initially empty; no explanatory replay.
/// @param cap Optional whole-row quota, zero disables it.
/// @return Unsigned-gcd certificate, actual work and final normalization flag.
/// @throws std::length_error On capture admission/container limits.
/// @throws std::overflow_error On a checked final coefficient-range failure.
/// @throws std::bad_alloc On optional history allocation.
EuclidOutcome execute(std::int64_t a,std::int64_t b,EuclidTrace* trace,std::size_t cap){
    Wide old_r=a,r=b,old_s=1,s=0,old_t=0,t=1;EuclidOutcome out;
    if(trace){admit(trace,2,cap);trace->rows.reserve(2);trace->rows.push_back({old_r,old_s,old_t,0,false});trace->rows.push_back({r,s,t,0,false});}
    while(r!=0){
        admit(trace,1,cap);const Wide q=old_r/r;
        const Wide next_r=old_r-q*r,next_s=old_s-q*s,next_t=old_t-q*t;
        ++out.work.divisions;out.work.coefficient_updates+=2;
        if(trace)trace->rows.push_back({next_r,next_s,next_t,q,true});
        old_r=r;r=next_r;old_s=s;s=next_s;old_t=t;t=next_t;
    }
    if(old_r<0){old_r=-old_r;old_s=-old_s;old_t=-old_t;out.work.sign_normalized=true;}
    out.certificate={static_cast<std::uint64_t>(old_r),coefficient(old_s),coefficient(old_t)};return out;
}
/// @brief Convert an actual unsigned-range certificate to the unchanged legacy result layout.
/// @param c Actual normalized native-domain certificate.
/// @return Exact three-signed64 result.
/// @throws std::overflow_error If nonnegative gcd2^63 cannot fit the legacy signed field.
ExtendedGcd legacy(const ExtendedGcdCertificate& c){if(c.gcd>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))throw std::overflow_error("extended_gcd: gcd exceeds signed int64; use extended_gcd_certificate");return {static_cast<std::int64_t>(c.gcd),c.x,c.y};}
/// @brief Validate a modular-inverse modulus before solve/history admission.
/// @param m Requested signed modulus.
/// @throws std::invalid_argument If m is nonpositive.
void modulus(std::int64_t m){if(m<=0)throw std::invalid_argument("modular_inverse: modulus must be positive");}
/// @brief Normalize an actual unit-game coefficient to its canonical inverse residue.
/// @param c Actual certificate for inputs (a,m), with m positive.
/// @param m Validated positive modulus.
/// @return Inverse in [0,m), including zero when m=1.
/// @throws std::invalid_argument If the actual gcd is not one.
std::int64_t inverse(const ExtendedGcdCertificate& c,std::int64_t m){
    if(c.gcd!=1)throw std::invalid_argument("modular_inverse: value and modulus are not coprime");
    auto result=c.x%m;if(result<0)result+=m;return result;
}
/// @brief Obtain a signed64 magnitude without signed negation, including INT64_MIN.
/// @param value Arbitrary signed64 input.
/// @return Exact nonnegative uint64 magnitude, at most2^63.
std::uint64_t magnitude(std::int64_t value) noexcept {const auto word=static_cast<std::uint64_t>(value);return value<0?std::uint64_t{0}-word:word;}
} // namespace euclid_detail
ExtendedGcd extended_gcd(std::int64_t a,std::int64_t b){return euclid_detail::legacy(euclid_detail::execute(a,b,nullptr,0).certificate);}
ExtendedGcdCertificate extended_gcd_certificate(std::int64_t a,std::int64_t b){return euclid_detail::execute(a,b,nullptr,0).certificate;}
EuclidOutcome extended_gcd_work(std::int64_t a,std::int64_t b){return euclid_detail::execute(a,b,nullptr,0);}
EuclidTrace extended_gcd_trace(std::int64_t a,std::int64_t b,std::size_t max_rows){EuclidTrace trace;trace.a=a;trace.b=b;trace.outcome=euclid_detail::execute(a,b,&trace,max_rows);return trace;}
bool bezout_certificate_valid(std::int64_t a,std::int64_t b,const ExtendedGcdCertificate& c) noexcept {
    if(!c.gcd)return a==0&&b==0;
    if(euclid_detail::magnitude(a)%c.gcd||euclid_detail::magnitude(b)%c.gcd)return false;
    const euclid_detail::Wide first=static_cast<euclid_detail::Wide>(a)*c.x,second=static_cast<euclid_detail::Wide>(b)*c.y;
    if((second>0&&first>std::numeric_limits<euclid_detail::Wide>::max()-second)||(second<0&&first<std::numeric_limits<euclid_detail::Wide>::min()-second))return false;
    return first+second==static_cast<euclid_detail::Wide>(c.gcd);
}
std::int64_t modular_inverse(std::int64_t a,std::int64_t m){euclid_detail::modulus(m);return euclid_detail::inverse(euclid_detail::execute(a,m,nullptr,0).certificate,m);}
ModularInverseTrace modular_inverse_trace(std::int64_t a,std::int64_t m,std::size_t max_rows){euclid_detail::modulus(m);ModularInverseTrace out;out.value=a;out.modulus=m;out.euclid=extended_gcd_trace(a,m,max_rows);out.inverse=euclid_detail::inverse(out.euclid.outcome.certificate,m);return out;}
} // namespace datamunge::algorithms
