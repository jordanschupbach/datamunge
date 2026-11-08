#pragma once
/// @file extended_euclidean.hpp
/// @brief Checked signed-input extended Euclid, unsigned gcd certificates and actual widened quotient/coefficient rows.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {

/// @brief Legacy signed gcd and Bezout coefficients, preserving the three-int64 field layout.
/// For returned results, gcd>=0 and the mathematical identity a*x+b*y=gcd holds.
/// Evaluate products in a wider type: int64 products can overflow even when the
/// identity is small. Actual legacy gcd(0,0) returns {0,1,0}; this corrects the old
/// comment claiming zero coefficients. A gcd of2^63 cannot fit this record and
/// extended_gcd() throws instead. Manually constructed records are not validated.
///
///
/// @par Example
/// @code
/// const auto g=dm::algorithms::extended_gcd(240,46);
/// std::print("{} {} {}\n",g.gcd,g.x,g.y); // 2 -9 47
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see extended_gcd(), ExtendedGcdCertificate
struct ExtendedGcd {
    std::int64_t gcd; ///< Nonnegative representable gcd for a returned result.
    std::int64_t x; ///< Actual canonical coefficient of the first input.
    std::int64_t y; ///< Actual canonical coefficient of the second input.
};

/// @brief Unsigned gcd and signed canonical Bezout coefficients for the entire int64 input range.
/// gcd is at most2^63, so INT64_MIN magnitudes are representable. Actual final
/// coefficients fit signed int64; intermediate/terminal-zero coefficients need wider
/// arithmetic. Mathematical a*x+b*y=gcd requires widened evaluation, not unchecked
/// int64 multiplication. The returned zero-pair certificate is {0,1,0}. Default or
/// manually assigned fields do not prove a solve; bezout_certificate_valid checks
/// both the identity and common-divisor conditions without overflowing.
///
///
/// @par Example
/// @code
/// const auto c=dm::algorithms::extended_gcd_certificate(std::numeric_limits<std::int64_t>::min(),0);
/// std::print("{} {} {}\n",c.gcd,c.x,c.y); // 9223372036854775808 -1 0
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see extended_gcd_certificate(), bezout_certificate_valid()
struct ExtendedGcdCertificate {
    std::uint64_t gcd=0; ///< Nonnegative gcd, including2^63 for three boundary input pairs.
    std::int64_t x=0; ///< Actual final normalized coefficient of a.
    std::int64_t y=0; ///< Actual final normalized coefficient of b.
};

/// @brief Actual Euclidean divisions, coefficient updates and final normalization flag.
/// Each quotient/remainder iteration counts one division and two coefficient
/// updates, including an initial zero quotient when input magnitudes are reversed
/// and the terminal zero-remainder update. sign_normalized records the actual final
/// three-value sign flip. Counts exclude validation, copies, allocation, CPU time
/// and arbitrary-precision bit cost; native input width bounds these counters.
///
///
/// @par Example
/// @code
/// const auto w=dm::algorithms::extended_gcd_work(240,46).work;
/// std::print("{} {} {}\n",w.divisions,w.coefficient_updates,w.sign_normalized); // 5 10 false
/// @endcode
///
/// @see extended_gcd_work(), EuclidTrace
struct EuclidWork {
    std::size_t divisions=0; ///< Actual quotient/remainder iterations.
    std::size_t coefficient_updates=0; ///< Actual two coefficient recurrences per division.
    bool sign_normalized=false; ///< True if the final raw gcd was negative and all three final values were flipped.
};

/// @brief Owned unsigned-range Bezout certificate and work from one actual kernel execution.
/// Ordinary/work calls keep only running scalar states and this result; they do
/// not allocate optional row history or apply a capture quota. Certificate follows
/// the same signed-truncation ordering as the legacy API, with exact widened
/// intermediates. No borrowed input, shared mutable state or explanatory replay.
///
///
/// @par Example
/// @code
/// const auto e=dm::algorithms::extended_gcd_work(240,46);
/// std::print("{} {}\n",e.certificate.gcd,e.work.divisions); // 2 5
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see extended_gcd_work(), EuclidTrace
struct EuclidOutcome {
    ExtendedGcdCertificate certificate; ///< Actual final sign-normalized unsigned-gcd certificate.
    EuclidWork work; ///< Actual division/update/normalization counts.
};

/// @brief One actual raw Euclidean remainder and its signed-wide Bezout row.
/// remainder=s*a+t*b holds mathematically for the trace inputs. Initial rows are
/// (a,1,0) and (b,0,1), with divided=false and quotient=0 unused. Each derived row
/// has divided=true and the actual truncation-toward-zero quotient that produced it.
/// The terminal zero row is retained. Signed compiler-supported128-bit fields allow
/// quotient2^63 and terminal coefficients beyond int64 without wrapping. These raw
/// rows are not rewritten by the separately reported final sign normalization.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::extended_gcd_trace(240,46);
/// const auto& r=t.rows.at(2);
/// std::print("{} {} {} {}\n",dm::algorithms::int128_text(r.quotient),dm::algorithms::int128_text(r.remainder),dm::algorithms::int128_text(r.s),dm::algorithms::int128_text(r.t)); // 5 10 1 -5
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see EuclidTrace, extended_gcd_trace()
struct EuclidRow {
    __int128 remainder=0; ///< Actual raw signed remainder, including initial values and terminal zero.
    __int128 s=0; ///< Actual raw coefficient of a.
    __int128 t=0; ///< Actual raw coefficient of b.
    __int128 quotient=0; ///< Actual signed truncating quotient for derived rows; unused zero for initial rows.
    bool divided=false; ///< False for the initial two rows, true for each actually computed next row.
};

/// @brief Owned actual raw rows and unsigned-range outcome from the shared extended-Euclid loop.
/// For D actual divisions, rows/captured_entries=2+D WHOLE ROW RECORDS, including
/// two initial rows and the terminal zero row when a division occurs. Input scalars,
/// result/work metadata, vector capacity, runtime and allocator bytes are excluded;
/// this is not peak RAM. Initial two records are quota-checked before allocation;
/// each next record is admitted before its division and append. Default max_rows
/// one million; zero disables only quota. Arithmetic/container checks remain.
/// Exceptions expose no partial trace. Every returned container owns its values;
/// ordinary calls retain no optional history. Final sign normalization lives in
/// outcome, while raw rows preserve the actual signed-truncation computation.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::extended_gcd_trace(240,46,7);
/// std::print("{} {} {}\n",t.rows.size(),t.captured_entries,t.outcome.certificate.x); // 7 7 -9
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see extended_gcd_trace(), EuclidRow
struct EuclidTrace {
    std::int64_t a=0; ///< Original first signed input.
    std::int64_t b=0; ///< Original second signed input.
    EuclidOutcome outcome; ///< Actual normalized certificate and work.
    std::vector<EuclidRow> rows; ///< Owned raw initial/derived rows, in execution order.
    std::size_t captured_entries=0; ///< Defined whole-row record count, exactly rows.size() for a returned trace.
};

/// @brief Canonical modular inverse and the actual same-pass Euclid certificate history.
/// value and positive modulus are the original inputs; inverse lies in [0,modulus).
/// euclid captures the actual computation on (value,modulus), followed by reduction
/// of its coefficient of value. No explanatory rerun occurs. Modulus one returns
/// zero, retaining the original degenerate-residue convention. Nonunits or invalid
/// moduli throw instead of returning this record. All fields own their values.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::modular_inverse_trace(3,11);
/// std::print("{} {} {}\n",t.inverse,t.euclid.outcome.certificate.x,t.euclid.outcome.work.divisions); // 4 4 4
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see modular_inverse_trace(), modular_inverse()
struct ModularInverseTrace {
    std::int64_t value=0; ///< Original signed value being inverted.
    std::int64_t modulus=1; ///< Original strictly positive modulus.
    std::int64_t inverse=0; ///< Actual canonical residue obtained from the Bezout coefficient.
    EuclidTrace euclid; ///< Owned actual same-execution row path and normalized certificate.
};

/// @brief Return the legacy signed result through checked shared extended-Euclid execution.
/// Retains signed C++ truncation-toward-zero quotients, initial input ordering and
/// final sign normalization. Shared signed128-bit intermediates avoid INT64_MIN/-1,
/// negation and terminal-coefficient overflow. Input values are unchanged, no mutable
/// global state is shared, and independent calls are safe. For nonzero magnitudes,
/// O(1+log(min(|a|,|b|)+1)) fixed-word divisions and constant auxiliary storage;
/// zero cases take constant work. This is not an arbitrary-precision bit-cost claim.
/// Preserves the compiled signature and three-int64 result layout. The actual
/// zero-pair result is {0,1,0}. The three pairs (INT64_MIN,0), (0,INT64_MIN) and
/// (INT64_MIN,INT64_MIN) have gcd2^63 and cannot fit this legacy record. Use the
/// unsigned certificate/trace surface for them. No optional history/quota is kept.
///
/// @param a First arbitrary signed int64 input, including INT64_MIN.
/// @param b Second arbitrary signed int64 input, including INT64_MIN.
/// @return Owned signed gcd and actual canonical Bezout coefficients.
/// @throws std::overflow_error If the gcd or a checked final coefficient cannot fit signed int64.
///
/// @par Example
/// @code
/// const auto g=dm::algorithms::extended_gcd(-240,46);
/// std::print("{} {} {}\n",g.gcd,g.x,g.y); // 2 9 47
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see extended_gcd_certificate(), extended_gcd_trace()
ExtendedGcd extended_gcd(std::int64_t a,std::int64_t b);

/// @brief Return an unsigned-gcd certificate without retaining optional row history.
/// Retains signed C++ truncation-toward-zero quotients, initial input ordering and
/// final sign normalization. Shared signed128-bit intermediates avoid INT64_MIN/-1,
/// negation and terminal-coefficient overflow. Input values are unchanged, no mutable
/// global state is shared, and independent calls are safe. For nonzero magnitudes,
/// O(1+log(min(|a|,|b|)+1)) fixed-word divisions and constant auxiliary storage;
/// zero cases take constant work. This is not an arbitrary-precision bit-cost claim.
/// Supports a gcd of2^63 as well as zero. Final coefficients retain the legacy
/// canonical signed ordering; all native-domain final coefficients fit int64 and
/// are checked before narrowing. No allocation, history quota or input mutation.
///
/// @param a First arbitrary signed int64 input, including INT64_MIN.
/// @param b Second arbitrary signed int64 input, including INT64_MIN.
/// @return Owned uint64 gcd and signed int64 Bezout coefficients.
/// @throws std::overflow_error On a checked final coefficient-range failure.
///
/// @par Example
/// @code
/// const auto c=dm::algorithms::extended_gcd_certificate(0,0);
/// std::print("{} {} {}\n",c.gcd,c.x,c.y); // 0 1 0
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see ExtendedGcdCertificate, bezout_certificate_valid(), extended_gcd_work()
ExtendedGcdCertificate extended_gcd_certificate(std::int64_t a,std::int64_t b);

/// @brief Return the unsigned-range certificate and actual shared-loop work without history.
/// Retains signed C++ truncation-toward-zero quotients, initial input ordering and
/// final sign normalization. Shared signed128-bit intermediates avoid INT64_MIN/-1,
/// negation and terminal-coefficient overflow. Input values are unchanged, no mutable
/// global state is shared, and independent calls are safe. For nonzero magnitudes,
/// O(1+log(min(|a|,|b|)+1)) fixed-word divisions and constant auxiliary storage;
/// zero cases take constant work. This is not an arbitrary-precision bit-cost claim.
/// No allocation or optional capture limit. Counts include terminal coefficient
/// updates and any initial zero quotient; the final sign flag reflects the actual
/// raw gcd. The original (0,0) canonical coefficients remain one and zero.
///
/// @param a First arbitrary signed int64 input, including INT64_MIN.
/// @param b Second arbitrary signed int64 input, including INT64_MIN.
/// @return Owned actual unsigned-gcd certificate and work.
/// @throws std::overflow_error On a checked final coefficient-range failure.
///
/// @par Example
/// @code
/// const auto e=dm::algorithms::extended_gcd_work(46,240);
/// std::print("{} {} {}\n",e.certificate.gcd,e.work.divisions,e.work.coefficient_updates); // 2 6 12
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see EuclidOutcome, extended_gcd_trace()
EuclidOutcome extended_gcd_work(std::int64_t a,std::int64_t b);

/// @brief Capture actual initial and quotient-produced rows during the same Euclid execution.
/// Retains signed C++ truncation-toward-zero quotients, initial input ordering and
/// final sign normalization. Shared signed128-bit intermediates avoid INT64_MIN/-1,
/// negation and terminal-coefficient overflow. Input values are unchanged, no mutable
/// global state is shared, and independent calls are safe. For nonzero magnitudes,
/// O(1+log(min(|a|,|b|)+1)) fixed-word divisions and constant auxiliary storage;
/// zero cases take constant work. This is not an arbitrary-precision bit-cost claim.
/// The unsigned result supports every int64 pair. Raw signed128-bit rows preserve
/// the terminal zero and potentially2^63 quotients/coefficients. Defined logical
/// capture C=2+D follows EuclidTrace; quota admission precedes each retained row's
/// allocation/arithmetic. O(D) owned row storage, no replay or partial result.
///
/// @param a First arbitrary signed int64 input, including INT64_MIN.
/// @param b Second arbitrary signed int64 input, including INT64_MIN.
/// @param max_rows Maximum whole retained row records, default1000000; zero disables only quota.
/// @return Owned original inputs, actual raw rows, normalized certificate and work.
/// @throws std::length_error If initial/next rows exceed quota or vector limits.
/// @throws std::overflow_error On a checked final coefficient-range failure.
/// @throws std::bad_alloc If optional row storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::extended_gcd_trace(240,46,7);
/// std::print("{} {}\n",t.outcome.work.divisions,dm::algorithms::int128_text(t.rows.back().t)); // 5 -120
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see EuclidTrace, modular_inverse_trace()
EuclidTrace extended_gcd_trace(std::int64_t a,std::int64_t b,std::size_t max_rows=1000000);

/// @brief Validate both the Bezout identity and greatest-common-divisor certificate safely.
/// For positive gcd, requires it to divide both unsigned input magnitudes and
/// requires widened a*x+b*y==gcd. These conditions prove it is the greatest common
/// divisor, not merely a linear combination. A zero gcd is valid exactly for two
/// zero inputs. Handles every manually assigned uint64 gcd and int64 coefficient,
/// including malformed values whose product sum would overflow signed128: returns
/// false before that addition. No allocation, mutation, shared state or exceptions.
///
/// @param a First arbitrary signed int64 input, including INT64_MIN.
/// @param b Second arbitrary signed int64 input, including INT64_MIN.
/// @param certificate Borrowed claimed nonnegative gcd and signed coefficients; need not be generated by the solver.
/// @return True exactly when the claimed common divisor and identity certify the gcd.
///
/// @par Example
/// @code
/// const auto c=dm::algorithms::extended_gcd_certificate(240,46);
/// std::print("{} {}\n",dm::algorithms::bezout_certificate_valid(240,46,c),dm::algorithms::bezout_certificate_valid(240,46,{0,0,0})); // true false
/// @endcode
///
/// @see ExtendedGcdCertificate, extended_gcd_certificate()
bool bezout_certificate_valid(std::int64_t a,std::int64_t b,const ExtendedGcdCertificate& certificate) noexcept;

/// @brief Compute the legacy canonical modular inverse using safe widened Euclid arithmetic.
/// Requires m>0 and gcd(a,m)=1; arbitrary signed a, including INT64_MIN, is
/// accepted. Uses actual signed-input Euclid coefficients, then reduces x%m and
/// adds m only to a negative remainder, avoiding an unreduced x+m sum. Modulus one
/// returns zero by the legacy degenerate-residue convention (1 is congruent to0).
/// No cryptographic timing guarantee is supplied. Input values stay unchanged;
/// independent calls share no mutable state.
/// Ordinary execution retains no optional rows or capture quota and allocates no
/// history. The output residue lies in [0,m); no inverse exists for a nonunit.
///
/// @param a Any signed int64 value to invert.
/// @param m Strictly positive signed int64 modulus, including one.
/// @return Canonical inverse residue in [0,m).
/// @throws std::invalid_argument If m is nonpositive or a and m are not coprime.
/// @throws std::overflow_error On a checked final coefficient-range failure.
///
/// @par Example
/// @code
/// std::print("{} {} {}\n",dm::algorithms::modular_inverse(3,11),dm::algorithms::modular_inverse(-3,11),dm::algorithms::modular_inverse(0,1)); // 4 7 0
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see modular_inverse_trace(), extended_gcd_certificate()
std::int64_t modular_inverse(std::int64_t a,std::int64_t m);

/// @brief Return the canonical inverse with its actual same-pass widened Euclid row path.
/// Requires m>0 and gcd(a,m)=1; arbitrary signed a, including INT64_MIN, is
/// accepted. Uses actual signed-input Euclid coefficients, then reduces x%m and
/// adds m only to a negative remainder, avoiding an unreduced x+m sum. Modulus one
/// returns zero by the legacy degenerate-residue convention (1 is congruent to0).
/// No cryptographic timing guarantee is supplied. Input values stay unchanged;
/// independent calls share no mutable state.
/// Validates positive m before optional history admission. The shared kernel then
/// captures rows once; no explanatory second solve occurs. A nonunit throws when
/// its gcd is known, unless an earlier quota/allocation failure stops execution.
/// C=2+D and ownership follow EuclidTrace. No partial trace escapes an exception.
///
/// @param a Any signed int64 value to invert.
/// @param m Strictly positive signed int64 modulus, including one.
/// @param max_rows Maximum whole Euclid row records, default1000000; zero disables only quota.
/// @return Owned original inverse inputs, canonical residue and actual same-pass Euclid trace.
/// @throws std::invalid_argument If m is nonpositive or the reached gcd is not one.
/// @throws std::length_error If initial/next row capture exceeds quota or vector limits.
/// @throws std::overflow_error On a checked final coefficient-range failure.
/// @throws std::bad_alloc If optional history allocation fails.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::modular_inverse_trace(3,11,6);
/// std::print("{} {}\n",t.inverse,t.euclid.captured_entries); // 4 6
/// @endcode
///
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography, chapter2, algorithms2.107 and2.142. https://cacr.uwaterloo.ca/hac/about/chap2.pdf. This variant retains signed-input truncating quotients and final sign normalization, without initially sorting inputs.
///
/// @see ModularInverseTrace, extended_gcd_trace()
ModularInverseTrace modular_inverse_trace(std::int64_t a,std::int64_t m,std::size_t max_rows=1000000);
} // namespace datamunge::algorithms
