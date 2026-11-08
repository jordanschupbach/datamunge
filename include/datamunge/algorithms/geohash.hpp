#pragma once

/// @file geohash.hpp
/// @brief Ordered double geohash encoding, actual bit traces and checked angular-cell analysis.
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned latitude/longitude bounds in degrees, initially the whole world rectangle.
/// These are actual ordered double-bisection endpoints, not certified exact-real
/// bounds at arbitrary precision. Long runs can stagnate or collapse an interval.
/// Longitude is not normalized across the date line. No shared state or mutation.
/// @par Example
/// @code
/// const auto c=dm::algorithms::geohash_decode("e");
/// std::cout << c.bounds.lon_lo << ' ' << c.bounds.lat_hi; // -45 45
/// @endcode
/// @see GeohashCell
struct GeohashBounds {
    double lat_lo{-90}; ///< Computed southern endpoint in degrees.
    double lat_hi{90}; ///< Computed northern endpoint in degrees.
    double lon_lo{-180}; ///< Computed western endpoint in degrees.
    double lon_hi{180}; ///< Computed eastern endpoint in degrees.
};

/// @brief Actual bit and character operations from one ordered encoder execution.
/// Counts include all requested steps even after interval stagnation. They are
/// selected operations, not timings or total floating-operation counts. Allocation,
/// copying, validation and trace capture are excluded.
/// @par Example
/// @code
/// const auto r=dm::algorithms::geohash_encode_run(42.6,-5.6,1);
/// std::cout << r.work.bisections << ' ' << r.work.longitude_steps; // 5 3
/// @endcode
/// @see GeohashRun
struct GeohashWork {
    std::size_t bisections{}; ///< Actual midpoint computations, five per emitted character.
    std::size_t longitude_steps{}; ///< Longitude bisections, starting with the first bit.
    std::size_t latitude_steps{}; ///< Latitude bisections, alternating with longitude.
    std::size_t emissions{}; ///< Actual appended base32 characters.
    std::size_t unchanged_intervals{}; ///< Steps whose selected axis endpoints remain unchanged.
    std::size_t collapsed_intervals{}; ///< Steps ending with equal selected-axis endpoints, including repeated equality.
};

/// @brief Owned encoded key, actual final bounds, work and first precision-limit flags.
/// first_unchanged_bit/first_collapsed_bit are one-based actual bit indices, zero
/// if absent. A repeated interval need not have zero width; these flags do not
/// claim additional geographic accuracy, true convergence or unique ownership of
/// a shared boundary. Ordinary result needs O(p) output bytes, O(1) other storage.
/// @par Example
/// @code
/// const auto r=dm::algorithms::geohash_encode_run(42.6,-5.6,1);
/// std::cout << r.hash << ' ' << r.bounds.lon_lo; // e -45
/// @endcode
/// @see geohash_encode_trace()
struct GeohashRun {
    std::string hash; ///< Owned base32 key in the legacy alphabet.
    GeohashBounds bounds; ///< Actual final double endpoints.
    GeohashWork work; ///< Selected counts from this execution; no replay.
    std::size_t first_unchanged_bit{}; ///< First unchanged selected interval, zero if none.
    std::size_t first_collapsed_bit{}; ///< First equal-endpoint selected interval, zero if none.
};

/// @brief One actual coordinate comparison and axis-endpoint update.
/// bit_in_symbol is zero-based 0..4 before the group count increments. symbol_value
/// is the actual accumulator after shifting in this bit. longitude=true selects
/// the longitude interval. Exact equality to midpoint chooses upper. No input is
/// borrowed and no event is reconstructed from a final key.
/// @par Example
/// @code
/// const auto t=dm::algorithms::geohash_encode_trace(42.6,-5.6,1);
/// std::cout << t.bits[0].longitude << ' ' << t.bits[0].upper; // 1 0
/// @endcode
/// @see GeohashEmission
struct GeohashBit {
    std::size_t step{}; ///< One-based actual bit number.
    bool longitude{}; ///< Whether this update used longitude rather than latitude.
    double lower_before{}; ///< Actual selected lower endpoint before the update.
    double upper_before{}; ///< Actual selected upper endpoint before the update.
    double midpoint{}; ///< Actual (lower+upper)/2 double used by the comparison.
    bool upper{}; ///< Actual coordinate>=midpoint decision.
    double lower_after{}; ///< Actual selected lower endpoint after the update.
    double upper_after{}; ///< Actual selected upper endpoint after the update.
    int symbol_value{}; ///< Actual partial five-bit accumulator after this bit.
    int bit_in_symbol{}; ///< Zero-based position in this five-bit group, before count increment.
};

/// @brief One actual base32 emission with endpoints at that control-flow point.
/// Records the appended character before the executing kernel resets its group
/// counter/accumulator. Owns its bounds; no prefix string is copied per record.
/// @par Example
/// @code
/// const auto t=dm::algorithms::geohash_encode_trace(42.6,-5.6,1);
/// std::cout << t.emissions[0].symbol << ' ' << t.emissions[0].symbol_value; // e 13
/// @endcode
/// @see GeohashTrace
struct GeohashEmission {
    std::size_t bit_step{}; ///< Actual bit number that completed the symbol.
    std::size_t characters{}; ///< Actual output length immediately after appending.
    int symbol_value{}; ///< Completed five-bit alphabet index 0..31.
    char symbol{}; ///< Actual appended lowercase alphabet character.
    GeohashBounds bounds; ///< Actual bounds immediately after this emission.
};

/// @brief Owned initial-coordinate record, actual bit/emission records and final ordinary run.
/// captured_records counts 1 initial fixed-schema coordinate/bounds record, one
/// per GeohashBit and one per GeohashEmission. These are records, not numeric
/// entries or bytes. Root/work/final ordinary output, working state and allocator
/// storage are excluded; this is not a peak RAM limit. Vectors are independent.
/// @par Example
/// @code
/// const auto t=dm::algorithms::geohash_encode_trace(42.6,-5.6,1,7);
/// std::cout << t.captured_records << ' ' << t.emissions.size(); // 7 1
/// @endcode
/// @see geohash_trace_records()
struct GeohashTrace {
    double latitude{}; ///< Owned validated initial coordinate in degrees.
    double longitude{}; ///< Owned validated initial coordinate in degrees.
    GeohashBounds initial_bounds; ///< Owned whole-world endpoints before any bit.
    GeohashRun run; ///< Final ordinary output/work; excluded from record accounting.
    std::vector<GeohashBit> bits; ///< Actual ordered bisection/comparison updates.
    std::vector<GeohashEmission> emissions; ///< Actual character append events.
    std::size_t captured_records{}; ///< Actual initial-plus-bit-plus-emission record count.
};

/// @brief Owned decoded double cell bounds, unrounded center and actual angular spans.
/// latitude_bits/longitude_bits count decoded bit positions, not significant
/// coordinate bits. limited is true if any update stagnated or ended collapsed.
/// Long keys may have degenerate/unchanged bounds; no physical-distance or exact
/// real-cell certificate is implied. Center is (lo+hi)/2 without decimal rounding.
/// @par Example
/// @code
/// const auto c=dm::algorithms::geohash_decode("e");
/// std::cout << c.longitude << ' ' << c.latitude << ' ' << c.longitude_span; // -22.5 22.5 45
/// @endcode
/// @see geohash_resolution()
struct GeohashCell {
    GeohashBounds bounds; ///< Actual ordered double endpoints in degrees.
    double latitude{}; ///< Computed latitude center, degrees.
    double longitude{}; ///< Computed longitude center, degrees.
    double latitude_span{180}; ///< Actual lat_hi-lat_lo, possibly zero.
    double longitude_span{360}; ///< Actual lon_hi-lon_lo, possibly zero.
    std::size_t latitude_bits{}; ///< Number of latitude positions consumed.
    std::size_t longitude_bits{}; ///< Number of longitude positions consumed.
    bool limited{}; ///< Any interval stagnation/collapse occurred while decoding.
};

/// @brief Ideal dyadic angular resolution rounded to double, separate from actual endpoint arithmetic.
/// Ideal widths use 360*2^-longitude_bits and 180*2^-latitude_bits. At high counts
/// these rounded spans can underflow to zero; spans_positive reports both positive.
/// This is neither a metric-distance estimate nor a guarantee the actual encoder
/// endpoints retain that width beyond floating precision. No allocation or mutation.
/// @par Example
/// @code
/// const auto r=dm::algorithms::geohash_resolution(2);
/// std::cout << r.longitude_bits << ' ' << r.latitude_bits << ' ' << r.longitude_degrees; // 5 5 11.25
/// @endcode
/// @see GeohashCell
struct GeohashResolution {
    std::size_t longitude_bits{}; ///< ceil(5p/2) ideal longitude positions.
    std::size_t latitude_bits{}; ///< floor(5p/2) ideal latitude positions.
    double longitude_degrees{360}; ///< Rounded ideal full longitude span in degrees.
    double latitude_degrees{180}; ///< Rounded ideal full latitude span in degrees.
    bool spans_positive{true}; ///< Whether both rounded ideal spans remain positive.
};

/// @brief Encode a finite latitude/longitude pair using the legacy ordered double bisections.
/// Starts with longitude, takes upper on equality and appends the alphabet
/// 0123456789bcdefghjkmnpqrstuvwxyz after every five bits. Coordinates include
/// poles/end-meridians; +180 and -180 are not normalized to one key. Finite valid
/// negative precision retains legacy zero-character compatibility, including
/// INT_MIN; zero yields empty. Coordinates are validated even for zero/negative
/// precision. No arbitrary positive precision cap or early stop is imposed;
/// high-count intervals can stagnate/collapse while output continues. O(p) time
/// and owned output bytes, O(1) additional working state; no caller/global mutation.
/// @param lat Finite latitude in [-90,90] degrees.
/// @param lon Finite longitude in [-180,180] degrees.
/// @param precision Requested character count; negative means zero for legacy compatibility.
/// @return Owned lowercase geohash string, empty at nonpositive precision.
/// @throws std::invalid_argument Nonfinite or out-of-range coordinates, before work.
/// @throws std::length_error Actual work or string capacity cannot fit.
/// @par Example
/// @code
/// std::cout << dm::algorithms::geohash_encode(42.6,-5.6,5); // ezs42
/// @endcode
/// @par References
/// - Chris Veness, Geohash encoding/decoding,
///   https://www.movable-type.co.uk/scripts/geohash.html , midpoint encoder/alphabet;
///   this variant preserves required precision, inclusive endpoints and negative-count compatibility.
/// @see geohash_encode_run(), geohash_decode()
std::string geohash_encode(double lat,double lon,int precision);

/// @brief Execute the same encoder while returning actual bounds, counts and precision-limit flags.
/// Same arithmetic/output as geohash_encode for nonnegative precision. Checked
/// analysis rejects negative counts rather than applying the legacy clamp. No
/// trace allocation or numerical replay, no early stop after stagnation.
/// @param lat Finite latitude in [-90,90] degrees.
/// @param lon Finite longitude in [-180,180] degrees.
/// @param precision Nonnegative character count.
/// @return Owned encoded string, actual bounds/work and first unchanged/collapsed bit flags.
/// @throws std::invalid_argument Invalid coordinates or negative count, before work.
/// @throws std::length_error Actual work/string capacity cannot fit.
/// @par Example
/// @code
/// const auto r=dm::algorithms::geohash_encode_run(42.6,-5.6,1);
/// std::cout << r.hash << ' ' << r.work.emissions; // e 1
/// @endcode
/// @see geohash_encode_trace()
GeohashRun geohash_encode_run(double lat,double lon,int precision);

/// @brief Count complete initial, bit and emission records without executing or allocating.
/// Bound and actual count are 1+6p, including the initial coordinate/bounds record
/// even when p=0. Does not include final ordinary run/work or working storage.
/// @param precision Nonnegative number of output characters.
/// @return Representable complete fixed-schema record count, not bytes or numeric entries.
/// @throws std::invalid_argument Negative count.
/// @throws std::length_error Whole count cannot fit size_t.
/// @par Example
/// @code
/// std::cout << dm::algorithms::geohash_trace_records(1); // 7
/// @endcode
/// @see geohash_encode_trace()
std::size_t geohash_trace_records(int precision);

/// @brief Capture actual encoder bisections and character emissions from the shared kernel.
/// Validates coordinates/count before quota. Checks the whole 1+6p record bound,
/// bit/emission vector capacities and positive quota before initialization/history
/// allocation. Zero disables only optional quota; no early-stagnation discount,
/// truncation, partial returned trace, numerical replay or caller mutation.
/// @param lat Finite latitude in [-90,90] degrees.
/// @param lon Finite longitude in [-180,180] degrees.
/// @param precision Nonnegative requested character count.
/// @param max_records Positive whole-record cap, default 1000000; zero disables quota.
/// @return Owned initial record, actual bit/emission vectors and final ordinary run.
/// @throws std::invalid_argument Invalid coordinate or negative count, before quota.
/// @throws std::length_error Whole count/capacity/quota or actual work/string capacity cannot fit.
/// @par Example
/// @code
/// const auto t=dm::algorithms::geohash_encode_trace(42.6,-5.6,1,7);
/// std::cout << t.bits.size() << ' ' << t.run.hash; // 5 e
/// @endcode
/// @see GeohashTrace, geohash_trace_records()
GeohashTrace geohash_encode_trace(double lat,double lon,int precision,std::size_t max_records=1000000);

/// @brief Decode a strict lowercase geohash by consuming longitude-first symbol bits.
/// Validates the entire alphabet before arithmetic. Empty denotes world bounds,
/// center(0,0) and zero consumed bits. Does not lowercase or decimal-round the
/// center; any uppercase, embedded NUL or non-alphabet byte is invalid. O(p) work,
/// O(1) owned result/working storage. High counts expose limited endpoint precision;
/// center re-encoding need not round-trip a degenerate high-precision key.
/// @param hash Complete key using exactly 0123456789bcdefghjkmnpqrstuvwxyz, or empty.
/// @return Owned actual bounds/center/spans, consumed bit counts and limited flag.
/// @throws std::invalid_argument Any invalid byte, checked before numeric work.
/// @throws std::length_error Complete bit count cannot fit size_t.
/// @par Example
/// @code
/// const auto c=dm::algorithms::geohash_decode("ezs42");
/// std::cout << c.longitude_bits << ' ' << c.latitude_bits; // 13 12
/// @endcode
/// @par References
/// - Veness, https://www.movable-type.co.uk/scripts/geohash.html , bounds decoder;
///   unlike that implementation, keys here are strict lowercase, empty means world,
///   and centers are not rounded to decimal text.
/// @see GeohashCell
GeohashCell geohash_decode(std::string_view hash);

/// @brief Return ideal angular bit counts and dyadic spans without running the encoder.
/// Longitude consumes ceil(5p/2) bits, latitude floor(5p/2). Each added character
/// divides angular area by 32; individual axes alternate factors 4 and 8. Spans
/// rounded to double may underflow to zero, exposed by spans_positive. O(1) work.
/// @param precision Nonnegative number of characters; zero is the whole world.
/// @return Owned ideal angular resolution, not a physical-distance certificate.
/// @throws std::invalid_argument Negative precision.
/// @throws std::length_error Five-times-precision cannot fit size_t.
/// @par Example
/// @code
/// const auto r=dm::algorithms::geohash_resolution(1);
/// std::cout << r.longitude_degrees << ' ' << r.latitude_degrees; // 45 45
/// @endcode
/// @see geohash_decode()
GeohashResolution geohash_resolution(int precision);

/// @brief Count the common leading characters of two fully validated geohash keys.
/// Validates both complete alphabets before comparison, even if their first
/// characters differ. Different lengths/empty strings are safe. Prefix agreement
/// identifies nested angular cells; proximity does not imply prefix agreement.
/// O(a.size()+b.size()) work, O(1) extra storage, no mutation or copies.
/// @param a First strict lowercase key or empty.
/// @param b Second strict lowercase key or empty.
/// @return Common prefix length, at most the smaller input length.
/// @throws std::invalid_argument Any invalid byte in either entire input.
/// @par Example
/// @code
/// std::cout << dm::algorithms::geohash_common_prefix("gcpvj0d","gcpuzgq"); // 3
/// @endcode
/// @see geohash_decode()
std::size_t geohash_common_prefix(std::string_view a,std::string_view b);

} // namespace datamunge::algorithms
