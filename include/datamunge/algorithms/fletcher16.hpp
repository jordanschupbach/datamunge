/// @file
/// @brief Canonical raw Fletcher-16 residues, actual folds and complete bounded corruption scans.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

/// @brief Canonical numerical and combinatorial algorithms.
namespace datamunge::algorithms {
/// @brief One actual modulo-255 byte fold from the production Fletcher-16 kernel.
/// Residues use the canonical representation 0..254; byte 255 contributes zero
/// modulo 255. Before/after values describe the actual sequential fold, not replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fletcher16_trace(std::string("\1\2",2));
/// std::cout << t.steps[1].after_a << ' ' << t.steps[1].after_b; // 3 4
/// @endcode
/// @see Fletcher16Trace, fletcher16_trace()
struct Fletcher16ByteStep {
    std::size_t index=0; ///< Zero-based byte position in the supplied view.
    std::uint8_t byte=0; ///< Actual unsigned input octet, including zero/255.
    std::uint16_t before_a=0; ///< First residue immediately before this byte.
    std::uint16_t before_b=0; ///< Second residue immediately before this byte.
    std::uint16_t after_a=0; ///< (before_a+byte) modulo 255.
    std::uint16_t after_b=0; ///< (before_b+after_a) modulo 255.
};

/// @brief Owned complete actual byte states for zero-initialised raw Fletcher-16.
/// No input view survives. Empty input has checksum/residues zero and no events.
/// The packed raw tag is B in the high byte and A in the low byte; it is not a
/// protocol-specific pair of complemented check bytes.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fletcher16_trace("abcde");
/// std::cout << t.sum1 << ' ' << t.sum2 << ' ' << t.steps.size(); // 240 200 5
/// @endcode
/// @see fletcher16_trace(), Fletcher16ByteStep
struct Fletcher16Trace {
    std::uint16_t checksum=0; ///< Canonical raw packed (B<<8)|A tag.
    std::uint16_t sum1=0; ///< Final A residue in [0,254].
    std::uint16_t sum2=0; ///< Final B residue in [0,254].
    std::vector<Fletcher16ByteStep> steps; ///< Owned actual folds in input order.
};

/// @brief One independent single-input-bit corruption, recomputed by the same full-message fold.
/// Bit numbering is byte-major, least-significant-bit first within each octet.
/// delta_a and delta_b are modular differences, not XOR popcounts or probabilities.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fletcher16_flip_scan("A");
/// std::cout << c.flips[0].bit << ' ' << c.flips[0].detected; // 0 1
/// @endcode
/// @see Fletcher16FlipScan
struct Fletcher16Flip {
    std::size_t bit=0; ///< Original input-bit address under the byte-major LSB-first convention.
    std::uint16_t checksum=0; ///< Recomputed canonical raw tag after that one bit toggle.
    std::uint16_t delta_a=0; ///< (changed A-original A) modulo 255, in [0,254].
    std::uint16_t delta_b=0; ///< (changed B-original B) modulo 255, in [0,254].
    bool detected=false; ///< Whether the complete raw checksum differs from baseline.
};

/// @brief Owned complete one-bit census and its exact full-message byte-fold budget.
/// Every candidate toggles/restores one owned scratch message. The original view
/// is unchanged and never retained. Empty input has zero records and counters.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fletcher16_flip_scan("abracadabra");
/// std::cout << c.flips.size() << ' ' << c.detected << ' ' << c.bytes_processed; // 88 88 979
/// @endcode
/// @see fletcher16_flip_scan(), Fletcher16Flip
struct Fletcher16FlipScan {
    std::uint16_t checksum=0; ///< Original raw checksum from the production fold.
    std::size_t bytes_processed=0; ///< Complete N*(8N+1) byte folds, including the baseline.
    std::size_t detected=0; ///< Number of independently changed tags.
    std::vector<Fletcher16Flip> flips; ///< Owned records in increasing input-bit order.
};

/// @brief One replacement candidate at a fixed byte position, including the unchanged control.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fletcher16_replacement_scan(std::string(1,'\0'),0);
/// std::cout << c.replacements[255].changed << ' ' << c.replacements[255].detected; // 1 0
/// @endcode
/// @see Fletcher16ReplacementScan
struct Fletcher16Replacement {
    std::uint8_t replacement=0; ///< Candidate octet value, in [0,255].
    std::uint16_t checksum=0; ///< Complete recomputed raw tag for this replacement.
    bool changed=false; ///< Candidate differs from the actual original byte.
    bool detected=false; ///< Candidate checksum differs from baseline.
};

/// @brief Owned exact 256-candidate single-position replacement census.
/// There are 255 changed candidates and one unchanged control. A zero/255
/// original has one changed candidate that aliases; other originals have none.
/// Records are complete actual folds, not an assumed detection theorem.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fletcher16_replacement_scan(std::string(1,'\0'),0);
/// std::cout << c.changed << ' ' << c.detected << ' ' << c.missed; // 255 254 1
/// @endcode
/// @see fletcher16_replacement_scan(), Fletcher16Replacement
struct Fletcher16ReplacementScan {
    std::uint16_t checksum=0; ///< Original canonical raw tag.
    std::size_t index=0; ///< Validated target byte position.
    std::uint8_t original=0; ///< Actual unsigned original octet.
    std::size_t bytes_processed=0; ///< Exact 257*N byte folds, baseline plus all 256 candidates.
    std::size_t changed=0; ///< Number of candidates differing from the original byte, exactly 255.
    std::size_t detected=0; ///< Number of candidate checksums differing from baseline.
    std::size_t missed=0; ///< Changed candidates whose checksum still equals baseline.
    std::vector<Fletcher16Replacement> replacements; ///< Owned candidates in numeric byte order0..255.
};

/// @brief Compute the original zero-initialised raw Fletcher-16, packed as (B<<8)|A.
/// Processes every unsigned octet, including embedded zeros/255. Each byte updates
/// A=(A+x)%255 and B=(B+A)%255, using canonical residues 0..254; empty input returns0.
/// All single-input-bit errors change the raw tag when the original tag is held fixed.
/// It does not detect every single-byte substitution:0 and255 alias. Transpositions
/// can also alias. No authentication, collision probability or speed ranking follows.
/// O(n) time, O(1) storage, two modulo reductions per byte, no allocation/mutation,
/// retained input or shared mutable state. Preserves the existing const-string API.
/// @param data Valid binary string; embedded zeros do not terminate it.
/// @return Canonical raw 16-bit checksum of all supplied bytes.
/// @par Example
/// @code
/// std::cout << std::hex << dm::algorithms::fletcher16("abcde") << std::dec; // c8f0
/// @endcode
/// @par References
/// - J. G. Fletcher (1982), An Arithmetic Checksum for Serial Transmissions,
///   IEEE Transactions on Communications30(1),247-252,doi:10.1109/TCOM.1982.1095369.
/// - RFC926,appendices A.3/C.2,https://www.rfc-editor.org/rfc/rfc926.html :
///   modulo 255 arithmetic/error patterns; this API exposes raw residues, not
///   that protocol's complemented/offset-dependent check bytes.
/// @see fletcher16_update(), fletcher16_trace(), fletcher16_flip_scan()
std::uint16_t fletcher16(const std::string& data);

/// @brief Continue one canonical raw Fletcher-16 state over the next byte view.
/// A is the low byte and B the high byte; both must be in 0..254. Empty input returns
/// the same validated state. Shares the ordinary byte kernel, with two modulo 255
/// reductions per byte and no trace/allocation. O(n) time/O(1) storage, no retained
/// views, mutation or shared state. Arbitrary packed lane255 states are rejected.
/// @param checksum Canonical current raw checksum, zero for an empty prefix.
/// @param data Valid read-only bytes to append, including embedded zeros/255.
/// @return Canonical raw checksum of the prefix followed by data.
/// @throws std::invalid_argument If either initial checksum lane equals255.
/// @par Example
/// @code
/// const auto a=dm::algorithms::fletcher16_update(0,"ab");
/// std::cout << std::hex << dm::algorithms::fletcher16_update(a,"cde") << std::dec; // c8f0
/// @endcode
/// @par References
/// - J. G. Fletcher (1982), An Arithmetic Checksum for Serial Transmissions,
///   IEEE Transactions on Communications30(1),247-252,doi:10.1109/TCOM.1982.1095369.
/// - RFC926,appendices A.3/C.2,https://www.rfc-editor.org/rfc/rfc926.html :
///   modulo 255 arithmetic/error patterns; this API exposes raw residues, not
///   that protocol's complemented/offset-dependent check bytes.
/// @see fletcher16(), fletcher16_combine()
std::uint16_t fletcher16_update(std::uint16_t checksum,std::string_view data);

/// @brief Combine two zero-initialised raw tags without reading either message.
/// For a right message of length m, A=A1+A2 and B=B1+B2+m*A1, modulo 255. Reduces
/// m before multiplication, supporting every uint64 length. Both tags must have
/// canonical lanes 0..254. Their consistency with the claimed message/length is the
/// caller's responsibility. O(1) time/storage, no allocation/shared state.
/// @param left Canonical raw checksum of the left message.
/// @param right Canonical zero-initialised raw checksum of the right message.
/// @param right_length Right message length in bytes; uint64 values are supported.
/// @return Canonical raw tag of the concatenation under the documented message/length assumption.
/// @throws std::invalid_argument If either input tag contains a lane255.
/// @par Example
/// @code
/// const auto a=dm::algorithms::fletcher16("ab"),b=dm::algorithms::fletcher16("cde");
/// std::cout << std::hex << dm::algorithms::fletcher16_combine(a,b,3) << std::dec; // c8f0
/// @endcode
/// @par References
/// - J. G. Fletcher (1982), An Arithmetic Checksum for Serial Transmissions,
///   IEEE Transactions on Communications30(1),247-252,doi:10.1109/TCOM.1982.1095369.
/// - RFC926,appendices A.3/C.2,https://www.rfc-editor.org/rfc/rfc926.html :
///   modulo 255 arithmetic/error patterns; this API exposes raw residues, not
///   that protocol's complemented/offset-dependent check bytes.
/// @see fletcher16_update(), fletcher16()
std::uint16_t fletcher16_combine(std::uint16_t left,std::uint16_t right,std::uint64_t right_length);

/// @brief Capture every actual byte fold from the same zero-initialised production kernel.
/// Checks the whole n-event cap and vector capacity before allocation/folding. Each
/// record owns its byte and states; no source view or replay survives. O(n) time
/// and O(n) history entries, with checksum0/no events for empty input.
/// @param data Valid read-only binary byte view.
/// @param max_bytes Complete event cap,default 1000000;zero disables only this optional cap.
/// @return Owned actual input-byte states and canonical packed raw tag.
/// @throws std::length_error If the complete event cap or container capacity is exceeded.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fletcher16_trace(std::string("\1\2",2),2);
/// std::cout << t.sum1 << ' ' << t.sum2; // 3 4
/// @endcode
/// @par References
/// - J. G. Fletcher (1982), An Arithmetic Checksum for Serial Transmissions,
///   IEEE Transactions on Communications30(1),247-252,doi:10.1109/TCOM.1982.1095369.
/// - RFC926,appendices A.3/C.2,https://www.rfc-editor.org/rfc/rfc926.html :
///   modulo 255 arithmetic/error patterns; this API exposes raw residues, not
///   that protocol's complemented/offset-dependent check bytes.
/// @see Fletcher16Trace, fletcher16_update()
Fletcher16Trace fletcher16_trace(std::string_view data,std::size_t max_bytes=1'000'000);

/// @brief Census every independent one-input-bit toggle using complete production checksum folds.
/// Computes one baseline and 8n changed tags, each over n bytes; exact fold work is
/// n*(8n+1). One owned scratch message is toggled/restored, not copied per candidate.
/// Whole bit/work/capacity checks precede allocation or checksum evaluation. O(n²)
/// byte-fold work plus toggles, O(n) scratch/records; no partial/truncated scan.
/// Empty input has no records and zero counters. No input mutation/shared state.
/// @param data Valid read-only binary view, possibly empty.
/// @param max_bytes_processed Whole byte-fold cap,default 100000000;zero disables only this optional cap.
/// @return Owned baseline,all 8n actual changed tags and modular deltas.
/// @throws std::length_error If complete bit/work/capacity accounting or the nonzero cap is exceeded.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fletcher16_flip_scan("A",9);
/// std::cout << c.flips.size() << ' ' << c.detected << ' ' << c.bytes_processed; // 8 8 9
/// @endcode
/// @par References
/// - J. G. Fletcher (1982), An Arithmetic Checksum for Serial Transmissions,
///   IEEE Transactions on Communications30(1),247-252,doi:10.1109/TCOM.1982.1095369.
/// - RFC926,appendices A.3/C.2,https://www.rfc-editor.org/rfc/rfc926.html :
///   modulo 255 arithmetic/error patterns; this API exposes raw residues, not
///   that protocol's complemented/offset-dependent check bytes.
/// @see Fletcher16FlipScan, fletcher16_trace()
Fletcher16FlipScan fletcher16_flip_scan(std::string_view data,std::size_t max_bytes_processed=100'000'000);

/// @brief Recompute all 256 possible octet replacements at one validated input position.
/// Includes one unchanged control and255 changed candidates; uses one owned scratch
/// message and the ordinary complete checksum fold for each. Checks exact 257*n
/// byte work, capacities and optional whole cap before allocation/evaluation.
/// O(257n) time/O(n+256) output/scratch entries, no input mutation or retained view.
/// @param data Valid nonempty binary view.
/// @param index Existing zero-based byte position to replace.
/// @param max_bytes_processed Whole fold cap,default 100000000;zero disables only this optional cap.
/// @return Owned complete candidate tags,changed/detected/missed counts and original byte.
/// @throws std::invalid_argument If index lies outside data.
/// @throws std::length_error If complete work/capacities or the nonzero cap is exceeded.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto c=dm::algorithms::fletcher16_replacement_scan(std::string(1,'\0'),0,257);
/// std::cout << c.replacements.size() << ' ' << c.changed << ' ' << c.detected; // 256 255 254
/// @endcode
/// @par References
/// - J. G. Fletcher (1982), An Arithmetic Checksum for Serial Transmissions,
///   IEEE Transactions on Communications30(1),247-252,doi:10.1109/TCOM.1982.1095369.
/// - RFC926,appendices A.3/C.2,https://www.rfc-editor.org/rfc/rfc926.html :
///   modulo 255 arithmetic/error patterns; this API exposes raw residues, not
///   that protocol's complemented/offset-dependent check bytes.
/// @see Fletcher16ReplacementScan, fletcher16_flip_scan()
Fletcher16ReplacementScan fletcher16_replacement_scan(std::string_view data,std::size_t index,std::size_t max_bytes_processed=100'000'000);

/// @brief Predict the modular second-residue change for exchanging two positions at a given distance.
/// If the left byte a is replaced by b and the right b by a, deltaA=0 and
///  deltaB=distance*(b-a) modulo 255. Uses reduced factors, supporting uint64MAX.
/// Returns the exact arithmetic formula, not an actual message fold or probability.
/// Zero distance or equal/congruent bytes gives zero. O(1), no allocation/state.
/// @param distance Nonnegative separation between the two positions.
/// @param left Original left octet, including0/255.
/// @param right Original right octet, including0/255.
/// @return Canonical deltaB in 0..254;zero means this transposition leaves the raw tag unchanged.
/// @par Example
/// @code
/// std::cout << dm::algorithms::fletcher16_transposition_delta(3,1,86); // 0
/// @endcode
/// @par References
/// - J. G. Fletcher (1982), An Arithmetic Checksum for Serial Transmissions,
///   IEEE Transactions on Communications30(1),247-252,doi:10.1109/TCOM.1982.1095369.
/// - RFC926,appendices A.3/C.2,https://www.rfc-editor.org/rfc/rfc926.html :
///   modulo 255 arithmetic/error patterns; this API exposes raw residues, not
///   that protocol's complemented/offset-dependent check bytes.
/// @see checksum_swap_bytes(), fletcher16()
std::uint16_t fletcher16_transposition_delta(std::uint64_t distance,std::uint8_t left,std::uint8_t right);

/// @brief Compute the original report's plain unsigned32-bit byte sum, with intentional modulo 2^32 wrap.
/// Unsigned octets are added without positional weighting; reordering always keeps
/// this sum. Includes embedded zeros/high bytes. O(n) time/O(1) storage, no allocation,
/// input mutation or retained view. This is a checksum convention, not an exact
/// unbounded integer sum or authentication tag.
/// @param data Valid read-only binary byte view, possibly empty.
/// @return Sum of all unsigned input bytes modulo 2^32;zero for empty input.
/// @par Example
/// @code
/// std::cout << dm::algorithms::byte_sum32("ABCD"); // 266
/// @endcode
/// @see fletcher16(), checksum_swap_bytes()
std::uint32_t byte_sum32(std::string_view data);

/// @brief Swap two validated byte positions in an owned binary copy without changing the source.
/// Checks both indices before copying or mutation; equal valid indices return an
/// unchanged owned copy. Empty input has no valid index. O(n) copy work/storage,
/// constant swap work, arbitrary binary octets preserved and no retained view.
/// @param data Valid read-only binary byte view.
/// @param first Existing zero-based byte position.
/// @param second Existing zero-based byte position;may equal first.
/// @return Owned same-length message with precisely those byte values exchanged.
/// @throws std::invalid_argument If either index lies outside data.
/// @throws std::length_error If the copy cannot fit string capacity.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::cout << dm::algorithms::checksum_swap_bytes("ABCD",2,3); // ABDC
/// @endcode
/// @see byte_sum32(), fletcher16_transposition_delta()
std::string checksum_swap_bytes(std::string_view data,std::size_t first,std::size_t second);

} // namespace datamunge::algorithms
