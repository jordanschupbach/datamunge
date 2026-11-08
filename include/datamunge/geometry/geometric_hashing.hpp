#pragma once
/// @file geometric_hashing.hpp
/// @brief Checked planar similarity hashing, actual work and owned primitive histories.
#include <datamunge/geometry/point2d.hpp>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::geometry {
/// @brief An owned quantized-bin index of ordered model bases.
///
/// Each bucket stores every contributing basis pair, including repeated votes from
/// multiple model points in the same bin. Build from one ordered model and supply
/// that SAME model to recognition; this public aggregate does not certify provenance.
/// Coordinates/quantization are validated on build, and all stored ids on recognition.
/// Copies own their bucket vectors. Concurrent const reads require no caller mutation.
///
/// @par Example
/// @code
/// const dm::geometry::GeoHashTable t{};
/// std::cout << t.quant << " " << t.bins.size(); // 0.1 0
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see build_geometric_hash(), recognize()
struct GeoHashTable {
    double quant{0.1}; ///< Positive invariant-coordinate bin width, default0.1.
    std::unordered_map<std::uint64_t, std::vector<std::pair<int,int>>> bins; ///< Packed signed bin to ordered model basis pairs, with multiplicity.
};
/// @brief A highest-vote candidate and its first-hit geometric verification.
///
/// Original field order/defaults are preserved. found means a vote-qualified
/// nondegenerate candidate transform was recovered; it does NOT require any minimum
/// inlier count. Each model point counts if ANY scene point is within tolerance;
/// several model points may match the same scene point. No one-to-one assignment,
/// least-squares refinement, alternate candidate retry or reflection is performed.
/// Tied maxima follow unordered-map iteration and are not portable canonical ties.
///
/// @par Example
/// @code
/// const dm::geometry::GeoMatch m{};
/// std::cout << m.found << " " << m.model_i; // 0 -1
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see recognize(), recognize_run()
struct GeoMatch {
    bool found{false}; ///< A vote-qualified candidate transform was recovered, not an inlier acceptance flag.
    int votes{0}; ///< Highest global tally; repeated bin contributions count separately.
    int model_i{-1}; ///< Selected ordered model basis, or minus1 before selection.
    int model_j{-1}; ///< Selected ordered model basis, or minus1 before selection.
    int scene_a{-1}; ///< Selected ordered scene basis, or minus1 before selection.
    int scene_b{-1}; ///< Selected ordered scene basis, or minus1 before selection.
    double scale{0}; ///< Recovered nonnegative scale and atan2 rotation in radians; default0.
    double angle{0}; ///< Recovered nonnegative scale and atan2 rotation in radians; default0.
    Point2D translation{0,0}; ///< Translation in scene coordinate units.
    int inliers{0}; ///< Model points with at least one scene point within the squared tolerance.
};
/// @brief One actual double quotient in an ordered two-point frame.
///
/// valid is false when the actual squared basis length is below1e-18; u/v then
/// remain zero. This absolute cutoff is inherited and is not scale invariant near
/// zero. Otherwise u/v are the real/imaginary parts of (p-b0)/(b1-b0), in basis-length
/// units, with the counterclockwise perpendicular axis. Floating results are not
/// exact-real invariants. Values are copied and own no references.
///
/// @par Example
/// @code
/// const auto f=dm::geometry::geometric_hash_frame({1,1},{0,0},{2,0});
/// std::cout << f.valid << " " << f.u << " " << f.v; // 1 0.5 0.5
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see geometric_hash_frame(), GeoHashBin
struct GeoHashFrame {
    bool valid{false}; ///< Whether the actual squared basis length passes the fixed cutoff.
    double u{0}; ///< Actual ordered-double invariant coordinates, zero if degenerate.
    double v{0}; ///< Actual ordered-double invariant coordinates, zero if degenerate.
    double length_squared{0}; ///< Actual double squared basis length, in input units squared.
};
/// @brief A floor-quantized signed grid cell and its lossless packed key.
///
/// u/v are floor(invariant coordinate/quant), checked to fit int32 before casting.
/// key places the uint32 bit pattern of u in the upper32 bits and v in the lower32.
/// Negative cells use floor, not truncation. No neighbouring-cell search is implied.
///
/// @par Example
/// @code
/// const auto b=dm::geometry::geometric_hash_bin(-0.1,0.6,0.5);
/// std::cout << b.u << " " << b.v; // -1 1
/// @endcode
/// @see geometric_hash_bin(), GeoHashTable
struct GeoHashBin {
    std::int32_t u{0}; ///< Signed floor cell coordinates.
    std::int32_t v{0}; ///< Signed floor cell coordinates.
    std::uint64_t key{0}; ///< Packed two signed-cell bit patterns.
};
/// @brief Selected actual kernel operation counts for one build or recognition.
///
/// Counts include every attempted nonbasis-point frame, even for degenerate bases.
/// Insertions and votes retain multiplicity; candidate comparisons include all
/// unordered tally entries, even nonimprovements. Verification counts actual first-hit
/// checks and stops per model point. Input/table validation, hash internals, allocation,
/// trigonometric FLOPs and wall time are excluded: these are not total cost or timing.
/// Each call owns fresh counters; size_t overflow throws length_error.
///
/// @par Example
/// @code
/// const dm::geometry::GeoHashWork w{};
/// std::cout << w.frame_attempts << " " << w.vote_updates; // 0 0
/// @endcode
/// @see build_geometric_hash_run(), recognize_run()
struct GeoHashWork {
    std::size_t frame_attempts{0}; ///< Actual quotient attempts, excluding the two basis points.
    std::size_t degenerate_frames{0}; ///< Attempts rejected by the fixed squared-length cutoff.
    std::size_t insertions{0}; ///< Model basis records appended to table buckets.
    std::size_t bin_lookups{0}; ///< Valid scene-frame table lookups.
    std::size_t bin_hits{0}; ///< Scene lookups whose bin exists, including empty public buckets.
    std::size_t vote_updates{0}; ///< Actual tally increments, one per bucket record.
    std::size_t candidate_comparisons{0}; ///< Tally entries compared against the current global best.
    std::size_t best_updates{0}; ///< Strict global vote improvements.
    std::size_t recoveries{0}; ///< Completed candidate transform recoveries, at most1.
    std::size_t model_points_checked{0}; ///< Model points entering final verification.
    std::size_t distance_checks{0}; ///< Actual scene distances checked, with first-hit breaks.
};
/// @brief An ordinary table build together with its actual operation counts.
///
/// Contains no history. Copies own table bucket vectors and counters independently.
///
/// @par Example
/// @code
/// const auto r=dm::geometry::build_geometric_hash_run({{0,0},{2,0},{0,1}},0.25);
/// std::cout << r.work.frame_attempts << " " << r.work.insertions; // 6 6
/// @endcode
/// @see build_geometric_hash_run(), GeoHashBuildTrace
struct GeoHashBuildRun {
    GeoHashTable table; ///< Owned table produced by the actual build.
    GeoHashWork work; ///< Selected actual build operations.
};
/// @brief An ordinary recognition candidate together with actual work.
///
/// Contains no replay or history. match.found is a recovered candidate, not geometric
/// acceptance. Input models and tables are never referenced after return.
///
/// @par Example
/// @code
/// const auto r=dm::geometry::recognize_run({}, {}, {});
/// std::cout << r.match.found << " " << r.work.frame_attempts; // 0 0
/// @endcode
/// @see recognize_run(), GeoHashTrace
struct GeoHashRun {
    GeoMatch match; ///< Actual selected candidate and first-hit inliers.
    GeoHashWork work; ///< Selected actual recognition operations.
};
/// @brief The primitive operation that emitted one owned history record.
///
/// Events are captured inside the actual kernel, in execution order. Records have
/// fixed primitive fields; no table snapshots or numerical replay are stored.
///
/// @par Example
/// @code
/// const auto k=dm::geometry::GeoHashEventKind::build_frame;
/// std::cout << (k==dm::geometry::GeoHashEventKind::build_frame); // 1
/// @endcode
/// @see GeoHashEvent, build_geometric_hash_trace(), recognize_trace()
enum class GeoHashEventKind {
    build_frame, ///< Actual model quotient attempt; accepted means nondegenerate.
    insert, ///< Actual table append; count is the resulting bucket length.
    scene_frame, ///< Actual scene quotient attempt; accepted means nondegenerate.
    lookup, ///< Actual table lookup; accepted means the bin exists.
    vote, ///< Actual tally increment; count is its new vote count.
    candidate, ///< Actual tally comparison; accepted means strict global improvement.
    recover, ///< Completed actual selected transform recovery.
    distance ///< Actual verification distance; accepted means within the squared tolerance.
};
/// @brief A copied fixed-schema record from the executing hashing kernel.
///
/// For frame/insert/lookup/vote/candidate: first/second identify the active model
/// or scene basis, point the other point (minus1 for candidate). Model ids identify
/// vote/candidate/recover bases. For distance, first is the model point and second
/// the checked scene point; point is unused. u/v store frame coordinates; key stores
/// a bin for insert/lookup or packed model basis for vote/candidate. count stores
/// bucket length/votes. x/y and squared_distance store actual verification values;
/// for frames squared_distance instead stores actual squared basis length.
/// Recovery uses cs/sn, scale/angle and translation. Unused fields retain defaults.
/// accepted means valid frame, hit, strict improvement or inlier as selected by kind.
/// All fields own primitives; modifying records does not change the ordinary result.
///
/// @par Example
/// @code
/// const dm::geometry::GeoHashEvent e{};
/// std::cout << e.first << " " << e.accepted; // -1 0
/// @endcode
/// @see GeoHashEventKind, GeoHashTrace
struct GeoHashEvent {
    GeoHashEventKind kind{GeoHashEventKind::build_frame}; ///< Operation defining active field meanings.
    int first{-1}; ///< Active ordered basis/query or verification model/scene ids.
    int second{-1}; ///< Active ordered basis/query or verification model/scene ids.
    int point{-1}; ///< Active ordered basis/query or verification model/scene ids.
    int model_i{-1}; ///< Basis receiving a vote, compared or recovered.
    int model_j{-1}; ///< Basis receiving a vote, compared or recovered.
    std::uint64_t key{0}; ///< Packed bin or model basis for the corresponding operation.
    std::size_t count{0}; ///< Resulting bucket length or nonnegative tally count.
    bool accepted{false}; ///< Valid frame, lookup hit, strict improvement or distance acceptance.
    double u{0}; ///< Actual invariant coordinates for frame records.
    double v{0}; ///< Actual invariant coordinates for frame records.
    double x{0}; ///< Actual transformed model point for distance records.
    double y{0}; ///< Actual transformed model point for distance records.
    double squared_distance{0}; ///< Actual checked distance, or frame squared basis length.
    double cs{0}; ///< Actual recovered scale*cos(angle) and scale*sin(angle).
    double sn{0}; ///< Actual recovered scale*cos(angle) and scale*sin(angle).
    double scale{0}; ///< Actual recovered scale and atan2 angle in radians.
    double angle{0}; ///< Actual recovered scale and atan2 angle in radians.
    Point2D translation{0,0}; ///< Actual recovered translation for a recovery record.
};
/// @brief An owned ordinary result and complete online primitive history.
///
/// events contains actual operations only, with no synthetic initial record.
/// The positive optional record quota is checked before each append; zero disables
/// that quota. Capacity/counter/domain guards remain. On rejection no partial trace
/// returns and caller data is unchanged. Input vectors, root/work/final ordinary
/// output, working hash/tally and allocator storage are excluded. A record is not a
/// numeric entry or byte; this is not a peak RAM limit. Copies are independent.
///
/// @par Example
/// @code
/// const auto t=dm::geometry::build_geometric_hash_trace({});
/// std::cout << t.events.size(); // 0
/// @endcode
/// @see build_geometric_hash_trace(), GeoHashEvent
struct GeoHashBuildTrace {
    GeoHashBuildRun run; ///< Owned ordinary result and selected work.
    std::vector<GeoHashEvent> events; ///< Complete actual primitive events, in execution order.
};
/// @brief An owned ordinary result and complete online primitive history.
///
/// events contains actual operations only, with no synthetic initial record.
/// The positive optional record quota is checked before each append; zero disables
/// that quota. Capacity/counter/domain guards remain. On rejection no partial trace
/// returns and caller data is unchanged. Input vectors, root/work/final ordinary
/// output, working hash/tally and allocator storage are excluded. A record is not a
/// numeric entry or byte; this is not a peak RAM limit. Copies are independent.
///
/// @par Example
/// @code
/// const auto t=dm::geometry::recognize_trace({}, {}, {});
/// std::cout << t.events.size(); // 0
/// @endcode
/// @see recognize_trace(), GeoHashEvent
struct GeoHashTrace {
    GeoHashRun run; ///< Owned ordinary result and selected work.
    std::vector<GeoHashEvent> events; ///< Complete actual primitive events, in execution order.
};
/// @brief Build the legacy model basis index.
///
/// For every ordered distinct basis, compute all other point frames, skipping
/// actual squared basis lengths below1e-18, and append every basis to its bin.
/// For n points, n(n−1)(n−2) frames when n>=3 and at most that many entries;
/// average hash cost O(n^3), storage O(n^3), with no artificial point-count cap.
/// Empty or fewer-than-three models produce no entries. Coordinate vectors are fully checked finite and int-sized before work.
/// Nonfinite double intermediates or out-of-int32 floor cells are rejected rather
/// than cast. Input/table mutation by another thread is unsupported; separate calls
/// are independent. Hash average-case bounds do not remove collision-list costs.
///
/// @param model Ordered finite model points; coordinates in arbitrary planar units.
/// @param quant Positive finite invariant-coordinate bin width, default0.1.
/// @return Owned quantized table with repeated basis records.
/// @throws std::invalid_argument Invalid finite domain, quantization, tolerance, threshold or table ids as applicable.
/// @throws std::overflow_error A required double intermediate or signed grid cell is out of range.
/// @throws std::length_error Point ids, counters or container capacities cannot be represented.
/// @par Example
/// @code
/// const auto t=dm::geometry::build_geometric_hash({{0,0},{2,0},{0,1}},0.25);
/// std::cout << t.bins.size(); // 6
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see GeoHashTable, geometric_hash_frame(), recognize()
GeoHashTable build_geometric_hash(const std::vector<Point2D>& model, double quant=0.1);
/// @brief Build the legacy model basis index with actual operation counts.
///
/// For every ordered distinct basis, compute all other point frames, skipping
/// actual squared basis lengths below1e-18, and append every basis to its bin.
/// For n points, n(n−1)(n−2) frames when n>=3 and at most that many entries;
/// average hash cost O(n^3), storage O(n^3), with no artificial point-count cap.
/// Empty or fewer-than-three models produce no entries. Coordinate vectors are fully checked finite and int-sized before work.
/// Nonfinite double intermediates or out-of-int32 floor cells are rejected rather
/// than cast. Input/table mutation by another thread is unsupported; separate calls
/// are independent. Hash average-case bounds do not remove collision-list costs.
///
/// @param model Ordered finite model points; coordinates in arbitrary planar units.
/// @param quant Positive finite invariant-coordinate bin width, default0.1.
/// @return Owned table and actual work.
/// @throws std::invalid_argument Invalid finite domain, quantization, tolerance, threshold or table ids as applicable.
/// @throws std::overflow_error A required double intermediate or signed grid cell is out of range.
/// @throws std::length_error Point ids, counters or container capacities cannot be represented.
/// @par Example
/// @code
/// const auto r=dm::geometry::build_geometric_hash_run({{0,0},{2,0},{0,1}});
/// std::cout << r.work.insertions; // 6
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see GeoHashTable, geometric_hash_frame(), recognize()
GeoHashBuildRun build_geometric_hash_run(const std::vector<Point2D>& model, double quant=0.1);
/// @brief Build the legacy model basis index with actual owned events.
///
/// For every ordered distinct basis, compute all other point frames, skipping
/// actual squared basis lengths below1e-18, and append every basis to its bin.
/// For n points, n(n−1)(n−2) frames when n>=3 and at most that many entries;
/// average hash cost O(n^3), storage O(n^3), with no artificial point-count cap.
/// Empty or fewer-than-three models produce no entries. Coordinate vectors are fully checked finite and int-sized before work.
/// Nonfinite double intermediates or out-of-int32 floor cells are rejected rather
/// than cast. Input/table mutation by another thread is unsupported; separate calls
/// are independent. Hash average-case bounds do not remove collision-list costs.
/// Uses online complete-record admission (frame plus successful insertion events),
/// not a precomputed replay or early-work estimate. Default1m events;0 disables only
/// optional quota. No partial/truncated return or caller mutation; root/work/final
/// ordinary table, input/working/allocator storage excluded. Not bytes or peak RAM.
///
/// @param model Ordered finite model points; coordinates in arbitrary planar units.
/// @param quant Positive finite invariant-coordinate bin width, default0.1.
/// @param max_records Optional positive complete-event quota, default1000000;0 disables it.
/// @return Owned ordinary table and complete primitive history.
/// @throws std::invalid_argument Invalid finite domain, quantization, tolerance, threshold or table ids as applicable.
/// @throws std::overflow_error A required double intermediate or signed grid cell is out of range.
/// @throws std::length_error Point ids, counters or container capacities cannot be represented.
/// @par Example
/// @code
/// const auto t=dm::geometry::build_geometric_hash_trace({{0,0},{2,0},{0,1}},0.25,12);
/// std::cout << t.events.size(); // 12
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see GeoHashTable, geometric_hash_frame(), recognize()
GeoHashBuildTrace build_geometric_hash_trace(const std::vector<Point2D>& model, double quant=0.1, std::size_t max_records=1000000);
/// @brief Select and verify the legacy highest-vote candidate.
///
/// Preserves all ordered scene bases, exact double operation order, bucket
/// multiplicity, unordered tally iteration and strict global maximum updates.
/// Only the selected maximum is recovered and verified; no other candidates retry.
/// found marks recovery, not inlier acceptance. Verification allows many-to-one
/// matches and breaks at the first scene point within squared tolerance. Ties depend
/// on the standard library, not a portable lexicographic rule. Empty/no-vote cases
/// return found=false. Table ids are fully validated before tally/history work.
/// For n model/m scene points, T stored entries and V actual bucket-record hits,
/// average hash work O(T+n+m+m^3+V+nm), extra tally storage O(n^2); hash internals
/// can cost more. Input/table validation is paid on every call. Coordinate vectors are fully checked finite and int-sized before work.
/// Nonfinite double intermediates or out-of-int32 floor cells are rejected rather
/// than cast. Input/table mutation by another thread is unsupported; separate calls
/// are independent. Hash average-case bounds do not remove collision-list costs.
///
/// @param table Table built from the SAME ordered model; positive finite quant and distinct valid ids required.
/// @param model Ordered finite points used to build table; provenance remains caller responsibility.
/// @param scene Ordered finite scene points, in scene coordinate units.
/// @param tol Nonnegative finite first-hit distance tolerance, default1e-6; its square must be finite.
/// @param min_votes Positive vote threshold, default2; not a minimum inlier count.
/// @return Owned highest-vote candidate and first-hit inlier count.
/// @throws std::invalid_argument Invalid finite domain, quantization, tolerance, threshold or table ids as applicable.
/// @throws std::overflow_error A required double intermediate or signed grid cell is out of range.
/// @throws std::length_error Point ids, counters or container capacities cannot be represented.
/// @par Example
/// @code
/// const auto m=dm::geometry::recognize({}, {}, {});
/// std::cout << m.found << " " << m.votes; // 0 0
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see GeoMatch, GeoHashWork, build_geometric_hash()
GeoMatch recognize(const GeoHashTable& table, const std::vector<Point2D>& model, const std::vector<Point2D>& scene, double tol=1e-6, int min_votes=2);
/// @brief Select and verify the legacy highest-vote candidate with actual work.
///
/// Preserves all ordered scene bases, exact double operation order, bucket
/// multiplicity, unordered tally iteration and strict global maximum updates.
/// Only the selected maximum is recovered and verified; no other candidates retry.
/// found marks recovery, not inlier acceptance. Verification allows many-to-one
/// matches and breaks at the first scene point within squared tolerance. Ties depend
/// on the standard library, not a portable lexicographic rule. Empty/no-vote cases
/// return found=false. Table ids are fully validated before tally/history work.
/// For n model/m scene points, T stored entries and V actual bucket-record hits,
/// average hash work O(T+n+m+m^3+V+nm), extra tally storage O(n^2); hash internals
/// can cost more. Input/table validation is paid on every call. Coordinate vectors are fully checked finite and int-sized before work.
/// Nonfinite double intermediates or out-of-int32 floor cells are rejected rather
/// than cast. Input/table mutation by another thread is unsupported; separate calls
/// are independent. Hash average-case bounds do not remove collision-list costs.
///
/// @param table Table built from the SAME ordered model; positive finite quant and distinct valid ids required.
/// @param model Ordered finite points used to build table; provenance remains caller responsibility.
/// @param scene Ordered finite scene points, in scene coordinate units.
/// @param tol Nonnegative finite first-hit distance tolerance, default1e-6; its square must be finite.
/// @param min_votes Positive vote threshold, default2; not a minimum inlier count.
/// @return Owned candidate and actual work.
/// @throws std::invalid_argument Invalid finite domain, quantization, tolerance, threshold or table ids as applicable.
/// @throws std::overflow_error A required double intermediate or signed grid cell is out of range.
/// @throws std::length_error Point ids, counters or container capacities cannot be represented.
/// @par Example
/// @code
/// const auto r=dm::geometry::recognize_run({}, {}, {});
/// std::cout << r.work.distance_checks; // 0
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see GeoMatch, GeoHashWork, build_geometric_hash()
GeoHashRun recognize_run(const GeoHashTable& table, const std::vector<Point2D>& model, const std::vector<Point2D>& scene, double tol=1e-6, int min_votes=2);
/// @brief Select and verify the legacy highest-vote candidate with actual owned events.
///
/// Preserves all ordered scene bases, exact double operation order, bucket
/// multiplicity, unordered tally iteration and strict global maximum updates.
/// Only the selected maximum is recovered and verified; no other candidates retry.
/// found marks recovery, not inlier acceptance. Verification allows many-to-one
/// matches and breaks at the first scene point within squared tolerance. Ties depend
/// on the standard library, not a portable lexicographic rule. Empty/no-vote cases
/// return found=false. Table ids are fully validated before tally/history work.
/// For n model/m scene points, T stored entries and V actual bucket-record hits,
/// average hash work O(T+n+m+m^3+V+nm), extra tally storage O(n^2); hash internals
/// can cost more. Input/table validation is paid on every call. Coordinate vectors are fully checked finite and int-sized before work.
/// Nonfinite double intermediates or out-of-int32 floor cells are rejected rather
/// than cast. Input/table mutation by another thread is unsupported; separate calls
/// are independent. Hash average-case bounds do not remove collision-list costs.
/// Online complete-event quota checked before each append; default1m,0 optional-off.
/// No synthetic initial record, numerical replay, partial/truncated return or caller
/// mutation. Inputs/root/work/final ordinary candidate, working table/tally/allocator
/// storage excluded; record counts are not bytes or a peak RAM limit.
///
/// @param table Table built from the SAME ordered model; positive finite quant and distinct valid ids required.
/// @param model Ordered finite points used to build table; provenance remains caller responsibility.
/// @param scene Ordered finite scene points, in scene coordinate units.
/// @param tol Nonnegative finite first-hit distance tolerance, default1e-6; its square must be finite.
/// @param min_votes Positive vote threshold, default2; not a minimum inlier count.
/// @param max_records Optional positive complete-event quota, default1000000;0 disables it.
/// @return Owned candidate/work and complete primitive history.
/// @throws std::invalid_argument Invalid finite domain, quantization, tolerance, threshold or table ids as applicable.
/// @throws std::overflow_error A required double intermediate or signed grid cell is out of range.
/// @throws std::length_error Point ids, counters or container capacities cannot be represented.
/// @par Example
/// @code
/// const auto t=dm::geometry::recognize_trace({}, {}, {},1e-6,2,1);
/// std::cout << t.events.size(); // 0
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see GeoMatch, GeoHashWork, build_geometric_hash()
GeoHashTrace recognize_trace(const GeoHashTable& table, const std::vector<Point2D>& model, const std::vector<Point2D>& scene, double tol=1e-6, int min_votes=2, std::size_t max_records=1000000);
/// @brief Evaluate the actual ordered-double two-point quotient.
///
/// The frame sends b0 to(0,0), b1 to(1,0); perpendicular points use the
/// counterclockwise axis. Degenerate actual squared lengths<1e-18 return valid=false.
/// No caller mutation; owned output, constant work/storage. Same kernel as hashing.
///
/// @param p Finite query point.
/// @param b0 Finite first basis point.
/// @param b1 Finite second basis point.
/// @return Actual validity, coordinates and squared basis length.
/// @throws std::invalid_argument A point is nonfinite.
/// @throws std::overflow_error A required double intermediate is nonfinite.
/// @par Example
/// @code
/// const auto f=dm::geometry::geometric_hash_frame({1,1},{0,0},{2,0});
/// std::cout << f.u << " " << f.v << " " << f.length_squared; // 0.5 0.5 4
/// @endcode
/// @par References
/// - Y. Lamdan and H. J. Wolfson (1988), "Geometric Hashing: A General and Efficient
///   Model-Based Recognition Scheme", ICCV,238–249, Sections2.1–2.2.
///   https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/395T/papers/Lamdan88.pdf
/// @see GeoHashFrame, geometric_hash_bin()
GeoHashFrame geometric_hash_frame(const Point2D& p,const Point2D& b0,const Point2D& b1);
/// @brief Check and pack a floor-quantized invariant-coordinate cell.
///
/// Uses the same floor(double/quant) and signed32-bit packing as the index.
/// Complete coordinate/quant validation precedes arithmetic; finite floor bounds
/// are checked before casts. Constant work/storage, no mutation. Adjacent bins can
/// separate arbitrarily close coordinates; quant is not a scene-distance tolerance.
///
/// @param u Finite first invariant coordinate.
/// @param v Finite second invariant coordinate.
/// @param quant Positive finite grid width, default0.1.
/// @return Signed floor indices and packed key.
/// @throws std::invalid_argument Nonfinite coordinates or nonpositive/nonfinite quant.
/// @throws std::overflow_error A floored quotient is nonfinite or outside int32 range.
/// @par Example
/// @code
/// const auto b=dm::geometry::geometric_hash_bin(-0.1,0.6,0.5);
/// std::cout << b.u << " " << b.v; // -1 1
/// @endcode
/// @see GeoHashBin, build_geometric_hash()
GeoHashBin geometric_hash_bin(double u,double v,double quant=0.1);
/// @brief Apply a checked orientation-preserving similarity to copied points.
///
/// Computes cs=scale*cos(angle), sn=scale*sin(angle), then (cs*x-sn*y+tx,
/// sn*x+cs*y+ty), matching the original report operation order. Inputs are completely
/// validated before allocation/computation. No mutation or references into caller
/// storage. Empty input returns empty after validation. O(n) work/output storage;
/// not an affine, reflective or physical-coordinate projection.
///
/// @param points Ordered finite points, with int-representable count.
/// @param scale Positive finite scale, default1.
/// @param angle Finite counterclockwise angle in radians, default0.
/// @param translation Finite scene-unit translation, default origin.
/// @return Owned transformed points in original order.
/// @throws std::invalid_argument Invalid scale, angle, translation or input point.
/// @throws std::overflow_error A required double intermediate/output is nonfinite.
/// @throws std::length_error Point count or output capacity cannot be represented.
/// @par Example
/// @code
/// const auto p=dm::geometry::geometric_transform({{0,0},{1,0}},2,0,{4,-3});
/// std::cout << p[1].x << " " << p[1].y; // 6 -3
/// @endcode
/// @see geometric_hash_frame(), recognize()
std::vector<Point2D> geometric_transform(const std::vector<Point2D>& points,double scale=1,double angle=0,Point2D translation={0,0});
/// @brief Legacy helper names retained for source compatibility, with checked numeric domains.
namespace detail {
/// @brief Evaluate the shared frame and write coordinates only for a valid basis.
/// @param p Finite query point.
/// @param b0 Finite first basis point.
/// @param b1 Finite second basis point.
/// @param u First coordinate output, unchanged for a degenerate basis.
/// @param v Second coordinate output, unchanged for a degenerate basis.
/// @return Whether the squared basis length passes the legacy cutoff.
/// @throws std::invalid_argument A point is nonfinite.
/// @throws std::overflow_error A required double intermediate is nonfinite.
bool gh_frame_coord(const Point2D& p,const Point2D& b0,const Point2D& b1,double& u,double& v);
/// @brief Return the checked legacy packed signed32-bit bin key.
/// @param u Finite first invariant coordinate.
/// @param v Finite second invariant coordinate.
/// @param quant Positive finite bin width.
/// @return Packed key from geometric_hash_bin().
/// @throws std::invalid_argument Nonfinite coordinate or nonpositive/nonfinite quant.
/// @throws std::overflow_error A floor cell is nonfinite or outside signed32-bit range.
std::uint64_t gh_bin_key(double u,double v,double quant);
} // namespace detail
} // namespace datamunge::geometry
