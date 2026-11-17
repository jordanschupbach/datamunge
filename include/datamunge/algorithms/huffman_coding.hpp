#pragma once
/// @file huffman_coding.hpp
/// @brief Exact-weight Huffman construction, actual merge histories and validated byte codecs.
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace datamunge::algorithms {
/// @brief No-node sentinel in integer-alphabet construction records.
inline constexpr std::size_t kHuffmanNoNode = static_cast<std::size_t>(-1);
/// @brief Owned byte codebook and full decoding tree, preserving historical aggregate layout.
/// Codes contain nonempty binary words; a single leaf uses"0". Empty code has
/// root=-1 and empty map/tree. Nonempty root may be any valid index. Unique
/// byte leaves, full binary structure and exact map/path agreement are required
/// by codecs. Containers own storage; do not mutate during a concurrent call.
/// @par Example
/// @code
/// const auto c=dm::algorithms::huffman_build("aab");
/// std::print("{}\n",c.codes.size()); // 2
/// @endcode
struct HuffmanCode {
  std::unordered_map<char, std::string>
      codes; ///< Owned byte-to-binary-word map, consistent with the tree.

  /// @brief Historical decoding-node layout; symbol is meaningful only for a leaf.
  /// A valid leaf has both children=-1; an internal node has two distinct valid
  /// children. Default construction alone is not a valid nonempty decoding tree.
  /// @par Example
  /// @code
  /// const dm::algorithms::HuffmanCode::Node n{'a',true,-1,-1};
  /// std::print("{}\n",n.leaf); // true
  /// @endcode
  struct Node {
    char symbol{0};   ///< Byte at a leaf; ignored for internal nodes.
    bool leaf{false}; ///< Whether the node denotes a byte leaf.
    int  left{-1};    ///< Child reached by0, or-1 at a leaf.
    int  right{-1};   ///< Child reached by1, or-1 at a leaf.
  };
  std::vector<Node> tree;     ///< Owned full connected binary tree, at most511 nodes.
  int               root{-1}; ///< Explicit root index, or-1 only for a wholly empty code.
};
/// @brief Complete borrowed-input and whole-merge-record policies, each zero disabling its cap.
/// Input entries are count-vector positions, including zeros, or source text
/// bytes depending on the entrypoint. Merge cap counts records only; owned input
/// copies, ordinary output/nodes/lengths, decimal/scalar metadata, capacity and
/// byte/peak memory are separate. Known m-1 records are admitted before heap work.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanOptions o;std::print("{}\n",o.max_input_entries); // 1000000
/// @endcode
struct HuffmanOptions {
  std::size_t max_input_entries{1000000}; ///< Complete borrowed input-element cap.
  std::size_t max_merge_records{
      1000000}; ///< Complete actual merge-record cap; ignored without capture.
};
/// @brief Actual generic construction topology, with original integer symbol indices at leaves.
/// Node IDs are compact positive leaves in ascending symbol order, then actual
/// parent creation order. Leaf child indices use kHuffmanNoNode.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanCountNode n;std::print("{}\n",n.leaf); // false
/// @endcode
struct HuffmanCountNode {
  std::size_t symbol{0};             ///< Original count-vector index for a leaf, ignored otherwise.
  bool        leaf{false};           ///< True for an original positive-count symbol.
  std::size_t left{kHuffmanNoNode};  ///< Actual first-popped child, assigned bit0.
  std::size_t right{kHuffmanNoNode}; ///< Actual second-popped child, assigned bit1.
};
/// @brief Actual selected construction visits, excluding validation/copies/allocations/decimal conversion.
/// Heap comparison internals and multiprecision limb operations are not counted.
/// Counts are not timings, FLOPs or complete memory traffic.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanWork w;std::print("{}\n",w.heap_pops); // 0
/// @endcode
struct HuffmanWork {
  std::size_t histogram_bytes{0}; ///< Actual byte-histogram visits, zero for count-vector input.
  std::size_t counts_visited{
      0}; ///< Entries visited by the initial complete support/total scan, including zeros.
  std::size_t heap_pushes{0}; ///< Actual leaf and parent insertions.
  std::size_t heap_pops{0};   ///< Actual two-child removals, excluding final root peek.
  std::size_t tree_visits{0}; ///< Actual nodes visited assigning integer lengths.
};
/// @brief Owned exact-weight construction result for an integer-symbol alphabet.
/// All uint64 counts are accepted without restricting their sum to uint64.
/// Exact totals/costs use decimal strings to avoid exposing an arithmetic backend.
/// Zero-count lengths are0; a lone positive symbol uses length1. For m>=2,
/// weighted_bits equals merge_cost; for m=1 they are count and0 respectively.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanCountsRun r;std::print("{}\n",r.active_symbols); // 0
/// @endcode
struct HuffmanCountsRun {
  std::vector<int> lengths; ///< Owned original-index lengths; zeros for absent symbols.
  std::vector<HuffmanCountNode>
              nodes; ///< Owned actual full construction tree, empty if no positive counts.
  std::size_t root{kHuffmanNoNode}; ///< Actual final root, or no-node sentinel for empty support.
  std::size_t active_symbols{0};    ///< Actual positive-count symbols.
  std::string total_weight{"0"};    ///< Exact complete sum of supplied counts, decimal.
  std::string weighted_bits{"0"};   ///< Exact sum(count[s]*length[s]), decimal.
  std::string merge_cost{"0"};      ///< Exact sum of all actual parent weights, decimal.
  HuffmanWork work;                 ///< Actual selected shared-kernel work.
};
/// @brief Owned actual two minimum removals and parent insertion from one producing merge.
/// Decimal weights are exact, including values beyond uint64. IDs refer to the
/// producing compact node array, not a reconstructed heap or canonical code.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanMerge m;std::print("{}\n",m.parent_weight); // 0
/// @endcode
struct HuffmanMerge {
  std::size_t left{0};            ///< Actual first minimum removed, bit0 child.
  std::size_t right{0};           ///< Actual second minimum removed, bit1 child.
  std::size_t parent{0};          ///< Actual newly inserted parent ID.
  std::string left_weight{"0"};   ///< Exact weight of first removed item.
  std::string right_weight{"0"};  ///< Exact weight of second removed item.
  std::string parent_weight{"0"}; ///< Exact sum reinserted under parent ID.
};
/// @brief Owned count input, actual complete merges and producing construction result.
/// Merge admission is complete before heap work: m-1 records for m>=1, else0.
/// No truncation/replay/partial return/input references. Input copy and ordinary
/// output are covered separately by complete input admission, not record count.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanCountsTrace t;std::print("{}\n",t.merges.size()); // 0
/// @endcode
struct HuffmanCountsTrace {
  HuffmanCountsRun           run;    ///< Actual producing result, costs and selected visits.
  std::vector<std::uint64_t> counts; ///< Owned complete original input, including zeros.
  std::vector<HuffmanMerge>
      merges; ///< Actual chronological minimum-removal/parent-insertion records.
};
/// @brief Owned byte code and the shared exact-weight integer construction that produced it.
/// Integer symbols are unsigned byte values0..255; ordinary topology/lengths
/// accompany the byte model. No references or shared mutable state.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanBuildRun r;std::print("{}\n",r.code.root); // -1
/// @endcode
struct HuffmanBuildRun {
  HuffmanCode      code;         ///< Actual projected byte decoding tree and codewords.
  HuffmanCountsRun construction; ///< Actual shared256-position frequency construction.
};
/// @brief Owned actual byte frequencies, merge records and producing byte model.
/// No source-text copy or retained references. Merge quota excludes fixed256
/// frequency entries and ordinary output; it is not a memory-byte limit.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanBuildTrace t;std::print("{}\n",t.frequencies.size()); // 0
/// @endcode
struct HuffmanBuildTrace {
  HuffmanBuildRun run; ///< Actual byte code, topology, exact costs and work.
  std::vector<std::uint64_t>
      frequencies; ///< Owned256-entry unsigned-byte histogram after successful construction.
  std::vector<HuffmanMerge> merges; ///< Actual producing shared-kernel merge order.
};
/// @brief Build byte Huffman codes with unsigned-byte leaf order and creation-ID ties.
/// Counts arbitrary bytes, including NUL/high bytes; not Unicode codepoints.
/// Leaves ascend0..255, equal weights break by node creation ID, first removed
/// child gets0. This portable rule can change older unordered-map tie choices;
/// no serialized wire/codebook format is implied. Empty yields empty code;
/// singleton uses"0". For at least2 positive symbols, H<=mean_length<H+1;
/// forced positive singleton length1 instead equalsH+1 for H=0.
/// Uses a shared exact-integer weight heap; no uint64-sum overflow. Heap operation
/// count O(m log(m+1)) plus O(bytes+256+nodes+total codeword characters), with
/// integer comparison cost depending on weight bit length. No optional policies.
/// Input borrowed only during the call; owned code/output, no shared state.
/// @param text Borrowed complete source byte string, empty permitted.
/// @return Owned valid byte codebook and decoding tree, historical layout preserved.
/// @throws std::length_error On count/container/length representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto c=dm::algorithms::huffman_build("aab");std::print("{}\n",c.codes.at('a')); // 1
/// @endcode
/// @par References
/// - D. A. Huffman (1952), A Method for the Construction of Minimum-Redundancy
///   Codes, doi:10.1109/JRPROC.1952.273898 . Binary, positive singleton convention.
/// @see huffman_build_run(), huffman_build_trace(), validate_huffman_code()
HuffmanCode huffman_build(const std::string& text);
/// @brief Build the shared byte model with complete input policy and selected construction visits.
/// Same order/domains/ownership as huffman_build(); merge quota ignored without
/// capture. Input quota counts source bytes, not fixed256 histogram positions.
/// @param text Borrowed complete arbitrary byte string.
/// @param options Complete source-byte cap, default1m; zero disables.
/// @return Owned actual byte code and producing exact construction result.
/// @throws std::length_error On complete input/count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::huffman_build_run("aab");std::print("{}\n",r.construction.weighted_bits); // 3
/// @endcode
/// @see HuffmanBuildRun
HuffmanBuildRun huffman_build_run(const std::string& text, HuffmanOptions options = {});
/// @brief Capture complete actual byte-model merges during their producing heap execution.
/// Same contracts as huffman_build_run(). Known full m-1 merge count admitted
/// before heap work; records own exact decimal weights, with no replay/truncation.
/// @param text Borrowed source bytes, never retained or mutated.
/// @param options Complete source-byte/merge-record caps, default1m each; zero disables each.
/// @return Owned byte histogram, actual merge order and producing result.
/// @throws std::length_error On input/complete merge/count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::huffman_build_trace("aab").merges.size()); // 1
/// @endcode
/// @see HuffmanBuildTrace
HuffmanBuildTrace huffman_build_trace(const std::string& text, HuffmanOptions options = {});
/// @brief Construct integer-symbol Huffman lengths with exact arbitrary-size combined weights.
/// All uint64 input values supported, including totals beyond uint64. Zeros get
/// length0; singleton positive length1. Empty/allzero support has root sentinel,
/// zero costs and no nodes. Heap ties use compact positive-leaf original-index
/// order then parent creation ID. Comparison cost depends on integer bit length;
/// O(K+m log(m+1)) heap/scan/node operations and O(K+m) owned/working entries.
/// Complete input policy counts K positions, including zeros; record quota ignored.
/// @param counts Borrowed K nonnegative uint64 frequencies, empty permitted.
/// @param options Complete input-entry cap, default1m; zero disables.
/// @return Owned original-index lengths, actual tree, exact decimal costs and selected work.
/// @throws std::length_error On input/count/container or int-depth representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::huffman_counts_run({1,1,2});std::print("{}\n",r.weighted_bits); // 6
/// @endcode
/// @see huffman_code_lengths(), huffman_counts_trace()
HuffmanCountsRun huffman_counts_run(
    const std::vector<std::uint64_t>& counts,
    HuffmanOptions                    options = {});
/// @brief Capture actual complete exact-weight merges from integer-count construction.
/// Same semantics as huffman_counts_run(). Full m-1 record admission precedes
/// heap allocation/work, excluding input copies, ordinary output and metadata.
/// @param counts Borrowed full original-index counts, including zeros.
/// @param options Complete count-entry/merge-record caps, default1m each; zero disables.
/// @return Owned input, producing result and actual chronological merge records.
/// @throws std::length_error On input/merge/count/container/int-depth limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::huffman_counts_trace({1,1,2});std::print("{}\n",t.merges[0].parent_weight); // 2
/// @endcode
/// @see HuffmanCountsTrace
HuffmanCountsTrace huffman_counts_trace(
    const std::vector<std::uint64_t>& counts,
    HuffmanOptions                    options = {});
/// @brief Preserve the original integer-length interface with no optional input policies.
/// Uses the same exact-weight kernel and original-index output as
/// huffman_counts_run(). Corrects old uint64 merge overflow; no sum restriction.
/// Singleton convention means mean_length=1=H+1, not the strict multi-symbol bound.
/// @param counts Borrowed complete arbitrary uint64 frequency vector, empty allowed.
/// @return Owned code lengths, zero for absent symbols and1 for a lone positive symbol.
/// @throws std::length_error On count/container/int-depth limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::huffman_code_lengths({0,7,0})[1]); // 1
/// @endcode
/// @see huffman_counts_run()
std::vector<int> huffman_code_lengths(const std::vector<std::uint64_t>& counts);
/// @brief Validate the complete byte model before any codec tree walk.
/// Requires a connected acyclic full tree with unique byte leaves, at most511
/// nodes, valid child/root indices, leaf children=-1 and exact code-map/path
/// agreement. Accepts valid older arbitrary node ordering. Singleton code is"0".
/// Empty requires root=-1 and empty tree/map. Bounded O(nodes*depth+codes) work
/// and bounded validation storage; no input mutation or retained references.
/// @param code Borrowed potentially malformed byte codebook/tree.
/// @throws std::invalid_argument On any structural/map/empty-model invariant failure.
/// @throws std::bad_alloc On bounded validation allocation failure.
/// @par Example
/// @code
/// dm::algorithms::validate_huffman_code(dm::algorithms::huffman_build("ab"));std::print("valid\n"); // valid
/// @endcode
/// @see huffman_encode(), huffman_decode()
void validate_huffman_code(const HuffmanCode& code);
/// @brief Encode with a validated byte model to an ASCII string of logical0/1 digits.
/// This is not packed storage or an archive; codebook/framing costs are separate.
/// Full model and symbol membership/encoded-size checks precede output allocation.
/// O(model validation+text length+output digits) work and owned output storage.
/// Empty text yields empty output after model validation. No input references.
/// @param hc Borrowed complete valid model; older valid node ordering supported.
/// @param text Borrowed arbitrary bytes, all of which must exist in the model.
/// @return Owned ASCII0/1 payload, one character per logical bit.
/// @throws std::invalid_argument On malformed model.
/// @throws std::out_of_range On a source byte missing from a valid model.
/// @throws std::length_error If the complete payload cannot fit a string.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::huffman_encode(dm::algorithms::huffman_build("aab"),"aab")); // 110
/// @endcode
/// @see huffman_decode(), huffman_average_bits()
std::string huffman_encode(const HuffmanCode& hc, const std::string& text);
/// @brief Decode only complete binary payloads under a fully validated byte tree/map.
/// Checks every input character is0/1 before walking. Rejects nonempty bits for
/// an empty model, singleton1 bits, and incomplete suffixes instead of silently
/// dropping them. Never follows unchecked indices or cycles. Empty payload gives
/// empty output for every valid model. O(model validation+bits) work/output bound;
/// no partial result, mutation or retained references.
/// @param hc Borrowed potentially malformed complete byte model.
/// @param bits Borrowed ASCII0/1 characters; not packed bits.
/// @return Owned decoded bytes, including NUL/high bytes when present.
/// @throws std::invalid_argument On model/nonbinary/singleton/truncated-input failure.
/// @throws std::length_error On output container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::huffman_decode(dm::algorithms::huffman_build("aab"),"110")); // aab
/// @endcode
/// @see validate_huffman_code(), huffman_encode()
std::string huffman_decode(const HuffmanCode& hc, const std::string& bits);
/// @brief Return complete weighted logical code length per source byte without sum wraparound.
/// Validates the whole model and source membership. Exact integer length sum is
/// converted to double then divided, preserving ordinary historical results.
/// Empty text returns0 after model validation. O(model validation+text length)
/// operations plus exact-integer accumulation cost, no input references/mutation.
/// @param hc Borrowed valid complete byte model.
/// @param text Borrowed source bytes, all present in the model.
/// @return Computed double mean logical bits per byte, singleton1 and empty0.
/// @throws std::invalid_argument On malformed model.
/// @throws std::out_of_range On a missing source byte.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::huffman_average_bits(dm::algorithms::huffman_build("aab"),"aab")); // 1
/// @endcode
/// @see huffman_counts_run()
double huffman_average_bits(const HuffmanCode& hc, const std::string& text);
} // namespace datamunge::algorithms
