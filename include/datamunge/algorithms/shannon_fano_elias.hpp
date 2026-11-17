#pragma once
/// @file shannon_fano_elias.hpp
/// @brief Exact integer-midpoint prefix code construction and checked complete-payload codecs.
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/prefix_code_types.hpp>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Complete alphabet, codebook/payload, symbol and capture bounds; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::SFEOptions value{};
/// std::print("{}\n",value.max_nodes); // 1000000
/// @endcode
struct SFEOptions {
  std::size_t max_symbols{1000000}; ///< Complete alphabet or source/decoded symbol count.
  std::size_t max_bits{10000000}; ///< Complete sum of codeword lengths or complete payload length.
  std::size_t max_nodes{1000000}; ///< Complete validated codec trie-node cap.
  std::size_t max_trace_entries{1000000}; ///< Actual construction-word or emitted-word records.
};
/// @brief One actual positive-frequency midpoint word.
/// @par Example
/// @code
/// const dm::algorithms::SFEWord value{};
/// std::print("{}\n",value.length); // 0
/// @endcode
struct SFEWord {
  std::size_t       symbol{};     ///< Actual input alphabet index.
  std::uint64_t     frequency{};  ///< Actual positive frequency.
  std::uint64_t     cumulative{}; ///< Actual preceding total in caller symbol order.
  std::size_t       length{};     ///< Actual ceil(log2(total/frequency))+1 word length.
  std::vector<bool> bits;         ///< Actual exact integer long-division midpoint prefix.
};
/// @brief Owned actual complete codebook and construction work.
/// @par Example
/// @code
/// const dm::algorithms::SFECodebookRun value{};
/// std::print("{}\n",value.total); // 0
/// @endcode
struct SFECodebookRun {
  std::vector<std::vector<bool>>
                codebook; ///< Owned actual words; zero-frequency symbols inactive with empty words.
  std::uint64_t total{};  ///< Exact checked complete frequency sum.
  double        entropy{}; ///< Empirical base2 entropy per symbol; zero for an empty alphabet.
  double
      average_bits{}; ///< Actual frequency-weighted emitted word length per symbol; zero when empty.
  std::size_t          emitted_bits{}; ///< Actual generated codeword bits across active symbols.
  std::vector<SFEWord> words;          ///< Optional actual native word records.
};
/// @brief Owned actual static-model payload or decoded symbols and emitted boundaries.
/// @par Example
/// @code
/// const dm::algorithms::SFECodecRun value{};
/// std::print("{}\n",value.bit_visits); // 0
/// @endcode
struct SFECodecRun {
  std::vector<bool>           bits;         ///< Actual encoded payload, encoding only.
  std::vector<int>            symbols;      ///< Actual decoded symbols, decoding only.
  std::size_t                 bit_visits{}; ///< Actual copied or traversed payload bits.
  std::size_t                 trie_nodes{}; ///< Actual validated prefix-trie nodes, including root.
  std::vector<PrefixWordSpan> words;        ///< Optional actual complete-word emissions.
};
/// @brief Construct an exact integer-midpoint Shannon-Fano-Elias codebook.
/// For each positive count f in original symbol order and prior count c, generate L=ceil(log2(total/f))+1 bits of (2*c+f)/(2*total). Length uses integer doubling; bits use exact unsigned128 long division, avoiding floating boundary rounding. Checks complete uint64 total and codeword-bit geometry before generating any words. Zero counts are inactive and receive empty words; an empty alphabet is supported, but a nonempty all-zero alphabet rejects. A singleton active symbol receives word1. Produces a prefix-free code with positive-symbol average length below entropy+2; no Huffman optimality claim. O(alphabet+generated bits) time/output plus optional actual records; inputs borrowed unchanged and outputs owned, no shared state.
/// @param freq Borrowed complete uint32 counts in symbol-id order.
/// @param options Complete alphabet, bit and optional record bounds.
/// @return Owned actual codebook, exact count total, generated-bit count and optional word records.
/// @throws std::invalid_argument Nonempty all-zero frequency model.
/// @throws std::length_error Alphabet/int-id, sum, codeword, history or container capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shannon_fano_elias_run({1,1});
/// std::print("{}\n",r.emitted_bits); // 4
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_encode_run()
SFECodebookRun shannon_fano_elias_run(
    const std::vector<std::uint32_t>& freq,
    const SFEOptions&                 options = {});
/// @brief Construct an exact integer-midpoint Shannon-Fano-Elias codebook with actual native words.
/// For each positive count f in original symbol order and prior count c, generate L=ceil(log2(total/f))+1 bits of (2*c+f)/(2*total). Length uses integer doubling; bits use exact unsigned128 long division, avoiding floating boundary rounding. Checks complete uint64 total and codeword-bit geometry before generating any words. Zero counts are inactive and receive empty words; an empty alphabet is supported, but a nonempty all-zero alphabet rejects. A singleton active symbol receives word1. Produces a prefix-free code with positive-symbol average length below entropy+2; no Huffman optimality claim. O(alphabet+generated bits) time/output plus optional actual records; inputs borrowed unchanged and outputs owned, no shared state.
/// @param freq Borrowed complete uint32 counts in symbol-id order.
/// @param options Complete alphabet, bit and optional record bounds.
/// @return Owned actual codebook, exact count total, generated-bit count and optional word records.
/// @throws std::invalid_argument Nonempty all-zero frequency model.
/// @throws std::length_error Alphabet/int-id, sum, codeword, history or container capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shannon_fano_elias_trace({1,1});
/// std::print("{}\n",r.emitted_bits); // 4
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_encode_run()
SFECodebookRun shannon_fano_elias_trace(
    const std::vector<std::uint32_t>& freq,
    const SFEOptions&                 options = {});
/// @brief Encode actual symbol words using the exact midpoint codebook.
/// Builds the exact native codebook once and delegates the same private prefix-trie producer shared with Shannon-Fano. Validates the complete model and payload geometry before output. Decoder consumes ALL bits and requires exactly count complete words; absent branches, truncation, extra words and mismatched counts reject instead of returning a partial prefix. Empty input/count zero is supported, including an empty alphabet. Encoder rejects absent, negative, out-of-range and zero-frequency source symbols. O(codebook bits+payload bits) time and intrinsic output/trie storage, optional one retained span per emitted word. Inputs unchanged, outputs owned; no shared state.
/// @param symbols Borrowed complete source ids; every selected frequency must be positive.
/// @param freq Borrowed complete static counts.
/// @param options Complete model/payload/symbol/trie/history policies.
/// @return Owned actual payload or decoded symbols, observed bit visits/trie nodes and optional producing word spans.
/// @throws std::invalid_argument Invalid model/source id or incomplete/extra/wrong-count payload.
/// @throws std::length_error Complete model/payload/symbol/trie/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shannon_fano_elias_encode_run({0,1},{1,1});
/// std::print("{}\n",r.bits.size()); // 4
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_run()
SFECodecRun shannon_fano_elias_encode_run(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& freq,
    const SFEOptions&                 options = {});
/// @brief Encode actual symbol words using the exact midpoint codebook with native emitted spans.
/// Builds the exact native codebook once and delegates the same private prefix-trie producer shared with Shannon-Fano. Validates the complete model and payload geometry before output. Decoder consumes ALL bits and requires exactly count complete words; absent branches, truncation, extra words and mismatched counts reject instead of returning a partial prefix. Empty input/count zero is supported, including an empty alphabet. Encoder rejects absent, negative, out-of-range and zero-frequency source symbols. O(codebook bits+payload bits) time and intrinsic output/trie storage, optional one retained span per emitted word. Inputs unchanged, outputs owned; no shared state.
/// @param symbols Borrowed complete source ids; every selected frequency must be positive.
/// @param freq Borrowed complete static counts.
/// @param options Complete model/payload/symbol/trie/history policies.
/// @return Owned actual payload or decoded symbols, observed bit visits/trie nodes and optional producing word spans.
/// @throws std::invalid_argument Invalid model/source id or incomplete/extra/wrong-count payload.
/// @throws std::length_error Complete model/payload/symbol/trie/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shannon_fano_elias_encode_trace({0,1},{1,1});
/// std::print("{}\n",r.bits.size()); // 4
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_run()
SFECodecRun shannon_fano_elias_encode_trace(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& freq,
    const SFEOptions&                 options = {});
/// @brief Decode complete actual midpoint-code words.
/// Builds the exact native codebook once and delegates the same private prefix-trie producer shared with Shannon-Fano. Validates the complete model and payload geometry before output. Decoder consumes ALL bits and requires exactly count complete words; absent branches, truncation, extra words and mismatched counts reject instead of returning a partial prefix. Empty input/count zero is supported, including an empty alphabet. Encoder rejects absent, negative, out-of-range and zero-frequency source symbols. O(codebook bits+payload bits) time and intrinsic output/trie storage, optional one retained span per emitted word. Inputs unchanged, outputs owned; no shared state.
/// @param bits Borrowed complete logical bits; no ignored tail.
/// @param freq Borrowed complete static counts.
/// @param count Required exact complete decoded count, including zero.
/// @param options Complete model/payload/symbol/trie/history policies.
/// @return Owned actual payload or decoded symbols, observed bit visits/trie nodes and optional producing word spans.
/// @throws std::invalid_argument Invalid model/source id or incomplete/extra/wrong-count payload.
/// @throws std::length_error Complete model/payload/symbol/trie/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shannon_fano_elias_decode_run({false,true,true,true},{1,1},2);
/// std::print("{}\n",r.symbols.size()); // 2
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_run()
SFECodecRun shannon_fano_elias_decode_run(
    const std::vector<bool>&          bits,
    const std::vector<std::uint32_t>& freq,
    std::size_t                       count,
    const SFEOptions&                 options = {});
/// @brief Decode complete actual midpoint-code words with native emitted spans.
/// Builds the exact native codebook once and delegates the same private prefix-trie producer shared with Shannon-Fano. Validates the complete model and payload geometry before output. Decoder consumes ALL bits and requires exactly count complete words; absent branches, truncation, extra words and mismatched counts reject instead of returning a partial prefix. Empty input/count zero is supported, including an empty alphabet. Encoder rejects absent, negative, out-of-range and zero-frequency source symbols. O(codebook bits+payload bits) time and intrinsic output/trie storage, optional one retained span per emitted word. Inputs unchanged, outputs owned; no shared state.
/// @param bits Borrowed complete logical bits; no ignored tail.
/// @param freq Borrowed complete static counts.
/// @param count Required exact complete decoded count, including zero.
/// @param options Complete model/payload/symbol/trie/history policies.
/// @return Owned actual payload or decoded symbols, observed bit visits/trie nodes and optional producing word spans.
/// @throws std::invalid_argument Invalid model/source id or incomplete/extra/wrong-count payload.
/// @throws std::length_error Complete model/payload/symbol/trie/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shannon_fano_elias_decode_trace({false,true,true,true},{1,1},2);
/// std::print("{}\n",r.symbols.size()); // 2
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_run()
SFECodecRun shannon_fano_elias_decode_trace(
    const std::vector<bool>&          bits,
    const std::vector<std::uint32_t>& freq,
    std::size_t                       count,
    const SFEOptions&                 options = {});
/// @brief Return the checked exact midpoint codebook through the preserved signature.
/// Uses the canonical producer/default bounds.
/// @param freq Complete counts in symbol order.
/// @return Owned words, with inactive zero-count entries empty.
/// @throws std::invalid_argument Nonempty all-zero model.
/// @throws std::length_error Default geometry/sum policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shannon_fano_elias_codebook({1,1}).size()); // 2
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_run()
std::vector<std::vector<bool>> shannon_fano_elias_codebook(const std::vector<std::uint32_t>& freq);
/// @brief Return the complete encoded midpoint-code payload through the preserved signature.
/// Uses the shared checked native producer/default bounds.
/// @param symbols Complete active source ids.
/// @param freq Complete counts.
/// @return Owned actual logical bits.
/// @throws std::invalid_argument Invalid model or source symbol.
/// @throws std::length_error Default geometry/payload policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shannon_fano_elias_encode({0,1},{1,1}).size()); // 4
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_encode_run()
std::vector<bool> shannon_fano_elias_encode(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& freq);
/// @brief Return exactly the complete decoded count through the preserved signature.
/// Negative counts and partial/extra payloads reject rather than silently returning a prefix.
/// @param bits Complete payload.
/// @param freq Complete counts.
/// @param count Required nonnegative exact symbol count.
/// @return Owned actual decoded ids.
/// @throws std::invalid_argument Negative count, malformed model or incomplete/extra/wrong-count payload.
/// @throws std::length_error Default geometry/payload policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shannon_fano_elias_decode({false,true,true,true},{1,1},2)); // [0, 1]
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication; midpoint construction follows Cover and Thomas, Elements of Information Theory, second edition, section5.9 (2006).
/// @see shannon_fano_elias_decode_run()
std::vector<int> shannon_fano_elias_decode(
    const std::vector<bool>&          bits,
    const std::vector<std::uint32_t>& freq,
    int                               count);
} // namespace datamunge::algorithms
