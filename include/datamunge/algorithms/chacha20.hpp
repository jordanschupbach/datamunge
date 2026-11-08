#pragma once
/// @file chacha20.hpp
/// @brief RFC 8439 byte-oriented ChaCha20 with actual ARX/block/XOR traces and counter checks.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned four-word input, four actual compound ARX stages and final quarter round.
/// Each stage follows an add/xor/rotate group, with rotation amounts 16,12,8,7.
/// Arithmetic is modulo 2^32. No replay or shared state; this record may contain
/// key-derived material. It performs no memory erasure or security certification.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chacha20_quarter_trace({0x11111111,0x01020304,0x9b8d6f43,0x01234567});
/// std::print("{:08x}\n",t.output[0]); // ea2a92f4
/// @endcode
/// @see chacha20_quarter_trace()
struct ChaChaQuarterTrace {
  std::array<std::uint32_t,4> input{}; ///< Original a,b,c,d words.
  std::array<std::array<std::uint32_t,4>,4> stages{}; ///< Actual states after each compound ARX group.
  std::array<std::uint32_t,4> output{}; ///< Final actual a,b,c,d, equal to stages.back().
};
/// @brief One actual quarter-round application inside a full block.
/// before/after use the indices order; round zero is the first column round.
/// Owned values and indices remain independent of the live working state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chacha20_block_trace({},0,{});
/// std::print("{}\n",t.quarter_rounds[0].indices); // [0, 4, 8, 12]
/// @endcode
/// @see ChaChaBlockTrace
struct ChaChaQuarterStep {
  std::size_t round=0; ///< Zero-based round in [0,19].
  std::array<std::size_t,4> indices{}; ///< Actual selected row-major state-word indices.
  std::array<std::uint32_t,4> before{}; ///< Actual selected words before the quarter round.
  std::array<std::uint32_t,4> after{}; ///< Actual selected words after the quarter round.
};
/// @brief Owned actual state setup, 80 quarter rounds, 20 round states and serialization.
/// Trace factories populate the record in the same kernel used by ordinary blocks;
/// no round is replayed. initial contains the key, and later fields contain key-derived
/// state/keystream. Use public test material for exposed diagnostics; records do not
/// erase their memory. Default construction alone does not execute a block.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chacha20_block_trace({},0,{});
/// std::print("{} {:02x}\n",t.quarter_rounds.size(),t.output[0]); // 80 76
/// @endcode
/// @see chacha20_block_trace(), chacha20_block()
struct ChaChaBlockTrace {
  std::array<std::uint32_t,16> initial{}; ///< Actual constants/key/counter/nonce setup, little-endian words.
  std::vector<ChaChaQuarterStep> quarter_rounds; ///< Actual 80 quarter-round selections and values.
  std::array<std::array<std::uint32_t,16>,20> round_states{}; ///< Actual state after each round, before feed-forward.
  std::array<std::uint32_t,16> feed_forward{}; ///< Actual final mixed state plus initial, modulo 2^32.
  std::array<std::uint8_t,64> output{}; ///< Actual little-endian serialized keystream block.
};
/// @brief One actual block used by one-shot XOR, retaining the full keystream.
/// bytes may be less than 64 for the final block; unused tail bytes are retained
/// here but discarded by the one-shot stream position. No source-data reference is held.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chacha20_encrypt_trace({},1,{},std::string(65,'x'));
/// std::print("{} {}\n",t.blocks[1].offset,t.blocks[1].bytes); // 64 1
/// @endcode
/// @see ChaChaEncryptTrace
struct ChaChaXorBlock {
  std::uint32_t counter=0; ///< Actual counter passed to the block function, never wrapped.
  std::size_t offset=0; ///< First message byte processed by this block.
  std::size_t bytes=0; ///< Actual number of message bytes XORed, in [1,64].
  std::array<std::uint8_t,64> keystream{}; ///< Full actual block, including any unused tail.
};
/// @brief Owned one-shot output and actual block/counter/offset/keystream records.
/// Contains sensitive keystream and output material; no erasure/authentication is
/// performed. Source input/key/nonce are borrowed only during the call. O(n+64*b)
/// storage for n bytes and b=ceil(n/64) blocks; empty input has no block records.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chacha20_encrypt_trace({},0,{},std::string(65,'x'));
/// std::print("{} {}\n",t.output.size(),t.blocks.size()); // 65 2
/// @endcode
/// @see chacha20_encrypt_trace(), chacha20_encrypt()
struct ChaChaEncryptTrace {
  std::string output; ///< Owned binary XOR output, same length as input.
  std::vector<ChaChaXorBlock> blocks; ///< Actual block uses in increasing message-offset order.
};
/// @brief Number of bytes available before a 32-bit counter would repeat.
/// Includes the starting counter and UINT32_MAX. Raw counter zero permits 2^32
/// blocks, or 2^38 bytes. This is a per-call range limit, not nonce management or
/// a check for reuse across separate calls. Constant work/storage, no mutation.
/// @param counter Initial 32-bit block counter.
/// @return Exact uint64 byte capacity, at least 64 and at most 2^38.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::chacha20_capacity_bytes(0xffffffffU)); // 64
/// @endcode
/// @par References
/// - RFC 8439, Sections 2.3 and 2.4. https://www.rfc-editor.org/rfc/rfc8439.html
/// @see chacha20_blocks_required(), chacha20_encrypt()
inline constexpr std::uint64_t chacha20_capacity_bytes(std::uint32_t counter) noexcept {
  return ((std::uint64_t{1}<<32)-counter)*64;
}
/// @brief Validate a complete byte range and return its required block count without allocation.
/// Checks before any counter addition or message copy. Empty input needs zero blocks
/// even at the final counter. Ceiling division avoids overflowing bytes+63.
/// @param counter Initial 32-bit block counter.
/// @param bytes Complete message length in bytes, any uint64 value.
/// @return Exact ceil(bytes/64) when the counter range is sufficient.
/// @throws std::length_error Message would require a counter beyond UINT32_MAX.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::chacha20_blocks_required(0xfffffffeU,128)); // 2
/// @endcode
/// @par References
/// - RFC 8439, Sections 2.3 and 2.4. https://www.rfc-editor.org/rfc/rfc8439.html
/// @see chacha20_capacity_bytes(), chacha20_encrypt()
inline std::uint64_t chacha20_blocks_required(std::uint32_t counter,std::uint64_t bytes) {
  if(bytes>chacha20_capacity_bytes(counter))throw std::length_error("ChaCha20: counter range exhausted");
  return bytes/64+static_cast<std::uint64_t>(bytes%64!=0);
}
/// @brief Private shared ARX, setup and optional-recording kernels.
namespace detail {
/// @brief Rotate a 32-bit word left by a signed count reduced modulo 32.
/// @param x Unsigned 32-bit input.
/// @param c Rotation count, including zero, negative and multiples of 32.
/// @return Rotated word without an invalid shift width.
inline std::uint32_t chacha_rotl(std::uint32_t x,int c) {
  const auto shift=static_cast<unsigned int>(c)&31U;
  return shift?static_cast<std::uint32_t>((x<<shift)|(x>>(32-shift))):x;
}
/// @brief Apply four actual compound ARX groups, optionally recording their states.
/// @param a First selected word, mutated modulo 2^32.
/// @param b Second selected word, mutated.
/// @param c Third selected word, mutated.
/// @param d Fourth selected word, mutated; all four references must be distinct.
/// @param trace Optional owned record target; null skips all snapshots.
inline void chacha_quarter_core(std::uint32_t& a,std::uint32_t& b,std::uint32_t& c,std::uint32_t& d,
    ChaChaQuarterTrace* trace) {
  if(trace)trace->input={a,b,c,d};
  a+=b;d=chacha_rotl(d^a,16);if(trace)trace->stages[0]={a,b,c,d};
  c+=d;b=chacha_rotl(b^c,12);if(trace)trace->stages[1]={a,b,c,d};
  a+=b;d=chacha_rotl(d^a,8);if(trace)trace->stages[2]={a,b,c,d};
  c+=d;b=chacha_rotl(b^c,7);if(trace){trace->stages[3]={a,b,c,d};trace->output={a,b,c,d};}
}
/// @brief Retain the legacy private quarter-round spelling for internal callers.
/// @param a Distinct first selected word, mutated.
/// @param b Distinct second selected word, mutated.
/// @param c Distinct third selected word, mutated.
/// @param d Distinct fourth selected word, mutated.
inline void chacha_quarter(std::uint32_t& a,std::uint32_t& b,std::uint32_t& c,std::uint32_t& d) {
  chacha_quarter_core(a,b,c,d,nullptr);
}
/// @brief Read four bytes into a little-endian 32-bit word without signed shifts.
/// @param p Borrowed pointer to at least four readable bytes.
/// @return Decoded unsigned word.
inline std::uint32_t chacha_load32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0])|(static_cast<std::uint32_t>(p[1])<<8)|
      (static_cast<std::uint32_t>(p[2])<<16)|(static_cast<std::uint32_t>(p[3])<<24);
}
/// @brief Apply the actual 20-round permutation to a working state without feed-forward.
/// @param x Sixteen owned working words, mutated in place.
/// @param trace Optional record target, containing actual quarter-round and round snapshots.
inline void chacha_permute(std::array<std::uint32_t,16>& x,ChaChaBlockTrace* trace=nullptr) {
  /// @brief RFC column selections followed by diagonal selections, reused for ten pairs.
  static constexpr std::array<std::array<std::size_t,4>,8> schedule{{
      {{0,4,8,12}},{{1,5,9,13}},{{2,6,10,14}},{{3,7,11,15}},
      {{0,5,10,15}},{{1,6,11,12}},{{2,7,8,13}},{{3,4,9,14}}}};
  for(std::size_t round=0;round<20;++round) {
    for(std::size_t q=0;q<4;++q) {
      const auto& indices=schedule[(round%2)*4+q];ChaChaQuarterStep step;
      if(trace) {step.round=round;step.indices=indices;for(std::size_t j=0;j<4;++j)step.before[j]=x[indices[j]];}
      chacha_quarter(x[indices[0]],x[indices[1]],x[indices[2]],x[indices[3]]);
      if(trace) {for(std::size_t j=0;j<4;++j)step.after[j]=x[indices[j]];trace->quarter_rounds.push_back(step);}
    }
    if(trace)trace->round_states[round]=x;
  }
}
/// @brief Execute one actual RFC block, optionally recording setup/permutation/feed-forward.
/// @param key Borrowed 32-byte key.
/// @param counter Exact block counter; no increment occurs in this function.
/// @param nonce Borrowed 12-byte nonce.
/// @param trace Optional record target; null omits provenance storage.
/// @return Owned 64-byte little-endian keystream block.
inline std::array<std::uint8_t,64> chacha_block_core(const std::array<std::uint8_t,32>& key,
    std::uint32_t counter,const std::array<std::uint8_t,12>& nonce,ChaChaBlockTrace* trace) {
  std::array<std::uint32_t,16> initial{};
  initial[0]=0x61707865U;initial[1]=0x3320646eU;initial[2]=0x79622d32U;initial[3]=0x6b206574U;
  for(std::size_t i=0;i<8;++i)initial[4+i]=chacha_load32(key.data()+4*i);
  initial[12]=counter;for(std::size_t i=0;i<3;++i)initial[13+i]=chacha_load32(nonce.data()+4*i);
  if(trace) {trace->initial=initial;trace->quarter_rounds.clear();trace->quarter_rounds.reserve(80);}
  auto mixed=initial;chacha_permute(mixed,trace);std::array<std::uint8_t,64> output{};
  for(std::size_t i=0;i<16;++i) {
    const std::uint32_t value=mixed[i]+initial[i];if(trace)trace->feed_forward[i]=value;
    for(std::size_t byte=0;byte<4;++byte)output[4*i+byte]=static_cast<std::uint8_t>(value>>(8*byte));
  }
  if(trace)trace->output=output;return output;
}
/// @brief Shared complete-range-checked one-shot XOR, with optional actual block records.
/// @param key Borrowed 32-byte key, unchanged.
/// @param counter Initial 32-bit block counter.
/// @param nonce Borrowed 12-byte nonce, unchanged.
/// @param data Borrowed binary input, unchanged.
/// @param trace Optional actual block/output record target.
/// @param max_blocks Whole-call block budget for tracing; zero means unlimited.
/// @return Owned binary output of the same length.
/// @throws std::length_error Counter range or a positive block budget would be exceeded.
inline std::string chacha_encrypt_core(const std::array<std::uint8_t,32>& key,std::uint32_t counter,
    const std::array<std::uint8_t,12>& nonce,const std::string& data,ChaChaEncryptTrace* trace,std::size_t max_blocks) {
  if(data.size()>chacha20_capacity_bytes(counter))throw std::length_error("ChaCha20: counter range exhausted");
  const auto blocks=chacha20_blocks_required(counter,static_cast<std::uint64_t>(data.size()));
  if(max_blocks && blocks>max_blocks)throw std::length_error("ChaCha20: block budget exceeded");
  if(trace){trace->blocks.clear();trace->blocks.reserve(static_cast<std::size_t>(blocks));}
  std::string out=data;std::uint64_t next_counter=counter;
  for(std::size_t offset=0;offset<out.size();) {
    const auto used=std::min<std::size_t>(64,out.size()-offset);
    const auto keystream=chacha_block_core(key,static_cast<std::uint32_t>(next_counter),nonce,nullptr);
    for(std::size_t j=0;j<used;++j)out[offset+j]=static_cast<char>(static_cast<std::uint8_t>(out[offset+j])^keystream[j]);
    if(trace)trace->blocks.push_back({static_cast<std::uint32_t>(next_counter),offset,used,keystream});
    offset+=used;++next_counter;
  }
  if(trace)trace->output=out;return out;
}
} // namespace detail
/// @brief Apply the RFC 8439 quarter round to four owned unsigned words.
/// Four add/xor/rotate groups use rotations 16,12,8,7 and addition modulo 2^32.
/// All inputs are valid. Constant work/storage, no allocation or shared state;
/// returned words own their values and input is unchanged.
/// @param words Borrowed a,b,c,d words in that order.
/// @return Owned actual quarter-round output.
/// @par Example
/// @code
/// const auto q=dm::algorithms::chacha20_quarter_round({0x11111111,0x01020304,0x9b8d6f43,0x01234567});
/// std::print("{:08x}\n",q[3]); // 5881c4bb
/// @endcode
/// @par References
/// - RFC 8439, Sections 2.1 and 2.1.1. https://www.rfc-editor.org/rfc/rfc8439.html
/// @see chacha20_quarter_trace()
inline std::array<std::uint32_t,4> chacha20_quarter_round(const std::array<std::uint32_t,4>& words) {
  auto out=words;detail::chacha_quarter(out[0],out[1],out[2],out[3]);return out;
}
/// @brief Execute one quarter round and retain each actual compound ARX stage.
/// Same numeric kernel as ordinary quarter rounds, without replay. Owns input,
/// four intermediate states and output; no allocation/shared state or erasure.
/// @param words Borrowed a,b,c,d words, all values permitted.
/// @return Owned actual quarter-round trace; may contain key-derived material.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chacha20_quarter_trace({0x11111111,0x01020304,0x9b8d6f43,0x01234567});
/// std::print("{:08x}\n",t.output[1]); // cb1cf8ce
/// @endcode
/// @par References
/// - RFC 8439, Section 2.1.1. https://www.rfc-editor.org/rfc/rfc8439.html
/// @see ChaChaQuarterTrace, chacha20_quarter_round()
inline ChaChaQuarterTrace chacha20_quarter_trace(const std::array<std::uint32_t,4>& words) {
  ChaChaQuarterTrace trace;auto out=words;detail::chacha_quarter_core(out[0],out[1],out[2],out[3],&trace);return trace;
}
/// @brief Produce one RFC 8439 64-byte ChaCha20 block with the original signature.
/// Byte-oriented 256-bit key and 96-bit nonce, both loaded little-endian; one exact
/// 32-bit counter. Executes 20 rounds/80 quarter rounds, adds the original state and
/// serializes little-endian. Stateless deterministic output; inputs unchanged, owned
/// result, constant work/storage and no trace allocation. No authentication, nonce
/// reuse tracking, memory erasure or audited constant-time guarantee is supplied.
/// @param key Borrowed 32-byte key.
/// @param counter Exact 32-bit block counter, including UINT32_MAX; no increment here.
/// @param nonce Borrowed 12-byte nonce; callers manage uniqueness with the key.
/// @return Owned 64-byte keystream block.
/// @par Example
/// @code
/// const auto b=dm::algorithms::chacha20_block({},0,{});
/// std::print("{:02x}{:02x}{:02x}{:02x}\n",b[0],b[1],b[2],b[3]); // 76b8e0ad
/// @endcode
/// @par References
/// - RFC 8439, Sections 2.3 and 2.3.2. https://www.rfc-editor.org/rfc/rfc8439.html
/// @see chacha20_block_trace(), chacha20_encrypt()
inline std::array<std::uint8_t,64> chacha20_block(const std::array<std::uint8_t,32>& key,
    std::uint32_t counter,const std::array<std::uint8_t,12>& nonce) {
  return detail::chacha_block_core(key,counter,nonce,nullptr);
}
/// @brief Execute a complete block once and retain actual states, selections and serialization.
/// Same kernel as chacha20_block(); no permutation is replayed. Fixed state snapshots
/// plus 80 quarter-round records use O(1) storage per block. initial exposes the key;
/// all state/keystream records own their data and have no automatic erasure.
/// @param key Borrowed 32-byte key, normally public test material for exposed traces.
/// @param counter Exact 32-bit block counter.
/// @param nonce Borrowed 12-byte nonce, unchanged.
/// @return Owned actual block trace; default-constructing a record alone executes nothing.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chacha20_block_trace({},0,{});
/// std::print("{} {:08x}\n",t.quarter_rounds.size(),t.initial[0]); // 80 61707865
/// @endcode
/// @par References
/// - RFC 8439, Sections 2.2 and 2.3. https://www.rfc-editor.org/rfc/rfc8439.html
/// @see ChaChaBlockTrace, chacha20_block()
inline ChaChaBlockTrace chacha20_block_trace(const std::array<std::uint8_t,32>& key,std::uint32_t counter,
    const std::array<std::uint8_t,12>& nonce) {
  ChaChaBlockTrace trace;detail::chacha_block_core(key,counter,nonce,&trace);return trace;
}
/// @brief One-shot RFC 8439 XOR encryption/decryption, rejecting complete-call counter wrap.
/// Original signature and valid-range bytes are preserved. Each call starts at byte
/// zero of counter; an unused final-block tail is discarded. Arbitrary chunk splits
/// therefore do not automatically preserve stream position. Binary strings retain
/// embedded NUL/high-bit bytes. Empty returns empty without a block. Checks the whole
/// range before copying/processing; no partial result on failure and inputs unchanged.
/// O(n) work/output storage, O(1) block scratch; ordinary calls allocate no trace records.
/// This raw cipher supplies no authentication, nonce tracking, erasure or audited
/// constant-time guarantee. The same operation decrypts by XOR cancellation.
/// @param key Borrowed 32-byte key.
/// @param counter Starting 32-bit block counter.
/// @param nonce Borrowed 12-byte nonce; callers manage key/nonce/range reuse.
/// @param data Borrowed binary plaintext or ciphertext, unchanged.
/// @return Owned binary output, exactly data.size() bytes.
/// @throws std::length_error Required counter range extends beyond UINT32_MAX.
/// @par Example
/// @code
/// const auto encrypted=dm::algorithms::chacha20_encrypt({},1,{},"hello");
/// std::print("{}\n",dm::algorithms::chacha20_encrypt({},1,{},encrypted)); // hello
/// @endcode
/// @par References
/// - RFC 8439, Section 2.4. https://www.rfc-editor.org/rfc/rfc8439.html
/// @see chacha20_encrypt_trace(), chacha20_blocks_required()
inline std::string chacha20_encrypt(const std::array<std::uint8_t,32>& key,std::uint32_t counter,
    const std::array<std::uint8_t,12>& nonce,const std::string& data) {
  return detail::chacha_encrypt_core(key,counter,nonce,data,nullptr,0);
}
/// @brief Execute checked one-shot XOR once and retain actual block-use records.
/// Same kernel and binary output as chacha20_encrypt(), without replay. A positive
/// max_blocks cap rejects the whole call before allocation/block work. Records own
/// full keystreams, including unused tails, and output; do not erase themselves.
/// O(n) work and O(n+64*ceil(n/64)) owned trace storage, no shared mutable state.
/// @param key Borrowed 32-byte key; use public test material in exposed diagnostics.
/// @param counter Starting block counter, complete range checked before copying input.
/// @param nonce Borrowed 12-byte nonce, unchanged.
/// @param data Borrowed binary input, unchanged; empty creates no block records.
/// @param max_blocks Whole-call block cap; zero (default) means unlimited.
/// @return Owned output and actual counter/offset/used-byte/keystream records.
/// @throws std::length_error Counter range or positive block cap would be exceeded.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chacha20_encrypt_trace({},1,{},std::string(65,'x'),2);
/// std::print("{} {}\n",t.blocks[0].counter,t.blocks[1].counter); // 1 2
/// @endcode
/// @par References
/// - RFC 8439, Section 2.4. https://www.rfc-editor.org/rfc/rfc8439.html
/// @see ChaChaEncryptTrace, chacha20_encrypt()
inline ChaChaEncryptTrace chacha20_encrypt_trace(const std::array<std::uint8_t,32>& key,std::uint32_t counter,
    const std::array<std::uint8_t,12>& nonce,const std::string& data,std::size_t max_blocks=0) {
  ChaChaEncryptTrace trace;detail::chacha_encrypt_core(key,counter,nonce,data,&trace,max_blocks);return trace;
}
} // namespace datamunge::algorithms
