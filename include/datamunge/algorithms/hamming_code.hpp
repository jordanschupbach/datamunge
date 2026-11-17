#pragma once
/// @file hamming_code.hpp
/// @brief Checked binary Hamming(7,4) encoding, bounded-distance decoding and actual parity traces.
#include <array>
#include <cstddef>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Owned decoded data and the applied syndrome correction, without a recovery certificate.
/// corrected means a flip was applied. Only an at-most-one-bit channel error
/// assumption makes that flip correct; two errors are miscorrected. A zero
/// syndrome can also hide three or more errors. No input references retained.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hamming74_decode(dm::algorithms::hamming74_encode({1,0,1,1}));
/// std::print("{} {}\n",r.error_position,r.corrected); // 0 false
/// @endcode
/// @see hamming74_decode(), Hamming74Trace
struct Hamming74Decoded {
  std::array<int, 4> data; ///< Extracted positions3,5,6,7 after any applied flip.
  int  error_position;     ///< Nonzero syndrome1..7 interpreted as a flip location, otherwise0.
  bool corrected; ///< Whether decoder applied a flip, not whether original data was recovered.
};
/// @brief Owned actual inputs and running XORs of one even-parity check.
/// Positions are one-based. Prefix0 is the first input bit; prefix1..3 use
/// exactly three producing XORs. No replay, heap storage or borrowed state.
/// @par Example
/// @code
/// const dm::algorithms::Hamming74Parity p;
/// std::print("{}\n",p.prefix.size()); // 4
/// @endcode
/// @see Hamming74Trace
struct Hamming74Parity {
  std::array<int, 4> positions{}; ///< Actual visited one-based positions in order.
  std::array<int, 4> bits{};      ///< Actual bits read at those positions.
  std::array<int, 4> prefix{};    ///< Actual sequential XOR prefix values, last is check output.
};
/// @brief Selected actual decoder visits and XORs, excluding validation/capture/assembly.
/// Fixed-size input uses12 parity bit visits,9 parity XORs,0or1 correction
/// flips and4 final data reads. These counts are not timing or total machine work.
/// @par Example
/// @code
/// const dm::algorithms::Hamming74Work w;
/// std::print("{}\n",w.parity_xors); // 0
/// @endcode
/// @see Hamming74Trace
struct Hamming74Work {
  std::size_t parity_bit_visits{0}; ///< Producing parity input reads,12.
  std::size_t parity_xors{0};       ///< Producing sequential check XORs,9.
  std::size_t flips{0};             ///< Applied correcting-bit XORs,0or1.
  std::size_t data_reads{0};        ///< Extracted data positions,4.
};
/// @brief Owned fixed-size actual parity checks, syndrome and correction from one decoding.
/// All arrays own values; copying is deep. No heap, optional history, replay or
/// truncation is needed for this fixed7-bit/3-check trace. Work excludes snapshot
/// copies, validation and weighted syndrome assembly. Input remains unchanged.
/// @par Example
/// @code
/// auto c=dm::algorithms::hamming74_encode({1,0,1,1});c[5]^=1;
/// const auto t=dm::algorithms::hamming74_decode_trace(c);
/// std::print("{} {}\n",t.syndrome,t.work.parity_xors); // 6 9
/// @endcode
/// @see hamming74_decode_trace()
struct Hamming74Trace {
  std::array<int, 7>             received{};  ///< Actual binary decoder input.
  std::array<Hamming74Parity, 3> checks{};    ///< Actual p1,p2,p4 input/XOR records.
  int                            syndrome{0}; ///< Actual p1+2*p2+4*p4.
  std::array<int, 7>
                   corrected_word{}; ///< Actual post-flip word, always a codeword for binary input.
  Hamming74Decoded decoded{}; ///< Actual legacy result, not independent correctness verification.
  Hamming74Work    work;      ///< Selected producing decoder work.
};
/// @brief Private validation and shared bounded-distance decoding arithmetic.
namespace hamming74_detail {
/// @brief Validate every component before parity arithmetic or indexing.
/// @tparam N Fixed input length.
/// @param bits Borrowed values, each must be0or1.
/// @throws std::invalid_argument On any nonbinary component, before mutation.
template<std::size_t N>
inline void validate(const std::array<int, N>& bits) {
  for (int b : bits)
    if (b != 0 && b != 1)
      throw std::invalid_argument("Hamming(7,4) requires binary components");
}
/// @brief Compute actual sequential checks once with optional same-pass capture.
/// @param c Borrowed validated seven binary values.
/// @param trace Optional borrowed output record.
/// @return Weighted syndrome0..7.
inline int syndrome(const std::array<int, 7>& c, Hamming74Trace* trace) {
  constexpr std::array<std::array<int, 4>, 3> rows{{{1, 3, 5, 7}, {2, 3, 6, 7}, {4, 5, 6, 7}}};
  int                                         s = 0;
  for (int r = 0; r < 3; ++r) {
    int parity = 0;
    for (int j = 0; j < 4; ++j) {
      const int bit = c[rows[r][j] - 1];
      if (j)
        parity ^= bit;
      else
        parity = bit;
      if (trace) {
        trace->checks[r].positions[j] = rows[r][j];
        trace->checks[r].bits[j]      = bit;
        trace->checks[r].prefix[j]    = parity;
      }
    }
    s += parity * (1 << r);
  }
  if (trace) {
    trace->work.parity_bit_visits = 12;
    trace->work.parity_xors       = 9;
    trace->syndrome               = s;
  }
  return s;
}
/// @brief Decode a copied binary input through the common actual check/flip/extract path.
/// @param c Owned seven input values, validated before all operations.
/// @param trace Optional borrowed initially empty capture sink.
/// @return Owned historical result, with applied-flip semantics.
/// @throws std::invalid_argument On nonbinary input before indexing.
inline Hamming74Decoded decode(std::array<int, 7> c, Hamming74Trace* trace) {
  validate(c);
  if (trace)
    trace->received = c;
  const int s = syndrome(c, trace);
  if (s)
    c[s - 1] ^= 1;
  Hamming74Decoded out{{c[2], c[4], c[5], c[6]}, s, s != 0};
  if (trace) {
    trace->corrected_word  = c;
    trace->decoded         = out;
    trace->work.flips      = s != 0;
    trace->work.data_reads = 4;
  }
  return out;
}
} // namespace hamming74_detail
/// @brief Encode four binary data values in even-parity positions1..7.
/// Layout is p1,p2,d1,p4,d2,d3,d4, with parity positions1,2,4. The array
/// order is position order, independent of integer endianness. Six parity XORs;
/// constant fixed work/storage, no heap or mutation. Independent calls share no state.
/// @param d Borrowed four binary values0or1 in d1,d2,d3,d4 order.
/// @return Owned binary seven-value codeword with zero syndrome.
/// @throws std::invalid_argument On any nonbinary input value.
/// @par Example
/// @code
/// const auto c=dm::algorithms::hamming74_encode({1,0,1,1});
/// std::print("{}{}{}{}{}{}{}\n",c[0],c[1],c[2],c[3],c[4],c[5],c[6]); // 0110011
/// @endcode
/// @par References
/// - R. W. Hamming (1950), Error Detecting and Error Correcting Codes, Bell System
///   Technical Journal29,147–160, https://doi.org/10.1002/j.1538-7305.1950.tb00463.x .
/// @see hamming74_decode(), hamming74_syndrome()
inline std::array<int, 7> hamming74_encode(const std::array<int, 4>& d) {
  hamming74_detail::validate(d);
  std::array<int, 7> c{};
  c[2] = d[0];
  c[4] = d[1];
  c[5] = d[2];
  c[6] = d[3];
  c[0] = c[2] ^ c[4] ^ c[6];
  c[1] = c[2] ^ c[5] ^ c[6];
  c[3] = c[4] ^ c[5] ^ c[6];
  return c;
}
/// @brief Apply the syndrome flip and extract data, assuming at most one channel error for recovery.
/// Every binary received word lies within one bit of exactly one codeword. The
/// decoder always selects it; it cannot establish that it was transmitted.
/// Nonzero syndrome detects any two flipped bits, but this correcting interface
/// then miscorrects them. Zero syndrome only establishes code membership.
/// Constant fixed work/storage, no heap, no retained references or shared state.
/// @param c Owned copy of seven binary received values, in position order1..7.
/// @return Owned data and applied-flip summary; input argument unchanged at caller.
/// @throws std::invalid_argument On any nonbinary component before correction indexing.
/// @par Example
/// @code
/// auto c=dm::algorithms::hamming74_encode({1,0,1,1});c[5]^=1;
/// const auto r=dm::algorithms::hamming74_decode(c);
/// std::print("{} {}\n",r.error_position,r.corrected); // 6 true
/// @endcode
/// @par References
/// - Hamming (1950), https://doi.org/10.1002/j.1538-7305.1950.tb00463.x .
/// @see hamming74_decode_trace(), hamming74_syndrome()
inline Hamming74Decoded hamming74_decode(std::array<int, 7> c) {
  return hamming74_detail::decode(c, nullptr);
}
/// @brief Compute the three binary parity checks without applying a correction.
/// Nonzero means nonmembership, not a count of errors. Zero permits undetected
/// codeword-shaped errors of weights3,4or7. Actual12 parity visits/9XORs; no heap,
/// mutation, retained references or shared state.
/// @param c Borrowed seven binary values, in position order1..7.
/// @return Weighted check sum0..7.
/// @throws std::invalid_argument On any nonbinary component.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hamming74_syndrome({1,0,0,0,0,0,0})); // 1
/// @endcode
/// @see hamming74_decode(), Hamming74Parity
inline int hamming74_syndrome(const std::array<int, 7>& c) {
  hamming74_detail::validate(c);
  return hamming74_detail::syndrome(c, nullptr);
}
/// @brief Capture actual parity prefixes and the applied flip from the shared decoder execution.
/// Same binary domain and correction limits as hamming74_decode(). Fixed-size
/// owned records require no history policy cap; no replay, callbacks or input mutation.
/// @param c Borrowed seven received binary values in position order.
/// @return Owned producing records, corrected word, legacy result and selected work.
/// @throws std::invalid_argument On nonbinary input before any correction indexing.
/// @par Example
/// @code
/// auto c=dm::algorithms::hamming74_encode({1,0,1,1});c[0]^=1;c[1]^=1;
/// const auto t=dm::algorithms::hamming74_decode_trace(c);
/// std::print("{} {}\n",t.syndrome,t.decoded.corrected); // 3 true
/// @endcode
/// @see Hamming74Trace, hamming74_decode()
inline Hamming74Trace hamming74_decode_trace(const std::array<int, 7>& c) {
  Hamming74Trace t;
  (void)hamming74_detail::decode(c, &t);
  return t;
}
} // namespace datamunge::algorithms
