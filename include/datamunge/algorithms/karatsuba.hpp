#pragma once
/// @file karatsuba.hpp
/// @brief Checked unsigned-decimal Karatsuba multiplication and actual recursive call trees.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Optional complete input-digit and actual recursion ceilings.
/// Zero disables an optional cap, not size arithmetic. Depth counts root0; a
/// positive ceiling is the largest admitted depth, while0 disables that ceiling.
/// Caps count digits/calls/depth, excluding capacities/temporary vectors/bytes/RAM.
/// @par Example
/// @code
/// const dm::algorithms::KaratsubaOptions o{100,100,10};std::print("{}\n",o.max_calls); // 100
/// @endcode
/// @see karatsuba_trace()
struct KaratsubaOptions {
  std::size_t max_input_digits{
      1000000}; ///< Complete original a.size()+b.size() cap, million default, zero off.
  std::size_t max_calls{1000000}; ///< Actual recursion-entry ceiling, million default, zero off.
  std::size_t max_depth{
      1024}; ///< Largest allowed positive recursion depth, default1024, zero disables cap.
};
/// @brief Role of an actual multiplication invocation in its producing parent's three products.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::karatsuba_trace("2","3").calls[0].role==dm::algorithms::KaratsubaRole::Root); // true
/// @endcode
/// @see KaratsubaCall
enum class KaratsubaRole {
  Root, ///< Original multiplication invocation.
  Low,  ///< Actual low-half product z0.
  High, ///< Actual high-half product z2.
  Sum   ///< Actual product of half sums, before subtracting z2 and z0.
};
/// @brief Actual digit-vector operand lengths, split and result length for one invocation.
/// Calls are entered preorder; parent is an earlier record except root's sentinel
/// parent=size_t max. Low/high/sum roles preserve actual call order. The base case
/// is min(left_digits,right_digits)<=32, Root zero has one digit; recursive zero slices can retain their original widths.
/// No operands/products are re-multiplied to construct the tree.
/// @par Example
/// @code
/// const auto t=dm::algorithms::karatsuba_trace("12","34");std::print("{} {}\n",t.calls[0].leaf,t.calls[0].product_digits); // true 3
/// @endcode
/// @see KaratsubaTrace
struct KaratsubaCall {
  std::size_t   parent{0};                 ///< Actual parent entry id, size_t max on root.
  KaratsubaRole role{KaratsubaRole::Root}; ///< Actual parent-product role.
  std::size_t   depth{0};                  ///< Actual depth, root0.
  std::size_t   left_digits{
      0}; ///< Actual little-endian left vector size; recursive slices can retain high zero digits.
  std::size_t right_digits{
      0}; ///< Actual little-endian right vector size; recursive slices can retain high zero digits.
  std::size_t split{0};          ///< Actual decimal split exponent min(lengths)/2, zero for a leaf.
  bool        leaf{false};       ///< Whether schoolbook multiplication actually executed.
  std::size_t product_digits{0}; ///< Actual normalized returned product length.
};
/// @brief Selected actual recursive calls and schoolbook digit products, not total arithmetic/time.
/// Excludes validation, digit additions/subtractions, carries, shifts, copies,
/// allocations and formatting. A zero operand still runs the actual one-digit leaf.
/// @par Example
/// @code
/// const auto t=dm::algorithms::karatsuba_trace("12","34");std::print("{} {}\n",t.work.schoolbook_leaves,t.work.digit_products); // 1 4
/// @endcode
/// @see KaratsubaTrace
struct KaratsubaWork {
  std::size_t recursive_calls{0};   ///< Actual admitted multiplication invocations.
  std::size_t schoolbook_leaves{0}; ///< Actual base-case invocations.
  std::size_t digit_products{0};    ///< Actual leaf digit-pair multiplications.
  std::size_t max_depth{0};         ///< Maximum actual admitted depth, root0.
};
/// @brief Owned original operands, actual decimal result and producing recursive call records.
/// Complete original bytes are owned, including leading zeros/empty conventions.
/// Recursion records store lengths/roles rather than full intermediate big integers.
/// O(calls) extra history in addition to kernel digit temporaries and owned input/output.
/// @par Example
/// @code
/// const auto t=dm::algorithms::karatsuba_trace("0012","34");std::print("{} {}\n",t.product,t.calls[0].left_digits); // 408 2
/// @endcode
/// @see karatsuba_trace()
struct KaratsubaTrace {
  std::string   left;    ///< Owned original left decimal byte string.
  std::string   right;   ///< Owned original right decimal byte string.
  std::string   product; ///< Actual normalized unsigned decimal product, zero as"0".
  KaratsubaWork work;    ///< Actual selected operation counts.
  std::vector<KaratsubaCall>
      calls; ///< Actual preorder entries, completed with producing result lengths.
};
/// @brief Multiply unsigned decimal strings with the original crossover32 Karatsuba kernel.
/// Accepts only ASCII digits, with leading zeros normalized; empty is retained as
/// legacy zero. Rejects signs/space/nondecimal bytes. Uses little-endian decimal
/// vectors, splitting at min(lengths)/2; leaf if either length<=32. Computes low,
/// high and sum products, then combines z2*10^(2m)+(zsum-z2-z0)*10^m+z0.
/// Balanced n-digit recurrence gives O(n^log2(3)) arithmetic asymptotically;
/// unequal sizes, crossover/carries and copies must be accounted separately.
/// Conservative live digit storage O((n+m)*log(min(n,m)+2)); no shared state or
/// caller mutation. Legacy optional caps disabled, arithmetic/counters checked.
/// @param a Borrowed unsigned decimal digits, possibly empty/leading-zero.
/// @param b Borrowed unsigned decimal digits, possibly empty/leading-zero.
/// @return Owned normalized exact unsigned decimal product.
/// @throws std::invalid_argument On any nondecimal input byte.
/// @throws std::length_error On shape/counter/container representability overflow.
/// @throws std::bad_alloc On digit/output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::karatsuba_multiply("12","34")); // 408
/// @endcode
/// @par References
/// - Karatsuba and Ofman(1962), Dokl.Akad.NaukSSSR145(2),293-294, https://mi.mathnet.ru/dan26729 .
/// @see karatsuba_trace()
std::string karatsuba_multiply(const std::string& a, const std::string& b);
/// @brief Capture the actual recursion tree and selected leaf work from one multiplication.
/// Entire decimal syntax and complete original byte count are checked before
/// normalized digit/owned input copies. Actual calls/depth are admitted at entry,
/// after any caller-created argument temporaries but before that call's body.
/// Failure exposes no partial trace; records never replay multiplication.
/// @param a Borrowed unsigned decimal string, empty convention as scalar API.
/// @param b Borrowed unsigned decimal string.
/// @param options Complete input and actual recursion-entry/depth ceilings.
/// @return Owned original operands, exact product and actual call tree/work.
/// @throws std::invalid_argument On nondecimal input bytes.
/// @throws std::length_error On enabled ceilings or shape/counter/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::karatsuba_trace("0","123").product); // 0
/// @endcode
/// @see KaratsubaCall, KaratsubaOptions
KaratsubaTrace
karatsuba_trace(const std::string& a, const std::string& b, KaratsubaOptions options = {});
} // namespace datamunge::algorithms
