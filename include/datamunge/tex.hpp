#pragma once

/// @file tex.hpp
/// @brief Owned TeX fragments, fluent mathematical composition and document delimiters.

#include <ostream>
#include <string>
#include <string_view>

namespace datamunge {

/// @brief An owned TeX fragment with nonmutating mathematical composition helpers.
///
/// Construction copies raw TeX bytes, including embedded NULs; it does not parse,
/// validate or escape them. Methods return independent values and preserve the
/// source. Labels and delimiters are raw TeX too; text() escapes literal prose.
/// Compose in math mode, then call display() or inline_math() at the document
/// boundary. Requires amsmath for annotations, text and matrix environments.
/// Const operations may run concurrently; ordinary C++ rules apply to assignment.
/// Allocation failures propagate as std::bad_alloc throughout this value type.
///
/// @par Example
/// @code
/// const auto x = dm::Tex("x").underset("i").superscript("2");
/// std::print("{}\n", x.display());
/// @endcode
/// @par References
/// - American Mathematical Society, User's Guide for the amsmath Package, sections 4.10-4.14:
///   https://www.latex-project.org/help/documentation/amsldoc.pdf
/// @see Mat, Tex::text
class Tex {
 public:
  /// @brief Copy a raw TeX fragment into owned storage.
  /// @param source Raw TeX bytes; default empty. No reference to the argument is retained.
  /// @par Example
  /// @code
  /// const dm::Tex x("x+y");
  /// std::print("{}\n", x); // x+y
  /// @endcode
  explicit Tex(std::string_view source = {}) : source_(source) {}

  /// @brief Access the underlying TeX bytes without copying.
  /// @return Reference valid until this object is assigned to or destroyed.
  const std::string& str() const noexcept { return source_; }

  /// @brief Copy the raw bytes for existing std::string consumers.
  /// @return Independent string, without added math-mode delimiters.
  /// @par Example
  /// @code
  /// const std::string source = dm::Tex("x");
  /// std::print("{}\n", source); // x
  /// @endcode
  operator std::string() const { return source_; }

  /// @brief Borrow the raw bytes as a view for composing with another Tex value.
  /// @return View valid until this object is assigned to or destroyed; a view of a
  ///         temporary must be consumed within the same full expression.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x").underbrace(dm::Tex::text("fit_1")));
  /// @endcode
  operator std::string_view() const noexcept { return source_; }

  /// @brief Test whether the raw fragment contains no bytes.
  /// @return True for an empty fragment.
  bool empty() const noexcept { return source_.empty(); }

  /// @brief Count raw bytes, rather than mathematical symbols or Unicode characters.
  /// @return Number of stored bytes.
  std::size_t size() const noexcept { return source_.size(); }

  /// @brief Access a NUL-terminated representation of the raw bytes.
  /// @return Pointer valid until this object is assigned to or destroyed.
  const char* c_str() const noexcept { return source_.c_str(); }

  /// @brief Iterate over the raw TeX bytes without mutation.
  /// @return Iterator to the first byte, valid until assignment or destruction.
  std::string::const_iterator begin() const noexcept { return source_.begin(); }

  /// @brief Obtain the end of the raw TeX bytes.
  /// @return Past-the-end iterator, valid until assignment or destruction.
  std::string::const_iterator end() const noexcept { return source_.end(); }

  /// @brief Find a raw substring, without interpreting TeX commands.
  /// @param needle Bytes to search for; an empty needle matches at pos when pos <= size().
  /// @param pos Starting byte offset, default zero.
  /// @return First matching offset, or std::string::npos if absent.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x+y").find("y")); // 2
  /// @endcode
  std::size_t find(std::string_view needle, std::size_t pos = 0) const noexcept {
    return source_.find(needle, pos);
  }

  /// @brief Place a raw math label below the entire fragment.
  /// @param label Raw TeX label, including an empty label; copied into the result.
  /// @return Independent annotated fragment; this object is unchanged.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x").underset("i"));
  /// @endcode
  Tex underset(std::string_view label) const {
    return Tex("\\underset{" + std::string(label) + "}{" + source_ + "}");
  }

  /// @brief Place a raw math label above the entire fragment.
  /// @param label Raw TeX label, including an empty label; copied into the result.
  /// @return Independent annotated fragment; this object is unchanged.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x").overset("n"));
  /// @endcode
  Tex overset(std::string_view label) const {
    return Tex("\\overset{" + std::string(label) + "}{" + source_ + "}");
  }

  /// @brief Place a brace and raw math label below the entire fragment.
  /// @param label Raw TeX label, including an empty label; copied into the result.
  /// @return Independent annotated fragment; this object is unchanged.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x").underbrace("n"));
  /// @endcode
  Tex underbrace(std::string_view label) const {
    return Tex("\\underbrace{" + source_ + "}_{" + std::string(label) + "}");
  }

  /// @brief Place a brace and raw math label above the entire fragment.
  /// @param label Raw TeX label, including an empty label; copied into the result.
  /// @return Independent annotated fragment; this object is unchanged.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x").overbrace("n"));
  /// @endcode
  Tex overbrace(std::string_view label) const {
    return Tex("\\overbrace{" + source_ + "}^{" + std::string(label) + "}");
  }

  /// @brief Attach a raw subscript to the entire grouped fragment.
  /// @param label Raw TeX label, including an empty label; copied into the result.
  /// @return Independent annotated fragment; this object is unchanged.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x").subscript("i"));
  /// @endcode
  Tex subscript(std::string_view label) const {
    return Tex("{" + source_ + "}_{" + std::string(label) + "}");
  }

  /// @brief Attach a raw superscript to the entire grouped fragment.
  /// @param label Raw TeX label, including an empty label; copied into the result.
  /// @return Independent annotated fragment; this object is unchanged.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x").superscript("2"));
  /// @endcode
  Tex superscript(std::string_view label) const {
    return Tex("{" + source_ + "}^{" + std::string(label) + "}");
  }

  /// @brief Use this entire fragment as the numerator of a fraction.
  /// @param denominator Raw TeX denominator, including an empty fragment.
  /// @return Independent fraction; TeX groups both numerator and denominator.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("a+b").frac("c+d"));
  /// @endcode
  Tex frac(std::string_view denominator) const {
    return Tex("\\frac{" + source_ + "}{" + std::string(denominator) + "}");
  }

  /// @brief Enclose the entire fragment in a square root.
  /// @return Independent square-root fragment, preserving the source.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x+y").sqrt());
  /// @endcode
  Tex sqrt() const { return Tex("\\sqrt{" + source_ + "}"); }

  /// @brief Enclose the fragment in automatically sized paired delimiters.
  /// @param left Raw TeX left delimiter, default "("; e.g. "[", "\\langle" or ".".
  /// @param right Raw TeX right delimiter, default ")"; e.g. "]", "\\rangle" or ".".
  /// @return Fragment wrapped in left/right commands. Delimiters are not validated;
  ///         the caller supplies a valid pair. Spaces separate command names from content.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x+y").delimit().superscript("-1"));
  /// @endcode
  Tex delimit(std::string_view left = "(", std::string_view right = ")") const {
    return Tex("\\left" + std::string(left) + " " + source_ + " \\right" + std::string(right));
  }

  /// @brief Wrap the completed fragment for Markdown/MEPML display mathematics.
  /// @return Independent string with double-dollar delimiters on their own lines.
  ///         Call once after composition; existing delimiters are not detected or removed.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x=1").display());
  /// @endcode
  std::string display() const { return "$$\n" + source_ + "\n$$"; }

  /// @brief Wrap the completed fragment for Markdown/MEPML inline mathematics.
  /// @return Independent string with single-dollar delimiters; call once after composition.
  ///         Existing delimiters are not detected or removed.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x").inline_math()); // $x$
  /// @endcode
  std::string inline_math() const { return "$" + source_ + "$"; }

  /// @brief Escape literal text and enclose it in an amsmath text command.
  /// Escapes backslash, braces, dollar, ampersand, hash, percent, underscore,
  /// caret and tilde. CR, LF and tab each become a space; other bytes (including
  /// UTF-8) are preserved. This is prose escaping, not a general TeX sanitizer.
  /// @param literal Literal prose, including an empty string; copied into the result.
  /// @return Independent text fragment usable inside mathematics.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex::text("fit_1 (95%)"));
  /// @endcode
  static Tex text(std::string_view literal) {
    std::string escaped;
    for (const char c : literal) {
      switch (c) {
        case '\\':
          escaped += "\\textbackslash{}";
          break;
        case '{':
        case '}':
        case '$':
        case '&':
        case '#':
        case '%':
        case '_':
          escaped += '\\';
          escaped += c;
          break;
        case '^':
          escaped += "\\textasciicircum{}";
          break;
        case '~':
          escaped += "\\textasciitilde{}";
          break;
        case '\n':
        case '\r':
        case '\t':
          escaped += ' ';
          break;
        default:
          escaped += c;
      }
    }
    return Tex("\\text{" + escaped + "}");
  }

  /// @brief Concatenate raw TeX fragments without adding whitespace or grouping.
  /// @param left First fragment.
  /// @param right Second fragment.
  /// @return Independent concatenation; both operands remain unchanged.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x") + " = " + dm::Tex("y")); // x = y
  /// @endcode
  friend Tex operator+(const Tex& left, const Tex& right) {
    return Tex(left.source_ + right.source_);
  }

  /// @brief Append raw TeX bytes; see the fragment-concatenation example.
  /// @param left First fragment.
  /// @param right Raw bytes to append, copied into the result.
  /// @return Independent concatenation, without added whitespace or grouping.
  friend Tex operator+(const Tex& left, std::string_view right) { return left + Tex(right); }

  /// @brief Prepend raw TeX bytes; see the fragment-concatenation example.
  /// @param left Raw bytes to prepend, copied into the result.
  /// @param right Second fragment.
  /// @return Independent concatenation, without added whitespace or grouping.
  friend Tex operator+(std::string_view left, const Tex& right) { return Tex(left) + right; }

  /// @brief Compare raw TeX bytes, without mathematical equivalence checking.
  /// @param left First fragment.
  /// @param right Second fragment.
  /// @return True if the stored bytes are identical.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::Tex("x") == dm::Tex("x")); // true
  /// @endcode
  friend bool operator==(const Tex& left, const Tex& right) noexcept {
    return left.source_ == right.source_;
  }

  /// @brief Compare a fragment to raw text; see the fragment-equality example.
  /// @param left Fragment.
  /// @param right Raw bytes to compare.
  /// @return True if the stored bytes equal right.
  friend bool operator==(const Tex& left, std::string_view right) noexcept {
    return left.source_ == right;
  }

  /// @brief Compare raw text to a fragment; see the fragment-equality example.
  /// @param left Raw bytes to compare.
  /// @param right Fragment.
  /// @return True if left equals the stored bytes.
  friend bool operator==(std::string_view left, const Tex& right) noexcept {
    return left == right.source_;
  }

  /// @brief Write raw TeX bytes to a stream, without delimiters or escaping.
  /// @param out Destination stream.
  /// @param value Fragment to write.
  /// @return Destination stream after writing; normal stream error rules apply.
  /// @par Example
  /// @code
  /// std::cout << dm::Tex("x"); // x
  /// @endcode
  friend std::ostream& operator<<(std::ostream& out, const Tex& value) {
    return out << value.source_;
  }

 private:
  std::string source_; ///< Owned raw bytes; no borrowed TeX or matrix storage.
};

} // namespace datamunge
