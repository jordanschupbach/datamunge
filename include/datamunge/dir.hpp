#pragma once

/// @file dir.hpp
/// @brief `dm::dir(x)`: list an object's public methods and fields, like Python's `dir()`.
///
/// C++ has no runtime reflection, so the listing comes from a table generated
/// from the library's headers (include/datamunge/dir_registry.hpp, rebuilt by
/// `just generate-dir`). It covers every public class in namespace datamunge,
/// with inherited members included; other types report that no listing exists.
///
/// @code
/// const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
/// std::print("{}\n", dm::dir(model));          // grouped signatures with one-line docs
/// for (const auto& name : dm::dir(model).names()) std::print("{} ", name);  // Python-style names
/// @endcode

#include <datamunge/dir_registry.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge {

/// @brief One public member in a dir() listing.
struct DirMember {
  /// @brief What sort of member it is; the listing groups members by kind in this order.
  enum class Kind {
    Constructor,   ///< A constructor of the class.
    Method,        ///< A non-static member function.
    StaticMethod,  ///< A static member function.
    Field,         ///< A non-static data member.
    StaticField,   ///< A static data member or constant.
    Type,          ///< A member type alias or enum.
  };

  Kind kind = Kind::Method;  ///< The member's kind.
  std::string name;          ///< The member's name, e.g. "coefficients".
  std::string signature;     ///< Its declaration, e.g. "coefficients() const -> const std::vector<double>&".
  std::string brief;         ///< Its one-line description (empty when undocumented).
};

/// @brief The result of dir(): a class's public members, printable and queryable.
class DirListing {
 public:
  /// @brief Builds a listing.
  /// @param type_name  The type as shown to the reader, e.g. "stats::LM".
  /// @param description  The class's one-line description (may be empty).
  /// @param members  Its public members, any order.
  /// @param found  Whether the type has a generated member table.
  DirListing(std::string type_name, std::string description, std::vector<DirMember> members, bool found)
      : type_name_(std::move(type_name)),
        description_(std::move(description)),
        members_(std::move(members)),
        found_(found) {}

  /// @brief The type as shown to the reader, e.g. "stats::LM" or "linalg::DenseMatrix<double>".
  /// @return The type's name without the root namespace.
  [[nodiscard]] const std::string& type_name() const noexcept { return type_name_; }

  /// @brief The class's one-line description.
  /// @return The description, or an empty string when the class is undocumented.
  [[nodiscard]] const std::string& description() const noexcept { return description_; }

  /// @brief Every public member, inherited ones included, sorted by name.
  /// @return The members (empty when found() is false).
  [[nodiscard]] const std::vector<DirMember>& members() const noexcept { return members_; }

  /// @brief Whether the type has a generated member table (only datamunge's own classes do).
  /// @return True when the listing came from the registry.
  [[nodiscard]] bool found() const noexcept { return found_; }

  /// @brief The member names, sorted and de-duplicated: exactly what Python's `dir()` returns.
  /// @return One entry per distinct name (overloads collapse into one).
  [[nodiscard]] std::vector<std::string> names() const {
    std::vector<std::string> out;
    for (const DirMember& member : members_) out.push_back(member.name);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

  /// @brief The members of one kind.
  /// @param kind  Which kind to keep.
  /// @return Those members, in name order.
  [[nodiscard]] std::vector<DirMember> of_kind(DirMember::Kind kind) const {
    std::vector<DirMember> out;
    for (const DirMember& member : members_)
      if (member.kind == kind) out.push_back(member);
    return out;
  }

  /// @brief The printable listing: the type, its description, then members grouped by kind with docs.
  /// @return Multi-line text (what `std::print("{}", dm::dir(x))` shows).
  [[nodiscard]] std::string to_string() const {
    std::string out = type_name_;
    if (!description_.empty()) out += "\n  " + description_;
    if (!found_) {
      out += "\n  (no member listing: dm::dir() covers the datamunge library's own classes)";
      return out;
    }
    static constexpr std::pair<DirMember::Kind, const char*> kGroups[] = {
        {DirMember::Kind::Constructor, "constructors"}, {DirMember::Kind::Method, "methods"},
        {DirMember::Kind::StaticMethod, "static methods"}, {DirMember::Kind::Field, "fields"},
        {DirMember::Kind::StaticField, "static fields"}, {DirMember::Kind::Type, "types"},
    };
    for (const auto& [kind, heading] : kGroups) {
      const std::vector<DirMember> group = of_kind(kind);
      if (group.empty()) continue;
      out += "\n\n  " + std::string(heading) + " (" + std::to_string(group.size()) + "):";
      // Briefs line up in one column; a signature too long for it puts its brief on the next line.
      std::size_t column = 0;
      for (const DirMember& member : group)
        if (member.signature.size() <= kMaxAligned) column = std::max(column, member.signature.size());
      for (const DirMember& member : group) {
        out += "\n    " + member.signature;
        if (member.brief.empty()) continue;
        if (member.signature.size() <= column) {
          out += std::string(column - member.signature.size() + 2, ' ') + "// " + member.brief;
        } else {
          out += "\n    " + std::string(std::min(column, kMaxAligned) + 2, ' ') + "// " + member.brief;
        }
      }
    }
    if (members_.empty()) out += "\n  (no public members)";
    return out;
  }

 private:
  /// @brief Widest signature that still gets its brief on the same line.
  static constexpr std::size_t kMaxAligned = 56;

  std::string type_name_;          ///< The type as shown to the reader.
  std::string description_;        ///< The class's one-line description.
  std::vector<DirMember> members_; ///< Public members, sorted by name.
  bool found_;                     ///< Whether a generated table exists for the type.
};

/// @brief Implementation helpers for dir(); not part of the public API.
namespace dir_detail {

/// @brief The compiler's spelling of a type, e.g. "datamunge::stats::LM", via the function-signature macro.
/// @tparam T The type to name (cv-qualifiers and references already removed).
/// @return The type's fully qualified name, or an empty view on an unsupported compiler.
template <typename T>
std::string_view raw_type_name() {
#if defined(__clang__) || defined(__GNUC__)
  const std::string_view signature = __PRETTY_FUNCTION__;  // "... [with T = ns::Type; ...]" / "... [T = ns::Type]"
  const std::size_t start = signature.find("T = ");
  if (start == std::string_view::npos) return {};
  std::size_t end = signature.find(';', start);
  if (end == std::string_view::npos) end = signature.rfind(']');
  return signature.substr(start + 4, end - start - 4);
#elif defined(_MSC_VER)
  std::string_view signature = __FUNCSIG__;  // "... raw_type_name<class ns::Type>(void)"
  const std::size_t start = signature.find("raw_type_name<");
  const std::size_t end = signature.rfind(">(void)");
  if (start == std::string_view::npos || end == std::string_view::npos) return {};
  signature = signature.substr(start + 14, end - start - 14);
  for (const std::string_view tag : {"class ", "struct ", "enum "})
    if (signature.substr(0, tag.size()) == tag) signature.remove_prefix(tag.size());
  return signature;
#else
  return {};
#endif
}

/// @brief Drops every template-argument list: "a::B<int>::C<x<y>>" -> "a::B::C" (how the registry is keyed).
/// @param name A qualified type name.
/// @return The name without template arguments.
inline std::string without_template_args(std::string_view name) {
  std::string out;
  int depth = 0;
  for (const char c : name) {
    if (c == '<') depth++;
    if (depth == 0) out += c;
    if (c == '>' && depth > 0) depth--;
  }
  return out;
}

/// @brief Splits a registry line on tabs.
/// @param line One `kind\tname\tsignature\tbrief` line.
/// @return Its four fields (missing trailing fields are empty).
inline std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t from = 0;
  while (true) {
    const std::size_t tab = line.find('\t', from);
    fields.push_back(line.substr(from, tab == std::string_view::npos ? std::string_view::npos : tab - from));
    if (tab == std::string_view::npos) break;
    from = tab + 1;
  }
  fields.resize(4);
  return fields;
}

/// @brief Decodes a registry kind letter.
/// @param letter One of c, m, s, f, F, t.
/// @return The member kind.
inline DirMember::Kind kind_of(std::string_view letter) {
  if (letter == "c") return DirMember::Kind::Constructor;
  if (letter == "s") return DirMember::Kind::StaticMethod;
  if (letter == "f") return DirMember::Kind::Field;
  if (letter == "F") return DirMember::Kind::StaticField;
  if (letter == "t") return DirMember::Kind::Type;
  return DirMember::Kind::Method;
}

/// @brief Looks a qualified type name up in the generated registry.
/// @param qualified The type's fully qualified name, template arguments allowed ("datamunge::linalg::DenseMatrix<double>").
/// @return Its listing; `found()` is false when the registry has no entry.
inline DirListing lookup(std::string_view qualified) {
  std::string display(qualified);
  const std::string_view root = "datamunge::";
  if (display.compare(0, root.size(), root) == 0) display.erase(0, root.size());

  const std::string key = without_template_args(qualified);
  const RegistryEntry* first = kRegistry;
  const RegistryEntry* last = kRegistry + kRegistrySize;
  const RegistryEntry* hit =
      std::lower_bound(first, last, key, [](const RegistryEntry& entry, const std::string& k) { return entry.type < k; });
  if (hit == last || hit->type != key) return DirListing(std::move(display), "", {}, false);

  std::vector<DirMember> members;
  std::string description;
  std::size_t from = 0;
  bool first_line = true;
  const std::string_view text = hit->members;
  while (from <= text.size()) {
    const std::size_t newline = text.find('\n', from);
    const std::string_view line =
        text.substr(from, newline == std::string_view::npos ? std::string_view::npos : newline - from);
    if (first_line) {
      description = std::string(line);
      first_line = false;
    } else if (!line.empty()) {
      const std::vector<std::string_view> f = split_tabs(line);
      members.push_back(DirMember{kind_of(f[0]), std::string(f[1]), std::string(f[2]), std::string(f[3])});
    }
    if (newline == std::string_view::npos) break;
    from = newline + 1;
  }
  return DirListing(std::move(display), std::move(description), std::move(members), true);
}

}  // namespace dir_detail

/// @brief Lists a type's public methods and fields, like Python's `dir()`, without needing an object.
///
/// @tparam T The type to describe (cv/ref qualifiers are ignored); aliases such as dm::Plot
///           resolve to the class they name.
/// @return The listing; print it, or call names() for Python-style names. `found()` is false for
///         types outside the datamunge library.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::dir<dm::geometry::Point2D>().names());  // ["operator!=", "operator==", "x", "y"]
/// @endcode
/// @see dir(const T&)
template <typename T>
DirListing dir() {
  using Bare = std::remove_cv_t<std::remove_reference_t<T>>;
  return dir_detail::lookup(dir_detail::raw_type_name<Bare>());
}

/// @brief Lists an object's public methods and fields, like Python's `dir(obj)`.
///
/// Uses the object's static type (a derived object seen through a base reference lists the
/// base's members). Inherited members are included. Cheap: nothing about the object is read.
///
/// @tparam T The object's type (deduced).
/// @return The listing; print it with `std::print("{}\n", dm::dir(x))` (C++20) or
///         `std::cout << dm::dir(x).to_string()`, or call names() for Python-style names.
///
/// @par Example
/// @code
/// const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
/// const auto listing = dm::dir(model);
/// std::print("{} has {} methods\n", listing.type_name(),
///            listing.of_kind(dm::DirMember::Kind::Method).size());  // stats::LM has 43 methods
/// @endcode
/// @see dir()
template <typename T>
DirListing dir(const T& /*object*/) {
  return dir<T>();
}

}  // namespace datamunge

#if defined(__cpp_lib_format) || (__cplusplus >= 202002L && __has_include(<format>))
#include <format>
#ifdef __cpp_lib_format
#include <datamunge/format/text_formatter.hpp>

/// @brief Prints a dir() listing: the type, its description, then members grouped by kind.
template <>
struct std::formatter<datamunge::DirListing> : datamunge::format_detail::text_formatter<datamunge::DirListing> {
  /// @brief The listing's text.
  /// @param listing The listing to print.
  /// @return What DirListing::to_string() returns.
  static std::string text(const datamunge::DirListing& listing) { return listing.to_string(); }
};
#endif
#endif
