#pragma once

/// @file uci_sources.hpp
/// @brief Internal metadata for pinned UCI single-table snapshots.

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace datamunge::datasets::detail {

/// @brief One UCI table, with a verified source and attribution.
struct UciSource {
  unsigned id; ///< Stable repository identifier.
  std::string_view slug; ///< Alternative name, without the uci_ prefix.
  std::string_view title; ///< Original dataset title.
  std::size_t rows; ///< Verified CSV data rows.
  std::size_t columns; ///< Verified CSV field count.
  std::size_t bytes; ///< Exact download size in bytes.
  std::string_view sha256; ///< Lowercase digest of the original CSV.
  std::string_view missing; ///< Optional UCI missing-value marker.
  std::string_view license; ///< Explicit license shown on the official dataset page.
  std::string_view doi; ///< DOI of the original dataset.
  std::string_view creators; ///< Credited authors or donor organizations.
};

/// @brief All audited UCI snapshots in numerical ID order.
/// @return A view of the immutable registry.
std::span<const UciSource> uci_sources();

/// @brief Finds a registered UCI snapshot by its canonical id or name alias.
/// @param name A prefixed id such as uci_53 or alias such as uci_iris.
/// @return A pointer into the registry, or nullptr for an unknown name.
const UciSource* find_uci(std::string_view name);

/// @brief Returns the canonical cache and catalogue name for an entry.
/// @param source Registered entry.
/// @return A uci_-prefixed numerical id.
std::string uci_name(const UciSource& source);

/// @brief Returns the original HTTPS CSV URL for a pinned entry.
/// @param source Registered entry.
/// @return Official UCI download URL.
std::string uci_url(const UciSource& source);

} // namespace datamunge::datasets::detail
