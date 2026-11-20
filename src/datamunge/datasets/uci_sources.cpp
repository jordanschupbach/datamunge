/// @file uci_sources.cpp
/// @brief UCI table lookup and source attribution.

#include "uci_sources.hpp"

#include <datamunge/datasets/datasets.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <vector>

namespace datamunge::datasets::detail {
namespace {

/// @brief Pinned and licensed snapshots from data-raw/datasets/uci_catalog.tsv.
constexpr auto kSources = std::to_array<UciSource>({
#include "uci_catalog.inc"
});

} // namespace

std::span<const UciSource> uci_sources() { return kSources; }

const UciSource* find_uci(std::string_view name) {
  if (!name.starts_with("uci_")) return nullptr;
  name.remove_prefix(4);
  unsigned id{};
  const auto [end, error] = std::from_chars(name.data(), name.data() + name.size(), id);
  if (error == std::errc{} && end == name.data() + name.size()) {
    const auto found = std::ranges::lower_bound(kSources, id, {}, &UciSource::id);
    return found != kSources.end() && found->id == id ? &*found : nullptr;
  }
  const auto found = std::ranges::find(kSources, name, &UciSource::slug);
  return found == kSources.end() ? nullptr : &*found;
}

std::string uci_name(const UciSource& source) { return "uci_" + std::to_string(source.id); }

std::string uci_url(const UciSource& source) {
  return "https://archive.ics.uci.edu/static/public/" + std::to_string(source.id) + "/data.csv";
}

} // namespace datamunge::datasets::detail

namespace datamunge::datasets {

dstruct::DataFrame uci_catalog() {
  std::vector<std::string> names, aliases, titles, licenses, dois, creators, urls, hashes, missing;
  std::vector<double> rows, columns, bytes;
  for (const auto& source : detail::uci_sources()) {
    names.push_back(detail::uci_name(source));
    aliases.push_back("uci_" + std::string(source.slug));
    titles.emplace_back(source.title);
    rows.push_back(static_cast<double>(source.rows));
    columns.push_back(static_cast<double>(source.columns));
    bytes.push_back(static_cast<double>(source.bytes));
    licenses.emplace_back(source.license);
    dois.emplace_back(source.doi);
    creators.emplace_back(source.creators);
    urls.push_back(detail::uci_url(source));
    hashes.emplace_back(source.sha256);
    missing.emplace_back(source.missing);
  }
  dstruct::DataFrame frame;
  frame.add_column("name", names);
  frame.add_column("alias", aliases);
  frame.add_column("title", titles);
  frame.add_column("rows", rows);
  frame.add_column("columns", columns);
  frame.add_column("bytes", bytes);
  frame.add_column("license", licenses);
  frame.add_column("doi", dois);
  frame.add_column("creators", creators);
  frame.add_column("url", urls);
  frame.add_column("sha256", hashes);
  frame.add_column("missing", missing);
  return frame;
}

} // namespace datamunge::datasets
