/// @file network_data.cpp
/// @brief Bounded GML tokenization/topology mapping and pinned native karate-club acquisition.
#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <datamunge/datasets/assets.hpp>
#include <datamunge/datasets/download.hpp>
#include <datamunge/datasets/network_data.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace datamunge::datasets {
/// @brief Private bounded GML lexical and topology parsing helpers.
namespace {
/// @brief Token categories, separating lists and quoted strings from bare values.
enum class Kind {
  word,   ///< Bare attribute name or decimal integer.
  quoted, ///< Quoted string, with backslash-escaped characters accepted.
  open,   ///< Opening list bracket.
  close,  ///< Closing list bracket.
  end,    ///< End of source.
};

/// @brief One owned bounded token from the current source.
struct Token {
  Kind        kind{Kind::end}; ///< Lexical category.
  std::string text;            ///< Owned scalar contents, excluding quotes/brackets.
};
/// @brief Maintain one-token lookahead into an already bounded owned source string.
class Lexer {
  const std::string& source_;  ///< Source borrowed only during the reader call.
  std::size_t        at_{0};   ///< Next unconsumed byte offset.
  Token              current_; ///< Current owned lookahead.
 public:
  /// @brief Initialize and obtain the first token.
  /// @param source Bounded source string outliving this lexer.
  /// @throws std::invalid_argument Unterminated quoted string.
  explicit Lexer(const std::string& source) : source_(source) { next(); }
  /// @brief Inspect the current lookahead without advancing.
  /// @return Current token, borrowed until next().
  const Token& peek() const { return current_; }
  /// @brief Consume lookahead and lex the next token.
  /// @throws std::invalid_argument Unterminated quoted string.
  void next() {
    while (at_ < source_.size()) {
      const unsigned char c = static_cast<unsigned char>(source_[at_]);
      if (std::isspace(c)) {
        ++at_;
        continue;
      }
      if (c == '#') {
        while (at_ < source_.size() && source_[at_] != '\n')
          ++at_;
        continue;
      }
      break;
    }
    current_ = {};
    if (at_ == source_.size())
      return;
    char c = source_[at_++];
    if (c == '[') {
      current_.kind = Kind::open;
      return;
    }
    if (c == ']') {
      current_.kind = Kind::close;
      return;
    }
    if (c == '"') {
      current_.kind = Kind::quoted;
      while (at_ < source_.size()) {
        c = source_[at_++];
        if (c == '"')
          return;
        if (c == '\\') {
          if (at_ == source_.size())
            break;
          c = source_[at_++];
        }
        current_.text += c;
      }
      throw std::invalid_argument("GML: unterminated string");
    }
    current_.kind = Kind::word;
    current_.text += c;
    while (at_ < source_.size()) {
      const unsigned char ch = static_cast<unsigned char>(source_[at_]);
      if (std::isspace(ch) || ch == '[' || ch == ']' || ch == '#' || ch == '"')
        break;
      current_.text += source_[at_++];
    }
  }
};
/// @brief Require and consume a specific token category.
/// @param lexer Current token stream.
/// @param kind Expected category.
/// @throws std::invalid_argument Unexpected source category.
void expect(Lexer& lexer, Kind kind) {
  if (lexer.peek().kind != kind)
    throw std::invalid_argument("GML: unexpected token");
  lexer.next();
}
/// @brief Consume one bare attribute key.
/// @param lexer Current token stream.
/// @return Owned attribute name.
/// @throws std::invalid_argument The current token is not bare text.
std::string key(Lexer& lexer) {
  if (lexer.peek().kind != Kind::word)
    throw std::invalid_argument("GML: attribute key required");
  auto text = lexer.peek().text;
  lexer.next();
  return text;
}
/// @brief Parse one strict signed decimal int64 scalar.
/// @param lexer Current token stream.
/// @return Checked signed integer value.
/// @throws std::invalid_argument Missing/malformed decimal integer.
/// @throws std::length_error Integer does not fit int64.
std::int64_t integer(Lexer& lexer) {
  if (lexer.peek().kind != Kind::word)
    throw std::invalid_argument("GML: integer required");
  const auto&  s     = lexer.peek().text;
  std::int64_t value = 0;
  const auto   r     = std::from_chars(s.data(), s.data() + s.size(), value);
  if (r.ec == std::errc::result_out_of_range)
    throw std::length_error("GML: int64 capacity");
  if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
    throw std::invalid_argument("GML: invalid integer");
  lexer.next();
  return value;
}
/// @brief Skip one unknown scalar/list attribute without recursive stack use.
/// @param lexer Current token stream, positioned at an attribute value.
/// @throws std::invalid_argument Missing value or unbalanced list.
/// @throws std::length_error Bracket nesting counter cannot fit.
void skip(Lexer& lexer) {
  if (lexer.peek().kind == Kind::word || lexer.peek().kind == Kind::quoted) {
    lexer.next();
    return;
  }
  if (lexer.peek().kind != Kind::open)
    throw std::invalid_argument("GML: attribute value required");
  std::size_t depth = 0;
  do {
    if (lexer.peek().kind == Kind::end)
      throw std::invalid_argument("GML: unclosed list");
    if (lexer.peek().kind == Kind::open) {
      if (depth == std::numeric_limits<std::size_t>::max())
        throw std::length_error("GML: nesting capacity");
      ++depth;
    } else if (lexer.peek().kind == Kind::close)
      --depth;
    lexer.next();
  } while (depth);
}
/// @brief Check the next complete node/edge append against optional/capacity limits.
/// @param size Existing complete-record count.
/// @param maximum Positive optional quota, or0 to disable it.
/// @param capacity Actual vector maximum.
/// @throws std::length_error Quota or capacity exhausted.
void room(std::size_t size, std::size_t maximum, std::size_t capacity) {
  if ((maximum && size >= maximum) || size == capacity)
    throw std::length_error("GML: complete record quota/capacity");
}
/// @brief Read original bytes under a checked optional byte quota, including growth races.
/// @param path Readable source file.
/// @param max_bytes Positive optional byte limit, or0.
/// @return Owned complete original bytes.
/// @throws std::runtime_error File read failed.
/// @throws std::length_error Source quota/string capacity exceeded.
std::string read(const std::string& path, std::size_t max_bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("GML: cannot open " + path);
  std::string            data;
  std::array<char, 4096> buffer;
  while (f) {
    f.read(buffer.data(), buffer.size());
    const auto n = static_cast<std::size_t>(f.gcount());
    if (n > data.max_size() - data.size() ||
        (max_bytes && (data.size() > max_bytes || n > max_bytes - data.size())))
      throw std::length_error("GML: source byte quota/capacity");
    data.append(buffer.data(), n);
  }
  if (!f.eof())
    throw std::runtime_error("GML: read failed " + path);
  return data;
}
/// @brief Verify an existing extracted member before reuse or replacement.
/// @param path Existing or absent expected member.
/// @param sha Pinned lowercase SHA-256.
/// @return True if the member exists with the correct digest.
/// @throws std::runtime_error An existing member is corrupt or unreadable.
bool member(const std::filesystem::path& path, const std::string& sha) {
  if (!std::filesystem::exists(path))
    return false;
  if (file_digest(path.string()) != sha)
    throw std::runtime_error("karate_club: changed cached member " + path.string());
  return true;
}
} // namespace

GMLGraph read_gml_graph(
    const std::string& path,
    std::size_t        max_bytes,
    std::size_t        max_vertices,
    std::size_t        max_edges) {
  const auto                                         data = read(path, max_bytes);
  Lexer                                              lexer(data);
  GMLGraph                                           out;
  std::vector<std::pair<std::int64_t, std::int64_t>> raw;
  bool                                               graph_seen = false, directed_seen = false;
  while (lexer.peek().kind != Kind::end) {
    const auto top = key(lexer);
    if (top != "graph") {
      skip(lexer);
      continue;
    }
    if (graph_seen)
      throw std::invalid_argument("GML: multiple graph blocks");
    graph_seen = true;
    expect(lexer, Kind::open);
    while (lexer.peek().kind != Kind::close) {
      const auto attribute = key(lexer);
      if (attribute == "directed") {
        if (directed_seen)
          throw std::invalid_argument("GML: duplicate directed flag");
        directed_seen    = true;
        const auto value = integer(lexer);
        if (value != 0 && value != 1)
          throw std::invalid_argument("GML: directed must be0 or1");
        out.directed = value != 0;
        continue;
      }
      if (attribute != "node" && attribute != "edge") {
        skip(lexer);
        continue;
      }
      expect(lexer, Kind::open);
      bool         have_id = false, have_source = false, have_target = false;
      std::int64_t id = 0, source = 0, target = 0;
      while (lexer.peek().kind != Kind::close) {
        const auto name = key(lexer);
        if (attribute == "node" && name == "id") {
          if (have_id)
            throw std::invalid_argument("GML: duplicate node id attribute");
          id      = integer(lexer);
          have_id = true;
        } else if (attribute == "edge" && name == "source") {
          if (have_source)
            throw std::invalid_argument("GML: duplicate edge source");
          source      = integer(lexer);
          have_source = true;
        } else if (attribute == "edge" && name == "target") {
          if (have_target)
            throw std::invalid_argument("GML: duplicate edge target");
          target      = integer(lexer);
          have_target = true;
        } else
          skip(lexer);
      }
      expect(lexer, Kind::close);
      if (attribute == "node") {
        if (!have_id)
          throw std::invalid_argument("GML: node id missing");
        room(out.node_ids.size(), max_vertices, out.node_ids.max_size());
        out.node_ids.push_back(id);
      } else {
        if (!have_source || !have_target)
          throw std::invalid_argument("GML: edge endpoint missing");
        room(raw.size(), max_edges, raw.max_size());
        raw.emplace_back(source, target);
      }
    }
    expect(lexer, Kind::close);
  }
  if (!graph_seen)
    throw std::invalid_argument("GML: graph block missing");
  std::sort(out.node_ids.begin(), out.node_ids.end());
  if (std::adjacent_find(out.node_ids.begin(), out.node_ids.end()) != out.node_ids.end())
    throw std::invalid_argument("GML: duplicate node ids");
  out.edges.reserve(raw.size());
  for (auto [u, v] : raw) {
    const auto a = std::lower_bound(out.node_ids.begin(), out.node_ids.end(), u),
               b = std::lower_bound(out.node_ids.begin(), out.node_ids.end(), v);
    if (a == out.node_ids.end() || *a != u || b == out.node_ids.end() || *b != v)
      throw std::invalid_argument("GML: edge references unknown node");
    out.edges.emplace_back(
        static_cast<std::size_t>(a - out.node_ids.begin()),
        static_cast<std::size_t>(b - out.node_ids.begin()));
  }
  return out;
}
/// @brief Internal verified original-source acquisition.
namespace detail {
std::string fetch_karate_source(const std::string& directory) {
  const auto dir     = std::filesystem::absolute(directory).lexically_normal();
  const auto archive = dir / "karate.zip";
  /// @brief Pinned original author-collection ZIP SHA-256, not a mirror-specific hash.
  const std::string archive_sha =
      "e182d7fabbd44f3c75d307c1cf5259f0c4506954d63c37b9f30e629c78c578f6";
  /// @brief Pinned exact original GML member SHA-256.
  const std::string graph_sha = "f05e321c94b00a5cceab44671a65c34632aac7b152479ba619354be149ba8beb";
  /// @brief Pinned exact original source README SHA-256.
  const std::string readme_sha = "42b1769e51ae6dd09644f71357afadfc18128a29bad3d9c110664d3a75d08566";
  fetch_file(
      "https://websites.umich.edu/~mejn/netdata/karate.zip",
      archive.string(),
      archive_sha,
      970,
      "karate_club");
  const bool graph  = member(dir / "karate.gml", graph_sha),
             readme = member(dir / "karate.txt", readme_sha);
  if (!graph || !readme)
    extract_archive(archive.string(), dir.string(), {"karate.gml", "karate.txt"}, 16384);
  if (!member(dir / "karate.gml", graph_sha) || !member(dir / "karate.txt", readme_sha))
    throw std::runtime_error("karate_club: missing verified source members");
  return dir.string();
}
} // namespace detail
GMLGraph karate_club(const std::string& directory) {
  const auto root = fetch("karate_club", directory);
  auto       g    = read_gml_graph(root + "/karate.gml", 16384, 34, 78);
  if (g.directed || g.node_ids.size() != 34 || g.edges.size() != 78)
    throw std::runtime_error("karate_club: unexpected pinned topology");
  for (std::size_t i = 0; i < g.node_ids.size(); ++i)
    if (g.node_ids[i] != static_cast<std::int64_t>(i + 1))
      throw std::runtime_error("karate_club: unexpected source ids");
  return g;
}
} // namespace datamunge::datasets
