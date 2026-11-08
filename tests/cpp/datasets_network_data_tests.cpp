#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <datamunge/datasets/download.hpp>
#include <datamunge/datasets/network_data.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
namespace ds = datamunge::datasets;
namespace {
struct Temp {
  std::filesystem::path root;
  Temp() {
    static std::atomic<unsigned> serial{0};
    for (int i = 0; i < 100; ++i) {
      root =
          std::filesystem::temp_directory_path() /
          ("dm-gml-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
           "-" + std::to_string(serial++));
      if (std::filesystem::create_directory(root))
        return;
    }
    throw std::runtime_error("test temporary directory unavailable");
  }
  ~Temp() {
    std::error_code e;
    std::filesystem::remove_all(root, e);
  }
};
void write(const std::filesystem::path& p, const std::string& s) {
  std::ofstream f(p, std::ios::binary);
  f << s;
  if (!f)
    throw std::runtime_error("test write failed");
}
ds::GMLGraph parse(
    const std::string& text,
    std::size_t        bytes    = 16777216,
    std::size_t        vertices = 1000000,
    std::size_t        edges    = 1000000) {
  Temp       t;
  const auto p = t.root / "input.gml";
  write(p, text);
  return ds::read_gml_graph(p.string(), bytes, vertices, edges);
}
} // namespace
TEST(NetworkData, SignedIdMappingAttributesDirectionAndMultiplicity) {
  const auto g = parse(
      "Creator \"a [quoted] creator\" # ignored comment\ngraph [ directed 1 node [ id 9 label "
      "\"name\" graphics [ x 2 nested [a 3] ] ] node [ id -2 ] node [ id 5 ] edge [ source 9 "
      "target -2 weight 99 ] edge [ source -2 target 9 ] edge [ source 9 target -2 ] edge [source "
      "5 target 5] ]");
  EXPECT_EQ(g.node_ids, (std::vector<std::int64_t>{-2, 5, 9}));
  EXPECT_TRUE(g.directed);
  EXPECT_EQ(
      g.edges, (std::vector<std::pair<std::size_t, std::size_t>>{{2, 0}, {0, 2}, {2, 0}, {1, 1}}));
  auto copy        = g;
  copy.node_ids[0] = 999;
  copy.edges.clear();
  EXPECT_EQ(g.node_ids[0], -2);
  EXPECT_EQ(g.edges.size(), 4u);
}
TEST(NetworkData, EmptyGraphUnknownListsAndIterativeDeepSkipping) {
  EXPECT_TRUE(parse("graph [ ]").node_ids.empty());
  EXPECT_FALSE(parse("graph []").directed);
  const std::string prefix = "graph [ ignored ";
  const auto        deep =
      prefix + std::string(10000, '[') + " id 7 " + std::string(10000, ']') + " node [id 2] ]";
  EXPECT_EQ(parse(deep).node_ids, (std::vector<std::int64_t>{2}));
  const auto extreme = parse(
      "graph [node [id -9223372036854775808] node [id 9223372036854775807] edge [source "
      "-9223372036854775808 target 9223372036854775807] ]");
  EXPECT_EQ(extreme.node_ids.front(), std::numeric_limits<std::int64_t>::min());
  EXPECT_EQ(extreme.node_ids.back(), std::numeric_limits<std::int64_t>::max());
  EXPECT_EQ(extreme.edges[0].first, 0u);
}
TEST(NetworkData, MalformedAndIncompleteTopologyRejected) {
  for (const auto& s : std::vector<std::string>{
           "",
           "node [id 1]",
           "graph [",
           "graph ]",
           "graph [] graph []",
           "graph [ node [] ]",
           "graph [node [id 1 id 2]]",
           "graph [node [id 1] node [id 1]]",
           "graph [node [id 1] edge [source 1]]",
           "graph [node [id 1] edge [source 1 target 2]]",
           "graph [node [id 1] edge [source 1 source 1 target 1]]",
           "graph [directed 2]",
           "graph [directed 0 directed 1]",
           "graph [node [id 1.5]]",
           "graph [node [id \"1\"]]",
           "graph [foo \"unclosed]",
           "graph [foo [ x 2]",
           "graph [node [id 1] edge [source 1 target 1 target 1]]"})
    EXPECT_THROW(parse(s), std::invalid_argument) << s;
  EXPECT_THROW(parse("graph [node [id 9223372036854775808]]"), std::length_error);
}
TEST(NetworkData, ExactByteAndCompleteRecordLimits) {
  const std::string s =
      "graph [node [id 9] node [id 2] edge [source 9 target 2] edge [source 2 target 9]]";
  EXPECT_EQ(parse(s, s.size(), 2, 2).edges.size(), 2u);
  EXPECT_THROW(parse(s, s.size() - 1, 2, 2), std::length_error);
  EXPECT_THROW(parse(s, 0, 1, 0), std::length_error);
  EXPECT_THROW(parse(s, 0, 0, 1), std::length_error);
  EXPECT_EQ(parse(s, 0, 0, 0).node_ids.size(), 2u);
  Temp       t;
  const auto p = t.root / "input.gml";
  write(p, s);
  EXPECT_THROW(ds::read_gml_graph(p.string(), 1), std::length_error);
  std::ifstream     in(p);
  const std::string unchanged((std::istreambuf_iterator<char>(in)), {});
  EXPECT_EQ(unchanged, s);
  EXPECT_THROW(ds::read_gml_graph((t.root / "missing").string()), std::runtime_error);
}
TEST(NetworkData, VerifiedKarateCacheAndSourceMapping) {
  const char* root = std::getenv("DATAMUNGE_KARATE_TEST_CACHE");
  if (!root)
    GTEST_SKIP()
        << "Set DATAMUNGE_KARATE_TEST_CACHE to a native prepared original cache for integration.";
  const auto g = ds::karate_club(root);
  ASSERT_EQ(g.node_ids.size(), 34u);
  ASSERT_EQ(g.edges.size(), 78u);
  EXPECT_FALSE(g.directed);
  for (std::size_t i = 0; i < 34; ++i)
    EXPECT_EQ(g.node_ids[i], std::int64_t(i + 1));
  std::vector<std::size_t> degree(34);
  for (auto [u, v] : g.edges) {
    ASSERT_LT(u, 34u);
    ASSERT_LT(v, 34u);
    ++degree[u];
    ++degree[v];
  }
  EXPECT_EQ(degree[0], 16u);
  EXPECT_EQ(degree[33], 17u);
  const auto names = ds::downloadable();
  EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
  EXPECT_NE(std::find(names.begin(), names.end(), "karate_club"), names.end());
}
TEST(NetworkData, CorruptPinnedMemberRejectedWithoutReplacement) {
  const char* root = std::getenv("DATAMUNGE_KARATE_TEST_CACHE");
  if (!root)
    GTEST_SKIP() << "Requires native prepared original cache.";
  Temp t;
  for (const char* name : {"karate.zip", "karate.gml", "karate.txt"})
    std::filesystem::copy_file(std::filesystem::path(root) / name, t.root / name);
  {
    std::ofstream f(t.root / "karate.gml", std::ios::app);
    f << "# changed\n";
  }
  const auto size = std::filesystem::file_size(t.root / "karate.gml");
  EXPECT_THROW(ds::karate_club(t.root.string()), std::runtime_error);
  EXPECT_EQ(std::filesystem::file_size(t.root / "karate.gml"), size);
}
