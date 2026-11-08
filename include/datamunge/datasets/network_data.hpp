#pragma once
/// @file network_data.hpp
/// @brief Bounded native GML topology ingress and verified Zachary karate-club data.
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::datasets {
/// @brief An owned graph topology with explicit original-to-native vertex mapping.
/// node_ids are ascending unique original signed decimal ids; native index i
/// corresponds to node_ids[i]. edges preserve input order, direction, loops and
/// multiplicity. directed records graph metadata, default false. Weights, labels,
/// layout and other attributes are skipped: this is a topology reader, not a
/// weighted/labeled GML round trip. Copies own all vectors independently.
/// @par Example
/// @code
/// const dm::datasets::GMLGraph g{};
/// std::cout << g.node_ids.size() << " " << g.directed; // 0 0
/// @endcode
/// @see read_gml_graph(), karate_club()
struct GMLGraph {
    std::vector<std::int64_t> node_ids; ///< Ascending original ids; vector index is the native id.
    std::vector<std::pair<std::size_t,std::size_t>> edges; ///< Owned zero-based endpoint pairs in source order.
    bool directed{false}; ///< Graph's directed0/1 metadata, absent means false.
};
/// @brief Read bounded GML node/edge topology with an owned zero-based mapping.
///
/// Supports one graph block, node id and edge source/target signed decimal
/// int64 ids, directed0/1, quoted strings, comments beginning# and balanced
/// nested unknown attributes. Unknown labels/weights/layout are skipped. Vertex
/// ids are sorted before mapping; edge order, direction, loops and duplicates
/// remain. Rejects duplicate node ids, duplicate required attributes, missing
/// ids/endpoints, unknown endpoints, invalid directed flag, malformed integers,
/// unbalanced lists and unterminated strings. Graphs may be empty or directed;
/// callers choose a compatible analysis. No source-file mutation.
/// Positive quotas bound file bytes and whole node/edge records before append;
///0 disables the respective optional quota. Numeric/container bounds remain.
/// These are ingress limits, not peak RAM limits; text/tokens/maps/output coexist.
/// Work O(bytes+(nodes+edges)log(nodes+1)), owned storage O(bytes+nodes+edges).
/// @param path Readable original GML file.
/// @param max_bytes Positive optional source-byte quota, default16777216;0 disables it.
/// @param max_vertices Positive optional complete-node quota, default1000000;0 disables it.
/// @param max_edges Positive optional complete-edge quota, default1000000;0 disables it.
/// @return Owned topology and original-id mapping; no caller references.
/// @throws std::invalid_argument Malformed GML, duplicate/missing ids or unknown endpoints.
/// @throws std::length_error Positive quota, int64 integer range or container capacity exceeded.
/// @throws std::runtime_error File read or filesystem operation failed.
/// @par Example
/// @code
/// #include <fstream>
/// std::ofstream("build/tiny.gml") << "graph [ node [ id 9 ] node [ id 2 ] edge [ source 9 target 2 ] ]";
/// const auto g=dm::datasets::read_gml_graph("build/tiny.gml");
/// std::cout << g.node_ids[0] << " " << g.edges[0].first; // 2 1
/// @endcode
/// @par References
/// - M. E. J. Newman's original GML network collection, https://websites.umich.edu/~mejn/netdata/
/// @see GMLGraph, karate_club()
GMLGraph read_gml_graph(const std::string& path,std::size_t max_bytes=16777216,std::size_t max_vertices=1000000,std::size_t max_edges=1000000);
/// @brief Fetch and read the original unweighted Zachary karate-club topology.
///
/// Uses fetch("karate_club") and the shared cache/offline/mirror controls. The
/// author's970-byte ZIP and both extracted members are pinned/verified on every
/// call; corrupt existing bytes throw without replacement. Missing members are
/// extracted using the native bounded ZIP reader. curl and native data adapters
/// must be available. Returns34 nodes/source ids1..34 and78 undirected edges;
/// native id=source id−1. This archive contains no faction membership labels;
/// do not infer ground-truth community accuracy from this object. Reads source
/// README/citation alongside GML; no data synthesized or Python/manual ingress.
/// Owns all output; O(source bytes+edges log nodes) small fixed-data work/storage.
/// Parallel calls publish verified complete files through native ingress helpers;
/// callers must not mutate cache files during use.
/// @param directory Explicit cache directory, empty selects data_home()/karate_club.
/// @return Owned original graph with ascending source-id mapping.
/// @throws std::runtime_error Missing offline data, transfer/hash/archive/read failure or unexpected pinned topology.
/// @throws std::invalid_argument Malformed source or unsafe archive content.
/// @throws std::length_error Bounded source/node/edge ingress exceeded.
/// @par Example
/// @code
/// const auto g=dm::datasets::karate_club();
/// std::cout << g.node_ids.size() << " " << g.edges.size(); // 34 78
/// @endcode
/// @par References
/// - W. W. Zachary (1977), An Information Flow Model for Conflict and Fission
///   in Small Groups, Journal of Anthropological Research33:452–473.
///   https://doi.org/10.1086/jar.33.4.3629752
/// - M. E. J. Newman original collection, https://websites.umich.edu/~mejn/netdata/karate.zip
/// @see read_gml_graph(), fetch(), file_digest()
GMLGraph karate_club(const std::string& directory="");
/// @brief Internal verified source preparation used by fetch("karate_club").
namespace detail {
/// @brief Acquire the pinned archive and verify both native-extracted members.
/// @param directory Dataset directory chosen by fetch(), possibly relative.
/// @return Absolute dataset directory containing karate.zip, karate.gml and karate.txt.
/// @throws std::runtime_error Cache, transfer, digest, archive or filesystem failure.
/// @throws std::invalid_argument Native ingress rejects a path or archive.
std::string fetch_karate_source(const std::string& directory);
} // namespace detail
} // namespace datamunge::datasets
