#pragma once

/// @file dstruct.hpp
/// @brief std::format / std::print support for datamunge::dstruct.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Containers whose contents are publicly readable print a capped preview of
/// their elements (when the element type is itself formattable); the others
/// print their size or just their name. Array and DynamicArray are ranges and
/// already print through the standard range formatter (`[1, 2, 3]`). Graphs
/// print one adjacency line per vertex; matrices and images print a grid like
/// linalg::DenseMatrix.
///
/// @par Example
/// @code
/// dm::dstruct::DirectedGraph<int> g{1, 2, 3};
/// g.add_edge(1, 2);
/// g.add_edge(1, 3);
/// std::print("{}\n", g);
/// // DirectedGraph(vertices=3, edges=2)
/// //   1 -> [2, 3]
/// //   2 -> []
/// //   3 -> []
/// @endcode

#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/format/linalg.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <utility>
#include <variant>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge::format_detail {

/// @brief Helpers for the dstruct formatters.
namespace dstruct_fmt {

/// @brief Vertices listed by a graph formatter before eliding the rest.
inline constexpr std::size_t max_vertices = 10;

/// @brief `name` followed by a preview of `values` when T is formattable, else `name(size=n)`.
/// @tparam T Element type.
/// @param name The container's name.
/// @param values The elements in order.
/// @return E.g. "LinkedList[1, 2, 3]".
template <typename T>
std::string listing(const char* name, const std::vector<T>& values) {
  if constexpr (std::formattable<T, char>) return name + preview(values);
  else return std::format("{}(size={})", name, values.size());
}

/// @brief `name{k1: v1, k2: v2, ...}` (first 6 entries, then ", ...} (n=N)"), else `name(size=n)`.
/// @tparam Key Key type.
/// @tparam Value Mapped type.
/// @param name The container's name.
/// @param entries The key/value pairs in the order to print.
/// @return One line.
template <typename Key, typename Value>
std::string mapping(const char* name, const std::vector<std::pair<Key, Value>>& entries) {
  if constexpr (std::formattable<Key, char> && std::formattable<Value, char>) {
    std::string out = std::string(name) + "{";
    for (std::size_t i = 0; i < entries.size() && i < 6; ++i)
      out += std::format("{}{}: {}", i ? ", " : "", entries[i].first, entries[i].second);
    if (entries.size() > 6) out += std::format(", ...}} (n={})", entries.size());
    else out += "}";
    return out;
  } else {
    return std::format("{}(size={})", name, entries.size());
  }
}

/// @brief Tree nodes printed by a tree outline before eliding the rest.
inline constexpr std::size_t max_tree_nodes = 20;

/// @brief Header "name(nodes=n)" plus one indented line per node in preorder, capped at max_tree_nodes.
/// @tparam Node A node type with a public `value` member.
/// @tparam Children Callable returning `std::vector<std::pair<std::string, const Node*>>`: a label
///   (e.g. "left: ") and pointer for each present child, in print order.
/// @param name The tree's printed name.
/// @param root The root node.
/// @param children Lists a node's children.
/// @return The multi-line text (just the header when values are not formattable).
template <typename Node, typename Children>
std::string outline(const std::string& name, const Node& root, Children&& children) {
  struct Item {
    const Node* node;
    std::size_t depth;
    std::string label;
  };
  std::string body;
  std::size_t count = 0;
  std::vector<Item> pending{{&root, 1, ""}};
  while (!pending.empty()) {
    Item item = std::move(pending.back());
    pending.pop_back();
    if constexpr (std::formattable<decltype(root.value), char>) {
      if (count < max_tree_nodes)
        body += std::format("\n{}{}{}", std::string(2 * item.depth, ' '), item.label, item.node->value);
    }
    ++count;
    auto kids = children(*item.node);
    for (auto it = kids.rbegin(); it != kids.rend(); ++it) pending.push_back({it->second, item.depth + 1, it->first});
  }
  if constexpr (std::formattable<decltype(root.value), char>) {
    if (count > max_tree_nodes) body += std::format("\n  ... ({} more nodes)", count - max_tree_nodes);
  }
  return std::format("{}(nodes={})", name, count) + body;
}

/// @brief "Heap(size=n, top=v)" (top omitted when empty or not formattable).
/// @tparam Heap A heap with size(), empty() and top().
/// @param name The heap's printed name.
/// @param heap The heap.
/// @return One line.
template <typename Heap>
std::string heap(const std::string& name, const Heap& heap) {
  using T = std::remove_cvref_t<decltype(heap.top())>;
  if constexpr (std::formattable<T, char>) {
    if (!heap.empty()) return std::format("{}(size={}, top={})", name, heap.size(), heap.top());
  }
  return std::format("{}(size={})", name, heap.size());
}

/// @brief Header plus one adjacency line per vertex (`  v <arrow> [n1, n2]`), capped at max_vertices.
/// @tparam Vertex Vertex type.
/// @tparam Neighbors Callable returning the neighbour list text for a vertex.
/// @param header The first line.
/// @param vertices The graph's vertices in insertion order.
/// @param arrow The edge marker, "->" or "--".
/// @param neighbors Produces the bracketed neighbour list for a vertex.
/// @return The multi-line text.
template <typename Vertex, typename Neighbors>
std::string adjacency(std::string header, const std::vector<Vertex>& vertices, const char* arrow,
                      Neighbors&& neighbors) {
  if constexpr (std::formattable<Vertex, char>) {
    std::size_t shown = 0;
    for (const auto& v : vertices) {
      if (shown++ == max_vertices) {
        header += std::format("\n  ... ({} more vertices)", vertices.size() - max_vertices);
        break;
      }
      header += std::format("\n  {} {} {}", v, arrow, neighbors(v));
    }
  }
  return header;
}

}  // namespace dstruct_fmt
}  // namespace datamunge::format_detail

/// @brief Formats a dataframe using to_string() and the standard string width/alignment options.
template <>
struct std::formatter<datamunge::dstruct::DataFrame> : datamunge::format_detail::text_formatter<datamunge::dstruct::DataFrame> {
  /// @brief The dataframe's default textual preview.
  /// @param frame The dataframe to print.
  /// @return The text DataFrame::to_string() returns.
  static std::string text(const datamunge::dstruct::DataFrame& frame) { return frame.to_string(); }
};

// ---- foundational.hpp ----------------------------------------------------

/// @brief Prints a LinkedList as "LinkedList[a, b, c]" (or "LinkedList(size=n)" for unformattable elements).
template <typename T>
struct std::formatter<datamunge::dstruct::LinkedList<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::LinkedList<T>> {
  /// @brief The list's elements, front first.
  /// @param list The list.
  /// @return One line.
  static std::string text(const datamunge::dstruct::LinkedList<T>& list) {
    std::vector<T> values;
    if constexpr (std::formattable<T, char>) list.for_each([&](const T& v) { values.push_back(v); });
    else return std::format("LinkedList(size={})", list.size());
    return datamunge::format_detail::dstruct_fmt::listing("LinkedList", values);
  }
};

/// @brief Prints a DoublyLinkedList as "DoublyLinkedList[a, b, c]" (or "DoublyLinkedList(size=n)" for unformattable elements).
template <typename T>
struct std::formatter<datamunge::dstruct::DoublyLinkedList<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::DoublyLinkedList<T>> {
  /// @brief The list's elements, front first.
  /// @param list The list.
  /// @return One line.
  static std::string text(const datamunge::dstruct::DoublyLinkedList<T>& list) {
    return datamunge::format_detail::dstruct_fmt::listing("DoublyLinkedList", list.to_vector());
  }
};

/// @brief Prints a Stack as "Stack[top, ..., bottom]" (or "Stack(size=n)" for unformattable elements).
template <typename T>
struct std::formatter<datamunge::dstruct::Stack<T>> : datamunge::format_detail::text_formatter<datamunge::dstruct::Stack<T>> {
  /// @brief The stack's elements, top first.
  /// @param stack The stack.
  /// @return One line.
  static std::string text(const datamunge::dstruct::Stack<T>& stack) {
    return datamunge::format_detail::dstruct_fmt::listing("Stack", stack.to_vector());
  }
};

/// @brief Prints a Queue as "Queue[front, ..., back]" (or "Queue(size=n)" for unformattable elements).
template <typename T>
struct std::formatter<datamunge::dstruct::Queue<T>> : datamunge::format_detail::text_formatter<datamunge::dstruct::Queue<T>> {
  /// @brief The queue's elements, front first.
  /// @param queue The queue.
  /// @return One line.
  static std::string text(const datamunge::dstruct::Queue<T>& queue) {
    return datamunge::format_detail::dstruct_fmt::listing("Queue", queue.to_vector());
  }
};

/// @brief Prints a DoubleEndedQueue as "DoubleEndedQueue[front, ..., back]" (or "(size=n)" for unformattable elements).
template <typename T>
struct std::formatter<datamunge::dstruct::DoubleEndedQueue<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::DoubleEndedQueue<T>> {
  /// @brief The deque's elements, front first.
  /// @param deque The deque.
  /// @return One line.
  static std::string text(const datamunge::dstruct::DoubleEndedQueue<T>& deque) {
    return datamunge::format_detail::dstruct_fmt::listing("DoubleEndedQueue", deque.to_vector());
  }
};

/// @brief Prints a CircularBuffer as "CircularBuffer(size=n, capacity=c)".
template <typename T>
struct std::formatter<datamunge::dstruct::CircularBuffer<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::CircularBuffer<T>> {
  /// @brief The buffer's fill level.
  /// @param buffer The buffer.
  /// @return One line.
  static std::string text(const datamunge::dstruct::CircularBuffer<T>& buffer) {
    return std::format("CircularBuffer(size={}, capacity={})", buffer.size(), buffer.capacity());
  }
};

/// @brief Prints a BitArray as "BitArray(n=8) 01001101" (bit 0 first; more than 64 bits are elided).
template <>
struct std::formatter<datamunge::dstruct::BitArray> : datamunge::format_detail::text_formatter<datamunge::dstruct::BitArray> {
  /// @brief The size and bit string.
  /// @param bits The bit array.
  /// @return One line.
  static std::string text(const datamunge::dstruct::BitArray& bits) {
    std::string out = std::format("BitArray(n={})", bits.size());
    if (bits.size() == 0) return out;
    out += ' ';
    const std::size_t shown = bits.size() > 64 ? 64 : bits.size();
    for (std::size_t i = 0; i < shown; ++i) out += bits.at(i) ? '1' : '0';
    if (shown < bits.size()) out += "...";
    return out;
  }
};

// ---- array_variants.hpp / association_list.hpp ----------------------------

/// @brief Prints a SortedArray as "SortedArray[a, b, c]" in sorted order.
template <typename T, typename Compare>
struct std::formatter<datamunge::dstruct::SortedArray<T, Compare>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::SortedArray<T, Compare>> {
  /// @brief The sorted elements.
  /// @param array The array.
  /// @return One line.
  static std::string text(const datamunge::dstruct::SortedArray<T, Compare>& array) {
    std::vector<T> values;
    for (std::size_t i = 0; i < array.size(); ++i) values.push_back(array.at(i));
    return datamunge::format_detail::dstruct_fmt::listing("SortedArray", values);
  }
};

/// @brief Prints a LookupTable as "LookupTable(size=n, {i: v, ...})", listing only the present entries.
template <typename T>
struct std::formatter<datamunge::dstruct::LookupTable<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::LookupTable<T>> {
  /// @brief The table size and its present entries (at most 6 shown).
  /// @param table The table.
  /// @return One line.
  static std::string text(const datamunge::dstruct::LookupTable<T>& table) {
    std::string out = std::format("LookupTable(size={}", table.size());
    if constexpr (std::formattable<T, char>) {
      std::string entries;
      std::size_t present = 0;
      for (std::size_t i = 0; i < table.size(); ++i) {
        if (!table.contains(i)) continue;
        if (present < 6) entries += std::format("{}{}: {}", present ? ", " : "", i, table.at(i));
        ++present;
      }
      if (present > 6) entries += ", ...";
      out += ", {" + entries + "}";
    }
    return out + ")";
  }
};

/// @brief Prints a BitField as "BitField(width=w, bits=0b...)" (bits most-significant first; omitted past 64 bits).
template <>
struct std::formatter<datamunge::dstruct::BitField> : datamunge::format_detail::text_formatter<datamunge::dstruct::BitField> {
  /// @brief The width and, for fields of 1..64 bits, the bit pattern.
  /// @param field The bit field.
  /// @return E.g. "BitField(width=8, bits=0b00000101)".
  static std::string text(const datamunge::dstruct::BitField& field) {
    const std::size_t width = field.width();
    if (width == 0 || width > 64) return std::format("BitField(width={})", width);
    return std::format("BitField(width={}, bits=0b{:0{}b})", width, field.get(0, width), width);
  }
};

/// @brief Prints a Bitboard as "Bitboard(0x...)" with its 64-bit value in hex and the occupied-square count.
template <>
struct std::formatter<datamunge::dstruct::Bitboard> : datamunge::format_detail::text_formatter<datamunge::dstruct::Bitboard> {
  /// @brief The board value.
  /// @param board The bitboard.
  /// @return E.g. "Bitboard(0x0000000000000101, squares=2)".
  static std::string text(const datamunge::dstruct::Bitboard& board) {
    return std::format("Bitboard({:#018x}, squares={})", board.value(), std::popcount(board.value()));
  }
};

/// @brief Prints an AssociationList as "AssociationList{k: v, ...}" in lookup order (or "(size=n)" if unformattable).
template <typename Key, typename Value>
struct std::formatter<datamunge::dstruct::AssociationList<Key, Value>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::AssociationList<Key, Value>> {
  /// @brief The entries, most recently prepended first (shadowed duplicates included).
  /// @param list The association list.
  /// @return One line.
  static std::string text(const datamunge::dstruct::AssociationList<Key, Value>& list) {
    return datamunge::format_detail::dstruct_fmt::mapping("AssociationList", list.to_vector());
  }
};

// ---- associative.hpp -----------------------------------------------------

/// @brief Prints an AssociativeArray as "AssociativeArray{k: v, ...}" in bucket order (or "(size=n)" if unformattable).
template <typename Key, typename Value, typename Hash, typename Equal>
struct std::formatter<datamunge::dstruct::AssociativeArray<Key, Value, Hash, Equal>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::AssociativeArray<Key, Value, Hash, Equal>> {
  /// @brief The entries, in bucket order (first 6 shown).
  /// @param map The hash map.
  /// @return One line.
  static std::string text(const datamunge::dstruct::AssociativeArray<Key, Value, Hash, Equal>& map) {
    return datamunge::format_detail::dstruct_fmt::mapping("AssociativeArray", map.to_vector());
  }
};

/// @brief Prints a Multimap as "Multimap{k: v, ...}" in bucket order (or "(size=n)" if unformattable).
template <typename Key, typename Value, typename Hash, typename Equal>
struct std::formatter<datamunge::dstruct::Multimap<Key, Value, Hash, Equal>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::Multimap<Key, Value, Hash, Equal>> {
  /// @brief The entries, in bucket order (first 6 shown).
  /// @param map The multimap.
  /// @return One line.
  static std::string text(const datamunge::dstruct::Multimap<Key, Value, Hash, Equal>& map) {
    return datamunge::format_detail::dstruct_fmt::mapping("Multimap", map.to_vector());
  }
};

/// @brief Prints a Set as "Set[a, b, ...]" in bucket order (or "Set(size=n)" for unformattable elements).
template <typename T, typename Hash, typename Equal>
struct std::formatter<datamunge::dstruct::Set<T, Hash, Equal>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::Set<T, Hash, Equal>> {
  /// @brief The members, in bucket order.
  /// @param set The set.
  /// @return One line.
  static std::string text(const datamunge::dstruct::Set<T, Hash, Equal>& set) {
    return datamunge::format_detail::dstruct_fmt::listing("Set", set.to_vector());
  }
};

/// @brief Prints a Multiset as "Multiset[a, a, b, ...]" in bucket order (or "Multiset(size=n)" if unformattable).
template <typename T, typename Hash, typename Equal>
struct std::formatter<datamunge::dstruct::Multiset<T, Hash, Equal>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::Multiset<T, Hash, Equal>> {
  /// @brief The members (one per occurrence), in bucket order.
  /// @param set The multiset.
  /// @return One line.
  static std::string text(const datamunge::dstruct::Multiset<T, Hash, Equal>& set) {
    return datamunge::format_detail::dstruct_fmt::listing("Multiset", set.to_vector());
  }
};

/// @brief Prints a BinaryHeap as "BinaryHeap(size=n, top=v)".
template <typename T, typename Compare>
struct std::formatter<datamunge::dstruct::BinaryHeap<T, Compare>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::BinaryHeap<T, Compare>> {
  /// @brief The heap's size and top element.
  /// @param heap The heap.
  /// @return One line (top omitted when empty).
  static std::string text(const datamunge::dstruct::BinaryHeap<T, Compare>& heap) {
    return datamunge::format_detail::dstruct_fmt::heap("BinaryHeap", heap);
  }
};

/// @brief Prints a DisjointSet as "DisjointSet(size=n, sets=k)" (counted without path compression).
template <>
struct std::formatter<datamunge::dstruct::DisjointSet>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::DisjointSet> {
  /// @brief The element count and number of disjoint sets.
  /// @param set The disjoint-set forest (not modified).
  /// @return E.g. "DisjointSet(size=5, sets=3)".
  static std::string text(const datamunge::dstruct::DisjointSet& set) {
    return std::format("DisjointSet(size={}, sets={})", set.size(), set.set_count());
  }
};

// ---- bitmap / bloom / sketch / control table / dope vector ---------------

/// @brief Prints a Bitmap as "Bitmap WxH" followed by its rows as `#`/`.` (capped at 64 x 32).
template <>
struct std::formatter<datamunge::dstruct::Bitmap> : datamunge::format_detail::text_formatter<datamunge::dstruct::Bitmap> {
  /// @brief The size and a picture of the set bits.
  /// @param bitmap The bitmap.
  /// @return Header line, then one line per row.
  static std::string text(const datamunge::dstruct::Bitmap& bitmap) {
    std::string out = std::format("Bitmap {}x{}", bitmap.width(), bitmap.height());
    const std::size_t w = bitmap.width() > 64 ? 64 : bitmap.width();
    const std::size_t h = bitmap.height() > 32 ? 32 : bitmap.height();
    if (w == 0) return out;
    for (std::size_t y = 0; y < h; ++y) {
      out += '\n';
      for (std::size_t x = 0; x < w; ++x) out += bitmap.at(x, y) ? '#' : '.';
      if (w < bitmap.width()) out += "...";
    }
    if (h < bitmap.height()) out += "\n...";
    return out;
  }
};

/// @brief Prints a BloomFilter as "BloomFilter(bits=n, hashes=k)".
template <typename T, typename Hash>
struct std::formatter<datamunge::dstruct::BloomFilter<T, Hash>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::BloomFilter<T, Hash>> {
  /// @brief The filter's bit count and hash count.
  /// @param filter The filter.
  /// @return One line.
  static std::string text(const datamunge::dstruct::BloomFilter<T, Hash>& filter) {
    return std::format("BloomFilter(bits={}, hashes={})", filter.bit_count(), filter.hash_count());
  }
};

/// @brief Prints a CountMinSketch as "CountMinSketch(width=w, depth=d)" (its counters are private).
template <typename T, typename Hash>
struct std::formatter<datamunge::dstruct::CountMinSketch<T, Hash>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::CountMinSketch<T, Hash>> {
  /// @brief The sketch dimensions.
  /// @param sketch The sketch.
  /// @return One line.
  static std::string text(const datamunge::dstruct::CountMinSketch<T, Hash>& sketch) {
    return std::format("CountMinSketch(width={}, depth={})", sketch.width(), sketch.depth());
  }
};

/// @brief Prints a ControlTable as "ControlTable(size=n, enabled=e)[v0, v1, ...]" (values omitted if unformattable).
template <typename T>
struct std::formatter<datamunge::dstruct::ControlTable<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::ControlTable<T>> {
  /// @brief The size, enabled-entry count and value preview.
  /// @param table The control table.
  /// @return One line.
  static std::string text(const datamunge::dstruct::ControlTable<T>& table) {
    std::size_t enabled = 0;
    for (std::size_t i = 0; i < table.size(); ++i) enabled += table.enabled(i) ? 1 : 0;
    std::string out = std::format("ControlTable(size={}, enabled={})", table.size(), enabled);
    if constexpr (std::formattable<T, char>) {
      std::vector<T> values;
      for (std::size_t i = 0; i < table.size(); ++i) values.push_back(table.at(i));
      out += datamunge::format_detail::preview(values);
    }
    return out;
  }
};

/// @brief Prints a DopeVector as "DopeVector<T> r x c" plus a capped value grid, like dstruct::Matrix.
template <typename T>
struct std::formatter<datamunge::dstruct::DopeVector<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::DopeVector<T>> {
  /// @brief The shape and value preview.
  /// @param vector The dope vector.
  /// @return Header line, then the grid (if T is formattable).
  static std::string text(const datamunge::dstruct::DopeVector<T>& vector) {
    namespace g = datamunge::format_detail::grid_fmt;
    std::string head = std::format("DopeVector<{}> {} x {}", datamunge::info_detail::type_name<T>(), vector.rows(),
                                   vector.columns());
    if constexpr (std::formattable<T, char>)
      return g::with_body(head, g::grid(vector.rows(), vector.columns(),
                                        [&](std::size_t i, std::size_t j) { return g::cell_text(vector.at(i, j)); }));
    else return head;
  }
};

// ---- heaps, lists, buffers -------------------------------------------------

/// @brief Prints a DaryHeap as "DaryHeap<k>(size=n, top=v)".
template <typename T, std::size_t Arity, typename Compare>
struct std::formatter<datamunge::dstruct::DaryHeap<T, Arity, Compare>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::DaryHeap<T, Arity, Compare>> {
  /// @brief The heap's arity, size and top element.
  /// @param heap The heap.
  /// @return One line (top omitted when empty).
  static std::string text(const datamunge::dstruct::DaryHeap<T, Arity, Compare>& heap) {
    return datamunge::format_detail::dstruct_fmt::heap(std::format("DaryHeap<{}>", Arity), heap);
  }
};

/// @brief Prints a FreeList as "FreeList(available=n)".
template <typename T>
struct std::formatter<datamunge::dstruct::FreeList<T>> : datamunge::format_detail::text_formatter<datamunge::dstruct::FreeList<T>> {
  /// @brief The number of available items.
  /// @param list The free list.
  /// @return One line.
  static std::string text(const datamunge::dstruct::FreeList<T>& list) {
    return std::format("FreeList(available={})", list.size());
  }
};

/// @brief Prints a GapBuffer as "GapBuffer(cursor=k)[a, b, c]".
template <typename T>
struct std::formatter<datamunge::dstruct::GapBuffer<T>> : datamunge::format_detail::text_formatter<datamunge::dstruct::GapBuffer<T>> {
  /// @brief The cursor position and the buffer contents.
  /// @param buffer The gap buffer.
  /// @return One line.
  static std::string text(const datamunge::dstruct::GapBuffer<T>& buffer) {
    const auto values = buffer.values();
    std::vector<T> copy(values.begin(), values.end());
    if constexpr (std::formattable<T, char>)
      return std::format("GapBuffer(cursor={}){}", buffer.cursor(), datamunge::format_detail::preview(copy));
    else return std::format("GapBuffer(cursor={}, size={})", buffer.cursor(), copy.size());
  }
};

/// @brief Prints a SelfOrganizingList as "SelfOrganizingList[a, b, c]" (or "(size=n)" for unformattable elements).
template <typename T>
struct std::formatter<datamunge::dstruct::SelfOrganizingList<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::SelfOrganizingList<T>> {
  /// @brief The elements front first, without reorganizing the list.
  /// @param list The list.
  /// @return One line.
  static std::string text(const datamunge::dstruct::SelfOrganizingList<T>& list) {
    return datamunge::format_detail::dstruct_fmt::listing("SelfOrganizingList", list.to_vector());
  }
};

/// @brief Prints a VariableLengthArray as "VariableLengthArray[a, b, c]".
template <typename T>
struct std::formatter<datamunge::dstruct::VariableLengthArray<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::VariableLengthArray<T>> {
  /// @brief The array's elements.
  /// @param array The array.
  /// @return One line.
  static std::string text(const datamunge::dstruct::VariableLengthArray<T>& array) {
    std::vector<T> values;
    for (std::size_t i = 0; i < array.size(); ++i) values.push_back(array.at(i));
    return datamunge::format_detail::dstruct_fmt::listing("VariableLengthArray", values);
  }
};

/// @brief Prints a ThreadSafeVector as "ThreadSafeVector[a, b, c]" (from a locked snapshot).
template <typename T>
struct std::formatter<datamunge::dstruct::ThreadSafeVector<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::ThreadSafeVector<T>> {
  /// @brief A snapshot of the elements.
  /// @param vector The vector.
  /// @return One line.
  static std::string text(const datamunge::dstruct::ThreadSafeVector<T>& vector) {
    return datamunge::format_detail::dstruct_fmt::listing("ThreadSafeVector", vector.snapshot());
  }
};

// ---- trees -----------------------------------------------------------------

/// @brief Prints a TernaryTree as "TernaryTree(nodes=n)" plus a preorder outline with "left: "/"middle: "/"right: " labels.
template <typename T>
struct std::formatter<datamunge::dstruct::TernaryTree<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::TernaryTree<T>> {
  /// @brief The node count and a capped outline (first 20 nodes).
  /// @param tree The tree.
  /// @return Header line, then one indented line per node.
  static std::string text(const datamunge::dstruct::TernaryTree<T>& tree) {
    using Node = typename datamunge::dstruct::TernaryTree<T>::Node;
    return datamunge::format_detail::dstruct_fmt::outline("TernaryTree", tree.root(), [](const Node& node) {
      std::vector<std::pair<std::string, const Node*>> kids;
      if (node.left) kids.emplace_back("left: ", node.left.get());
      if (node.middle) kids.emplace_back("middle: ", node.middle.get());
      if (node.right) kids.emplace_back("right: ", node.right.get());
      return kids;
    });
  }
};

/// @brief Prints a KaryTree as "KaryTree<k>(nodes=n)" plus a preorder outline with "[slot] " labels.
template <typename T, std::size_t Arity>
struct std::formatter<datamunge::dstruct::KaryTree<T, Arity>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::KaryTree<T, Arity>> {
  /// @brief The arity, node count and a capped outline (first 20 nodes).
  /// @param tree The tree.
  /// @return Header line, then one indented line per node.
  static std::string text(const datamunge::dstruct::KaryTree<T, Arity>& tree) {
    using Node = typename datamunge::dstruct::KaryTree<T, Arity>::Node;
    return datamunge::format_detail::dstruct_fmt::outline(std::format("KaryTree<{}>", Arity), tree.root(), [](const Node& node) {
      std::vector<std::pair<std::string, const Node*>> kids;
      for (std::size_t i = 0; i < Arity; ++i)
        if (node.children[i]) kids.emplace_back(std::format("[{}] ", i), node.children[i].get());
      return kids;
    });
  }
};

/// @brief Prints a RoseTree as "RoseTree(nodes=n)" plus a preorder outline (children in insertion order).
template <typename T>
struct std::formatter<datamunge::dstruct::RoseTree<T>> : datamunge::format_detail::text_formatter<datamunge::dstruct::RoseTree<T>> {
  /// @brief The node count and a capped outline (first 20 nodes).
  /// @param tree The tree.
  /// @return Header line, then one indented line per node.
  static std::string text(const datamunge::dstruct::RoseTree<T>& tree) {
    using Node = typename datamunge::dstruct::RoseTree<T>::Node;
    return datamunge::format_detail::dstruct_fmt::outline("RoseTree", tree.root(), [](const Node& node) {
      std::vector<std::pair<std::string, const Node*>> kids;
      for (const auto& child : node.children) kids.emplace_back("", child.get());
      return kids;
    });
  }
};

/// @brief Prints a BinarySearchTree as "BinarySearchTree(size=n, min=v)".
template <typename T, typename Compare>
struct std::formatter<datamunge::dstruct::BinarySearchTree<T, Compare>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::BinarySearchTree<T, Compare>> {
  /// @brief The tree's size and minimum.
  /// @param tree The tree.
  /// @return One line (min omitted when empty).
  static std::string text(const datamunge::dstruct::BinarySearchTree<T, Compare>& tree) {
    if constexpr (std::formattable<T, char>) {
      if (const auto min = tree.minimum()) return std::format("BinarySearchTree(size={}, min={})", tree.size(), *min);
    }
    return std::format("BinarySearchTree(size={})", tree.size());
  }
};

/// @brief Prints a Trie as "Trie(words=n)".
template <>
struct std::formatter<datamunge::dstruct::Trie> : datamunge::format_detail::text_formatter<datamunge::dstruct::Trie> {
  /// @brief The number of stored words.
  /// @param trie The trie.
  /// @return One line.
  static std::string text(const datamunge::dstruct::Trie& trie) { return std::format("Trie(words={})", trie.size()); }
};

/// @brief Prints a FenwickTree as "FenwickTree[v0, v1, ...]", the point values recovered from range sums.
template <typename T>
struct std::formatter<datamunge::dstruct::FenwickTree<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::FenwickTree<T>> {
  /// @brief The point values.
  /// @param tree The tree.
  /// @return One line.
  static std::string text(const datamunge::dstruct::FenwickTree<T>& tree) {
    std::vector<T> values;
    for (std::size_t i = 0; i < tree.size(); ++i) values.push_back(tree.range_sum(i, i + 1));
    return datamunge::format_detail::dstruct_fmt::listing("FenwickTree", values);
  }
};

/// @brief Prints a SegmentTree as "SegmentTree(size=n, total=v)[leaf0, leaf1, ...]" (or "(size=n)" if unformattable).
template <typename T, typename Combine>
struct std::formatter<datamunge::dstruct::SegmentTree<T, Combine>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::SegmentTree<T, Combine>> {
  /// @brief The size, the combined value over the whole range, and the leaf values.
  /// @param tree The tree.
  /// @return One line.
  static std::string text(const datamunge::dstruct::SegmentTree<T, Combine>& tree) {
    if constexpr (std::formattable<T, char>) {
      std::vector<T> leaves;
      for (std::size_t i = 0; i < tree.size(); ++i) leaves.push_back(tree.query(i, i + 1));
      return std::format("SegmentTree(size={}, total={})", tree.size(), tree.query(0, tree.size())) +
             datamunge::format_detail::preview(leaves);
    } else {
      return std::format("SegmentTree(size={})", tree.size());
    }
  }
};

// ---- matrix.hpp / image.hpp -----------------------------------------------

/// @brief Prints a dstruct::Matrix as "Matrix<T> r x c" plus a capped value grid.
template <typename T>
struct std::formatter<datamunge::dstruct::Matrix<T>> : datamunge::format_detail::text_formatter<datamunge::dstruct::Matrix<T>> {
  /// @brief The shape and value preview.
  /// @param m The matrix.
  /// @return Header line, then the grid (if T is formattable).
  static std::string text(const datamunge::dstruct::Matrix<T>& m) {
    namespace g = datamunge::format_detail::grid_fmt;
    std::string head = std::format("Matrix<{}> {} x {}", datamunge::info_detail::type_name<T>(), m.rows(), m.columns());
    if constexpr (std::formattable<T, char>)
      return g::with_body(head, g::grid(m.rows(), m.columns(), [&](std::size_t i, std::size_t j) { return g::cell_text(m.at(i, j)); }));
    else return head;
  }
};

/// @brief Prints a dstruct::SparseMatrix as "SparseMatrix<T> r x c, nnz=n" plus a grid with `.` for zeros.
template <typename T>
struct std::formatter<datamunge::dstruct::SparseMatrix<T>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::SparseMatrix<T>> {
  /// @brief The shape, non-zero count and value preview.
  /// @param m The matrix.
  /// @return Header line, then the grid (if T is formattable).
  static std::string text(const datamunge::dstruct::SparseMatrix<T>& m) {
    namespace g = datamunge::format_detail::grid_fmt;
    std::string head = std::format("SparseMatrix<{}> {} x {}, nnz={}", datamunge::info_detail::type_name<T>(), m.rows(),
                                   m.columns(), m.nonzero_count());
    if constexpr (std::formattable<T, char>)
      return g::with_body(head, g::grid(m.rows(), m.columns(), [&](std::size_t i, std::size_t j) {
                            const T value = m.at(i, j);
                            return value == T{} ? std::string(".") : g::cell_text(value);
                          }));
    else return head;
  }
};

/// @brief Prints a ParallelArray as "ParallelArray{k1: v1, k2: v2, ...}".
template <typename Key, typename Value>
struct std::formatter<datamunge::dstruct::ParallelArray<Key, Value>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::ParallelArray<Key, Value>> {
  /// @brief The key/value pairs (at most 6 shown).
  /// @param array The parallel array.
  /// @return One line.
  static std::string text(const datamunge::dstruct::ParallelArray<Key, Value>& array) {
    if constexpr (std::formattable<Key, char> && std::formattable<Value, char>) {
      std::string out = "ParallelArray{";
      for (std::size_t i = 0; i < array.size() && i < 6; ++i)
        out += std::format("{}{}: {}", i ? ", " : "", array.key_at(i), array.value_at(i));
      if (array.size() > 6) out += std::format(", ...}} (n={})", array.size());
      else out += "}";
      return out;
    } else {
      return std::format("ParallelArray(size={})", array.size());
    }
  }
};

/// @brief Prints a dstruct::Image as "Image WxH" plus a capped pixel grid (if pixels are formattable).
template <typename Pixel>
struct std::formatter<datamunge::dstruct::Image<Pixel>> : datamunge::format_detail::text_formatter<datamunge::dstruct::Image<Pixel>> {
  /// @brief The size and pixel preview (rows are y, columns are x).
  /// @param image The image.
  /// @return Header line, then the grid.
  static std::string text(const datamunge::dstruct::Image<Pixel>& image) {
    namespace g = datamunge::format_detail::grid_fmt;
    std::string head = std::format("Image {}x{}", image.width(), image.height());
    if constexpr (std::formattable<Pixel, char>)
      return g::with_body(head, g::grid(image.height(), image.width(),
                                        [&](std::size_t y, std::size_t x) { return g::cell_text(image.at(x, y)); }));
    else return head;
  }
};

// ---- value_types.hpp ------------------------------------------------------

/// @brief Prints a ProductType as "ProductType(first, second)".
template <typename First, typename Second>
struct std::formatter<datamunge::dstruct::ProductType<First, Second>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::ProductType<First, Second>> {
  /// @brief The two fields.
  /// @param p The pair.
  /// @return One line ("ProductType" alone if a field is not formattable).
  static std::string text(const datamunge::dstruct::ProductType<First, Second>& p) {
    if constexpr (std::formattable<First, char> && std::formattable<Second, char>)
      return std::format("ProductType({}, {})", p.first, p.second);
    else return "ProductType";
  }
};

/// @brief Prints a Tuple as "Tuple(a, b, c)".
template <typename... Fields>
struct std::formatter<datamunge::dstruct::Tuple<Fields...>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::Tuple<Fields...>> {
  /// @brief The fields in order.
  /// @param t The tuple.
  /// @return One line ("Tuple" alone if a field is not formattable).
  static std::string text(const datamunge::dstruct::Tuple<Fields...>& t) {
    if constexpr ((std::formattable<Fields, char> && ...)) {
      std::string out = "Tuple(";
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((out += std::format("{}{}", I ? ", " : "", t.template get<I>())), ...);
      }(std::index_sequence_for<Fields...>{});
      return out + ")";
    } else {
      return "Tuple";
    }
  }
};

/// @brief Prints a TaggedUnion as "TaggedUnion(index=i, value=v)" (value omitted unless every alternative is formattable).
template <typename... Alternatives>
struct std::formatter<datamunge::dstruct::TaggedUnion<Alternatives...>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::TaggedUnion<Alternatives...>> {
  /// @brief The held alternative's index and value.
  /// @param u The tagged union.
  /// @return E.g. "TaggedUnion(index=1, value=hi)".
  static std::string text(const datamunge::dstruct::TaggedUnion<Alternatives...>& u) {
    if constexpr ((std::formattable<Alternatives, char> && ...)) {
      if (!u.value().valueless_by_exception())
        return std::visit([&](const auto& v) { return std::format("TaggedUnion(index={}, value={})", u.index(), v); },
                          u.value());
    }
    return std::format("TaggedUnion(index={})", u.index());
  }
};

/// @brief Prints a Union as "Union(v)" or "Union(empty)".
template <typename T>
struct std::formatter<datamunge::dstruct::Union<T>> : datamunge::format_detail::text_formatter<datamunge::dstruct::Union<T>> {
  /// @brief The held value.
  /// @param u The union.
  /// @return One line.
  static std::string text(const datamunge::dstruct::Union<T>& u) {
    if (!u.value()) return "Union(empty)";
    if constexpr (std::formattable<T, char>) return std::format("Union({})", *u.value());
    else return "Union(set)";
  }
};

// ---- graphs ----------------------------------------------------------------

/// @brief Prints a DirectedGraph as a "DirectedGraph(vertices=n, edges=m)" header plus "v -> [out-neighbours]" lines.
template <typename Vertex>
struct std::formatter<datamunge::dstruct::DirectedGraph<Vertex>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::DirectedGraph<Vertex>> {
  /// @brief The graph's size and adjacency (first 10 vertices).
  /// @param graph The graph.
  /// @return Multi-line text (header only if vertices are not formattable).
  static std::string text(const datamunge::dstruct::DirectedGraph<Vertex>& graph) {
    return datamunge::format_detail::dstruct_fmt::adjacency(
        std::format("DirectedGraph(vertices={}, edges={})", graph.vertex_count(), graph.edge_count()),
        graph.vertices(), "->", [&](const Vertex& v) { return datamunge::format_detail::preview(graph.outgoing_neighbors(v), 10); });
  }
};

/// @brief Prints a DirectedAcyclicGraph as "DirectedAcyclicGraph(vertices=n, edges=m, order=[...])".
template <typename Vertex>
struct std::formatter<datamunge::dstruct::DirectedAcyclicGraph<Vertex>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::DirectedAcyclicGraph<Vertex>> {
  /// @brief The graph's size and a topological order of its vertices.
  /// @param graph The DAG.
  /// @return One line (order omitted if vertices are not formattable).
  static std::string text(const datamunge::dstruct::DirectedAcyclicGraph<Vertex>& graph) {
    std::string out = std::format("DirectedAcyclicGraph(vertices={}, edges={}", graph.vertex_count(), graph.edge_count());
    if constexpr (std::formattable<Vertex, char>)
      out += ", order=" + datamunge::format_detail::preview(graph.topological_order(), 10);
    return out + ")";
  }
};

/// @brief Prints an UndirectedGraph as an "UndirectedGraph(vertices=n, edges=m)" header plus "v -- [neighbours]" lines.
template <typename Vertex>
struct std::formatter<datamunge::dstruct::UndirectedGraph<Vertex>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::UndirectedGraph<Vertex>> {
  /// @brief The graph's size and adjacency (first 10 vertices).
  /// @param graph The graph.
  /// @return Multi-line text (header only if vertices are not formattable).
  static std::string text(const datamunge::dstruct::UndirectedGraph<Vertex>& graph) {
    return datamunge::format_detail::dstruct_fmt::adjacency(
        std::format("UndirectedGraph(vertices={}, edges={})", graph.vertex_count(), graph.edge_count()),
        graph.vertices(), "--", [&](const Vertex& v) { return datamunge::format_detail::preview(graph.neighbors(v), 10); });
  }
};

/// @brief Prints a WeightedGraph as a "WeightedGraph(directed, vertices=n, edges=m)" header plus
///        "v -> [u (w), ...]" adjacency lines.
template <typename Vertex, typename Weight>
struct std::formatter<datamunge::dstruct::WeightedGraph<Vertex, Weight>>
    : datamunge::format_detail::text_formatter<datamunge::dstruct::WeightedGraph<Vertex, Weight>> {
  /// @brief The graph's kind, size and weighted adjacency (first 10 vertices).
  /// @param graph The graph.
  /// @return Multi-line text (header only if vertices are not formattable).
  static std::string text(const datamunge::dstruct::WeightedGraph<Vertex, Weight>& graph) {
    return datamunge::format_detail::dstruct_fmt::adjacency(
        std::format("WeightedGraph({}, vertices={}, edges={})", graph.directed() ? "directed" : "undirected",
                    graph.vertex_count(), graph.edge_count()),
        graph.vertices(), graph.directed() ? "->" : "--", [&](const Vertex& v) {
          std::vector<std::string> edges;
          if constexpr (std::formattable<Weight, char>)
            for (const auto& n : graph.neighbors(v)) edges.push_back(std::format("{} ({})", n, graph.edge_weight(v, n)));
          else
            for (const auto& n : graph.neighbors(v)) edges.push_back(std::format("{}", n));
          std::string out = "[";
          for (std::size_t i = 0; i < edges.size() && i < 10; ++i) out += (i ? ", " : "") + edges[i];
          if (edges.size() > 10) out += ", ...";
          return out + "]";
        });
  }
};
