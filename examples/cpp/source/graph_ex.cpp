#include <datamunge/dstruct/dstruct.hpp>

#include <iostream>

using datamunge::dstruct::UndirectedGraph;
using datamunge::dstruct::WeightedGraph;

namespace {
void print_path(const std::vector<std::string>& path) {
    for (std::size_t i = 0; i < path.size(); ++i) std::cout << (i ? " -> " : "") << path[i];
    std::cout << "\n";
}
} // namespace

int main() {
    std::cout << "=================== UndirectedGraph: a small social network ===================\n";
    UndirectedGraph<std::string> friends({"Alice", "Bob", "Carol", "Dave", "Erin"});
    friends.add_edge("Alice", "Bob");
    friends.add_edge("Bob", "Carol");
    friends.add_edge("Carol", "Alice"); // closes a triangle -> a cycle
    friends.add_edge("Dave", "Erin");   // a separate component

    std::cout << "vertices: " << friends.vertex_count() << ", edges: " << friends.edge_count() << "\n";
    std::cout << "Bob's friends: ";
    for (const auto& f : friends.neighbors("Bob")) std::cout << f << " ";
    std::cout << "\nhas a cycle: " << std::boolalpha << friends.has_cycle() << "\n";
    std::cout << "connected components: " << friends.connected_components().size() << " (expect 2)\n";
    std::cout << "Alice can reach Carol: " << friends.contains_path("Alice", "Carol") << "\n";
    std::cout << "Alice can reach Dave: " << friends.contains_path("Alice", "Dave") << "\n\n";

    std::cout << "=================== WeightedGraph: shortest paths on a small road network ===================\n";
    WeightedGraph<std::string, double> roads(/*directed=*/false);
    roads.add_edge("A", "B", 4.0);
    roads.add_edge("A", "C", 1.0);
    roads.add_edge("C", "B", 2.0);
    roads.add_edge("B", "D", 5.0);
    roads.add_edge("C", "D", 8.0);
    roads.add_edge("D", "E", 3.0);

    std::cout << "Dijkstra distances from A:\n";
    for (const auto& [vertex, dist] : roads.dijkstra_distances("A")) std::cout << "  A -> " << vertex << ": " << dist << "\n";

    const auto paths = roads.dijkstra("A");
    std::cout << "Shortest path A -> E: ";
    print_path(paths.path_to("E"));

    std::cout << "Bellman-Ford distances from A (same graph, no negative edges): ";
    bool matches = true;
    for (const auto& [vertex, dist] : roads.bellman_ford_distances("A")) {
        const auto dijkstra_dist = roads.dijkstra_distances("A");
        for (const auto& [v2, d2] : dijkstra_dist)
            if (v2 == vertex && d2 != dist) matches = false;
    }
    std::cout << (matches ? "matches Dijkstra, as expected" : "MISMATCH") << "\n";

    const auto mst = roads.minimum_spanning_tree();
    std::cout << "Minimum spanning tree has " << mst.edge_count() << " edges (expect vertices-1 = "
              << (roads.vertex_count() - 1) << ")\n\n";

    std::cout << "=================== WeightedGraph: cycle + strongly-connected-components on a directed graph ==="
              << "================\n";
    WeightedGraph<std::string, double> deps(/*directed=*/true);
    deps.add_edge("compile", "link", 1.0);
    deps.add_edge("link", "test", 1.0);
    deps.add_edge("test", "compile", 1.0); // a feedback loop: a real cycle in a build pipeline
    deps.add_edge("test", "deploy", 1.0);

    std::cout << "has a cycle: " << deps.has_cycle() << "\n";
    const auto sccs = deps.strongly_connected_components();
    std::cout << "strongly connected components: " << sccs.size() << "\n";
    for (const auto& scc : sccs) {
        std::cout << "  {";
        for (std::size_t i = 0; i < scc.size(); ++i) std::cout << (i ? ", " : "") << scc[i];
        std::cout << "}\n";
    }

    return 0;
}
