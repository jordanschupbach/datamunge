/// @file tape.cpp
/// @brief tape(), Tape's printers and plot_tape().

#include "datamunge/nn/tape.hpp"

#include "datamunge/nn/autograd.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief Formats a shape as `[2, 3]`.
/// @param shape  The shape.
/// @return The text.
std::string shape_text(const Shape& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) out += (i ? ", " : "") + std::to_string(shape[i]);
    return out + "]";
}

}  // namespace

Tape tape(const std::vector<Tensor>& outputs, const std::vector<std::pair<Tensor, std::string>>& names) {
    std::map<const TensorImpl*, std::string> leaf_names;
    for (const auto& [tensor, name] : names)
        if (tensor.defined()) leaf_names[tensor.impl()] = name;

    // Collect every node reachable from the outputs.
    std::vector<std::shared_ptr<Node>>       found;
    std::unordered_map<const Node*, bool>    seen;
    std::vector<std::shared_ptr<Node>>       stack;
    for (const auto& out : outputs)
        if (out.defined())
            if (auto n = gradient_edge(out).node) stack.push_back(n);
    while (!stack.empty()) {
        auto n = stack.back();
        stack.pop_back();
        if (!n || seen[n.get()]) continue;
        seen[n.get()] = true;
        found.push_back(n);
        for (const auto& e : n->next_edges())
            if (e.node) stack.push_back(e.node);
    }
    // Reverse recording order is a topological order (an op is recorded after its inputs).
    std::stable_sort(found.begin(), found.end(),
                     [](const auto& a, const auto& b) { return a->sequence_nr() > b->sequence_nr(); });

    Tape                                    t;
    std::unordered_map<const Node*, std::size_t> index;
    std::size_t                             unnamed = 0;
    for (std::size_t i = 0; i < found.size(); ++i) {
        const auto& n = found[i];
        index[n.get()] = i;
        TapeNode node;
        node.id          = i;
        node.sequence_nr = n->sequence_nr();
        if (auto acc = std::dynamic_pointer_cast<AccumulateGrad>(n)) {
            node.leaf = true;
            node.op   = "Leaf";
            const auto variable = acc->variable();
            const auto it       = variable ? leaf_names.find(variable.get()) : leaf_names.end();
            node.label = it != leaf_names.end() ? it->second : "leaf_" + std::to_string(unnamed++);
            if (variable) node.shape = variable->shape;
        } else {
            node.op    = n->name();
            node.label = n->name() + "_" + std::to_string(n->sequence_nr());
            if (!n->output_meta().empty()) node.shape = n->output_meta()[0].shape;
        }
        t.nodes.push_back(std::move(node));
    }
    for (std::size_t i = 0; i < found.size(); ++i)
        for (const auto& e : found[i]->next_edges())
            t.nodes[i].inputs.push_back(e.node ? std::optional<std::size_t>(index.at(e.node.get())) : std::nullopt);
    // Longest path from the outputs: nodes are in topological order, so one forward pass.
    for (const auto& node : t.nodes)
        for (const auto& in : node.inputs)
            if (in) t.nodes[*in].depth = std::max(t.nodes[*in].depth, node.depth + 1);
    return t;
}

std::size_t Tape::op_count() const {
    return static_cast<std::size_t>(std::count_if(nodes.begin(), nodes.end(), [](const auto& n) { return !n.leaf; }));
}

std::size_t Tape::leaf_count() const { return nodes.size() - op_count(); }

std::string Tape::to_text() const {
    std::ostringstream out;
    for (const auto& n : nodes) {
        out << n.label << " [" << n.op << "] " << shape_text(n.shape);
        if (!n.leaf) {
            out << " <-";
            for (const auto& in : n.inputs) out << ' ' << (in ? nodes[*in].label : std::string("const"));
        }
        out << '\n';
    }
    return out.str();
}

std::string Tape::to_dot() const {
    std::ostringstream out;
    out << "digraph tape {\n  rankdir=BT;\n";
    for (const auto& n : nodes)
        out << "  n" << n.id << " [label=\"" << n.label << "\\n" << shape_text(n.shape) << "\""
            << (n.leaf ? ", shape=box" : "") << "];\n";
    for (const auto& n : nodes)
        for (const auto& in : n.inputs)
            if (in) out << "  n" << *in << " -> n" << n.id << ";\n";
    out << "}\n";
    return out.str();
}

graphics::RPlot plot_tape(const Tape& t, const std::string& title) {
    if (t.nodes.empty()) throw std::invalid_argument("plot_tape: the tape is empty");
    std::size_t max_depth = 0;
    for (const auto& n : t.nodes) max_depth = std::max(max_depth, n.depth);
    // One row per depth. Order each row by the barycentre of its neighbours in the row above
    // (downward sweep) and below (upward sweep), a few times, to reduce edge crossings.
    std::vector<std::vector<std::size_t>> rows(max_depth + 1);
    for (const auto& n : t.nodes) rows[n.depth].push_back(n.id);
    std::vector<std::vector<std::size_t>> users(t.nodes.size());
    for (const auto& n : t.nodes)
        for (const auto& in : n.inputs)
            if (in) users[*in].push_back(n.id);
    std::vector<double> rank(t.nodes.size());
    auto renumber = [&](const std::vector<std::size_t>& row) {
        for (std::size_t k = 0; k < row.size(); ++k) rank[row[k]] = static_cast<double>(k) / std::max<std::size_t>(1, row.size() - 1);
    };
    for (const auto& r : rows) renumber(r);
    auto reorder = [&](std::vector<std::size_t>& row, bool from_above) {
        std::vector<std::pair<double, std::size_t>> keyed;
        for (const std::size_t id : row) {
            double total = 0;
            std::size_t count = 0;
            if (from_above) {
                for (const std::size_t u : users[id]) total += rank[u], ++count;
            } else {
                for (const auto& in : t.nodes[id].inputs)
                    if (in) total += rank[*in], ++count;
            }
            keyed.emplace_back(count ? total / static_cast<double>(count) : rank[id], id);
        }
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t k = 0; k < row.size(); ++k) row[k] = keyed[k].second;
        renumber(row);
    };
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (std::size_t d = 1; d < rows.size(); ++d) reorder(rows[d], true);
        for (std::size_t d = rows.size() - 1; d-- > 0;) reorder(rows[d], false);
    }
    std::size_t widest = 1;
    for (const auto& r : rows) widest = std::max(widest, r.size());
    std::vector<double> px(t.nodes.size()), py(t.nodes.size());
    for (std::size_t d = 0; d < rows.size(); ++d)
        for (std::size_t k = 0; k < rows[d].size(); ++k) {
            px[rows[d][k]] = (static_cast<double>(k) + 0.5) * static_cast<double>(widest) / static_cast<double>(rows[d].size());
            py[rows[d][k]] = -static_cast<double>(d);
        }
    auto plot = graphics::RPlot::create();
    std::vector<double> x0, y0, x1, y1;
    for (const auto& n : t.nodes)
        for (const auto& in : n.inputs)
            if (in) x0.push_back(px[n.id]), y0.push_back(py[n.id]), x1.push_back(px[*in]), y1.push_back(py[*in]);
    if (!x0.empty()) plot.segments(x0, y0, x1, y1, graphics::RGB{148, 163, 184}, 1.5);
    std::vector<double> ox, oy, lx, ly;
    for (const auto& n : t.nodes) (n.leaf ? lx : ox).push_back(px[n.id]), (n.leaf ? ly : oy).push_back(py[n.id]);
    if (!ox.empty()) plot.points(ox, oy, "op", graphics::RGB{17, 24, 39}, 10.0);
    if (!lx.empty()) plot.points(lx, ly, "leaf", graphics::RGB{37, 99, 235}, 10.0);
    for (const auto& n : t.nodes) plot.text(px[n.id], py[n.id] + 0.28, n.label, graphics::RGB{17, 24, 39}, 13.0);
    const double depth = static_cast<double>(max_depth);
    plot.x_limits(0.0, static_cast<double>(widest)).y_limits(-depth - 0.5, 1.2).hide_axes().title(title);
    plot.size(std::min<std::size_t>(1100, 200 + 140 * widest), 160 + 110 * (max_depth + 1));
    return plot;
}

}  // namespace datamunge::nn
