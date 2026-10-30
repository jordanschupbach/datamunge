#pragma once

/// @file tape.hpp
/// @brief Inspecting the autograd tape: the graph of recorded operations behind a tensor.
///
/// Every op that runs on a tensor requiring gradients records a Node (its name, the
/// edges its gradient flows along, and a closure for the backward step). tape() walks
/// those nodes back from an output and returns them as plain data, in an order in which
/// each node comes before the nodes it depends on, so the graph can be printed, counted,
/// tabulated or drawn (plot_tape()).

#include "datamunge/nn/tensor.hpp"
#include "datamunge/plot/plot.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::nn {

/// @brief One node of a recorded tape: an operation, or a leaf tensor that accumulates a gradient.
struct TapeNode {
    /// Position in Tape::nodes (0 is the output's node).
    std::size_t id{0};
    /// The op name ("Add", "MatMul", ...), or "Leaf" for a leaf tensor's gradient accumulator.
    std::string op;
    /// Display label: the op name with its sequence number ("Add_4"), or the leaf's name.
    std::string label;
    /// The order in which the node was recorded; later ops have larger numbers.
    std::uint64_t sequence_nr{0};
    /// True for a leaf tensor (requires_grad, no producing op).
    bool leaf{false};
    /// One entry per op input: the node the gradient flows to, or empty for an input that
    /// needs no gradient (a constant).
    std::vector<std::optional<std::size_t>> inputs;
    /// Shape of the node's (first) output; for a leaf, the leaf's shape.
    Shape shape;
    /// Length of the longest path from the output's node (0 for the output itself).
    std::size_t depth{0};
};

/// @brief The recorded graph behind one or more outputs, as returned by tape().
struct Tape {
    /// Nodes in reverse recording order: every node precedes the nodes its inputs come from.
    std::vector<TapeNode> nodes;

    /// @brief Number of op nodes (leaves excluded).
    /// @return The count.
    [[nodiscard]] std::size_t op_count() const;
    /// @brief Number of leaf nodes.
    /// @return The count.
    [[nodiscard]] std::size_t leaf_count() const;
    /// @brief One line per node, `label [op] shape <- input labels` (constants shown as `const`).
    /// @return The listing, output first.
    [[nodiscard]] std::string to_text() const;
    /// @brief The graph in Graphviz DOT, edges pointing in the forward direction (input to op).
    /// @return DOT source.
    [[nodiscard]] std::string to_dot() const;
};

/// @brief Collects the tape that produced `outputs`.
///
/// Leaves are labelled with the names given in `names` (matched by identity, so pass the
/// leaf tensors themselves); unnamed leaves are labelled "leaf_<n>".
/// @param outputs  Tensors whose history to collect; ones without history contribute nothing.
/// @param names  Optional (leaf tensor, name) pairs.
/// @return The nodes; empty if no output has a recorded history.
/// @par Example
/// @code
/// auto x = nn::scalar(2.0f).requires_grad_();
/// auto y = nn::pow(x, 3.0) + 2.0 * x;
/// std::print("{}", nn::tape({y}, {{x, "x"}}).to_text());
/// // Add_4 [Add] [] <- PowScalar_3 Mul_2
/// // PowScalar_3 [PowScalar] [] <- x
/// // Mul_2 [Mul] [] <- const x
/// // x [Leaf] []
/// @endcode
[[nodiscard]] Tape tape(const std::vector<Tensor>& outputs,
                        const std::vector<std::pair<Tensor, std::string>>& names = {});

/// @brief Draws a tape as a layered graph: the output at the top, one row per depth, leaves
///        in blue, ops in black, an edge from each node to each of its recorded inputs. Nodes in
///        each row are ordered by the barycentre of their neighbours to reduce edge crossings.
/// @param t  The tape.
/// @param title  Plot title.
/// @return A plot sized to fit the graph; save or add to a layout.
[[nodiscard]] graphics::RPlot plot_tape(const Tape& t, const std::string& title = "Autograd tape");

}  // namespace datamunge::nn
