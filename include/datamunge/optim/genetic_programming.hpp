#pragma once

/// @file
/// @brief Genetic search of tree, Cartesian and gene-expression programs.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace datamunge::optim {

/// @brief Protected scalar operations available to evolved programs.
/// @par Example
/// @code
/// const auto code = dm::optim::ProgramOperation::multiply;
/// @endcode
/// @see GeneticProgram
enum class ProgramOperation {
  constant,   ///< Finite literal.
  add,        ///< Sum of two earlier slots.
  subtract,   ///< Difference of two earlier slots.
  multiply,   ///< Product of two earlier slots.
  divide,     ///< Protected division; return numerator when absolute denominator is below 1e-12.
  sine,       ///< Sine of the first slot in radians.
  cosine      ///< Cosine of the first slot in radians.
};

/// @brief One feed-forward instruction of a compiled program.
/// @par Example
/// @code
/// const dm::optim::ProgramInstruction instruction{dm::optim::ProgramOperation::add, 0, 1};
/// @endcode
/// @see GeneticProgram
struct ProgramInstruction {
  ProgramOperation operation{ProgramOperation::constant}; ///< Instruction symbol.
  std::size_t first{}; ///< First operand index in inputs followed by earlier nodes.
  std::size_t second{}; ///< Second operand index when the operation is binary.
  double literal{}; ///< Constant when the operation is constant.
};

/// @brief Immutable scalar program using protected feed-forward arithmetic.
/// @par Example
/// @code
/// const dm::optim::GeneticProgram program(1, {{dm::optim::ProgramOperation::multiply, 0, 0}}, 1);
/// const std::array input{3.0};
/// std::print("{}\n", program.evaluate(input)); // 9
/// @endcode
/// @see genetic_programming_search()
class GeneticProgram {
public:
  /// @brief Validate and own a compiled scalar expression graph.
  /// @param inputs Number of input slots, at least one.
  /// @param instructions Feed-forward nodes whose operands reference earlier slots.
  /// @param output Result slot index among all inputs and computed nodes.
  /// @throws std::invalid_argument Invalid shape, operand, literal or operation.
  /// @par Example
  /// @code
  /// const dm::optim::GeneticProgram identity(1, {}, 0);
  /// @endcode
  /// @see evaluate()
  GeneticProgram(std::size_t inputs, std::vector<ProgramInstruction> instructions, std::size_t output);

  /// @brief Evaluate the program on finite input.
  /// Inputs are copied and intermediate values saturate at finite double
  /// extrema. Division is protected near a zero denominator.
  /// @param input Finite values, one per input slot.
  /// @return Finite value of the output slot.
  /// @throws std::invalid_argument Input has the wrong dimension or is nonfinite.
  /// @throws std::out_of_range The output slot is unavailable after moving from the program.
  /// @par Example
  /// @code
  /// const dm::optim::GeneticProgram identity(1, {}, 0);
  /// const std::array input{7.0};
  /// std::print("{}\n", identity.evaluate(input)); // 7
  /// @endcode
  /// @see instructions(), output_slot()
  [[nodiscard]] double evaluate(std::span<const double> input) const;

  /// @brief Borrow the compiled feed-forward instructions.
  /// @return Read-only instructions valid until assignment or destruction.
  [[nodiscard]] const std::vector<ProgramInstruction>& instructions() const noexcept { return instructions_; }

  /// @brief Return the index of the returned scalar slot.
  /// @return Zero-based slot among inputs followed by the instructions.
  [[nodiscard]] std::size_t output_slot() const noexcept { return output_; }

  /// @brief Return the required input dimension.
  /// @return Positive input count.
  [[nodiscard]] std::size_t input_count() const noexcept { return inputs_; }

private:
  std::size_t inputs_; ///< Number of input slots.
  std::vector<ProgramInstruction> instructions_; ///< Owned validated graph nodes.
  std::size_t output_; ///< Selected graph output.
};

/// @brief Genetic representation and variation rule.
/// @par Example
/// @code
/// const auto method = dm::optim::ProgramSearchMethod::gene_expression;
/// @endcode
/// @see ProgramSearchOptions
enum class ProgramSearchMethod {
  tree,           ///< Variable-size prefix tree with subtree crossover and mutation.
  cartesian,      ///< Feed-forward graph chromosome with neutral-drift (1+lambda) mutation.
  gene_expression ///< Linear head and terminal tail, expressed breadth-first.
};

/// @brief Budget, representation size and variation controls.
/// @par Example
/// @code
/// dm::optim::ProgramSearchOptions options;
/// options.method = dm::optim::ProgramSearchMethod::cartesian;
/// @endcode
/// @see genetic_programming_search()
struct ProgramSearchOptions {
  std::size_t evaluations{1000}; ///< Positive count of program fitness callbacks.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  ProgramSearchMethod method{ProgramSearchMethod::tree}; ///< Genotype type and evolutionary rule.
  std::size_t input_count{1}; ///< Positive number of scalar input slots.
  std::size_t size{6}; ///< Max tree height (1–12), CGP node count or GEP head length.
  std::size_t population{32}; ///< Generational population, or CGP offspring per generation; at least two.
  double mutation{0.15}; ///< Tree mutation probability or pointwise gene mutation rate, in [0,1].
  double crossover{0.8}; ///< Tree subtree or GEP one-point recombination probability, in [0,1].
};

/// @brief Borrowed fitness evaluation of an expressed program.
using ProgramObjective = std::function<double(const GeneticProgram&)>;

/// @brief Best expression and its actual program fitness-call count.
/// @par Example
/// @code
/// const dm::optim::ProgramSearchResult result{dm::optim::GeneticProgram(1, {}, 0), 0, 1};
/// @endcode
/// @see genetic_programming_search()
struct ProgramSearchResult {
  GeneticProgram program; ///< Earliest program with the best observed fitness.
  double value; ///< Objective measured at the retained program.
  std::size_t evaluations; ///< Actual program-level fitness evaluations.
};

/// @brief Evolve a protected scalar program under a fixed fitness-call budget.
///
/// Tree GP applies subtree operations with a maximum height and one elite.
/// Cartesian GP mutates feed-forward node genes and the output reference,
/// retaining a non-worse child to allow neutral drift. GEP uses one gene
/// with a function-capable head and a terminal-only tail of head length
/// plus one. It applies point mutation, one-point recombination and short
/// head transpositions, then expresses the used symbols breadth-first.
/// All methods score an explicit zero constant first. Duplicate fitness
/// calls count. The objective owns its dataset and can evaluate a program
/// on multiple samples inside one call. Exceptions propagate and replay
/// holds within a build for deterministic objectives.
/// @param objective Borrowed finite-return program fitness to minimize.
/// @param options Representation, arithmetic shape, budget and variation settings.
/// @return Best compiled expression, its loss and actual evaluation count.
/// @throws std::invalid_argument Invalid shape, options or empty objective.
/// @throws std::domain_error Objective returns nonfinite fitness.
/// @par Example
/// @code
/// const std::array input{0.0};
/// const auto result = dm::optim::genetic_programming_search(
///     [&](const auto& expression) { return std::abs(expression.evaluate(input)); }, {10});
/// std::print("{}\n", result.value); // 0
/// @endcode
/// @par References
/// - Koza (1992). https://mitpress.mit.edu/9780262527910/genetic-programming/
/// - Miller and Thomson (2000). https://doi.org/10.1007/978-3-540-46239-2_9
/// - Ferreira (2001). https://arxiv.org/abs/cs/0102027
/// @see GeneticProgram, discrete_genetic_search()
ProgramSearchResult genetic_programming_search(const ProgramObjective& objective,
                                                const ProgramSearchOptions& options = {});

} // namespace datamunge::optim
