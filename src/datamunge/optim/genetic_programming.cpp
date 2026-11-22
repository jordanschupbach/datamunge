/// @file
/// @brief Program expression, representation-specific variation and budgeted genetic search.

#include <datamunge/optim/genetic_programming.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::optim {
/// @brief Arithmetic, prefix-tree and linear-genome internals.
namespace {

/// @brief Number of operands consumed by one instruction.
/// @param operation Defined instruction symbol.
/// @return Arity from zero through two.
/// @throws std::invalid_argument Unknown instruction symbol.
std::size_t arity(ProgramOperation operation) {
  switch (operation) {
    case ProgramOperation::constant: return 0;
    case ProgramOperation::sine:
    case ProgramOperation::cosine: return 1;
    case ProgramOperation::add:
    case ProgramOperation::subtract:
    case ProgramOperation::multiply:
    case ProgramOperation::divide: return 2;
  }
  throw std::invalid_argument("Unknown program operation");
}

/// @brief Execute one protected operator with prevalidated earlier operands.
/// @param instruction Executable graph node.
/// @param values Input and previous instruction output slots.
/// @return Finite result after saturation.
double execute(const ProgramInstruction& instruction, const std::vector<double>& values) {
  constexpr long double max_value = std::numeric_limits<double>::max();
  if (instruction.operation == ProgramOperation::constant) return instruction.literal;
  const long double a = values[instruction.first];
  const long double b = values[instruction.second];
  long double result = 0;
  switch (instruction.operation) {
    case ProgramOperation::add: result = a + b; break;
    case ProgramOperation::subtract: result = a - b; break;
    case ProgramOperation::multiply: result = a * b; break;
    case ProgramOperation::divide: result = std::abs(b) < 1e-12L ? a : a / b; break;
    case ProgramOperation::sine: result = std::sin(a); break;
    case ProgramOperation::cosine: result = std::cos(a); break;
    case ProgramOperation::constant: break;
  }
  return static_cast<double>(std::clamp(result, -max_value, max_value));
}

/// @brief One function symbol or input terminal in a tree or head-tail gene.
struct Symbol {
  ProgramOperation op{}; ///< Evaluator operation.
  bool variable{}; ///< True for a direct input terminal.
  std::size_t input{}; ///< Input index when variable.
  double literal{}; ///< Constant value when nonvariable and op is constant.
};

/// @brief Evaluated program genotype in one of the selected representations.
struct Genome {
  std::vector<Symbol> symbols; ///< Prefix tree or GEP head and tail.
  std::vector<ProgramInstruction> graph; ///< Cartesian graph, when used.
  std::size_t output{}; ///< Cartesian output slot.
  double loss{}; ///< Observed fitness.
};

/// @brief Unified variation controls and local random engine.
struct Variation {
  const ProgramSearchOptions& options; ///< Validated optimizer settings.
  std::mt19937_64 rng; ///< Local random engine.
  std::uniform_real_distribution<double> uniform; ///< Draw in [0,1).
  std::normal_distribution<double> normal; ///< Gaussian constants.

  /// @brief Draw a slot index from a nonempty prefix.
  /// @param count Positive number of available slots.
  /// @return Index from zero through count minus one.
  std::size_t pick(std::size_t count) {
    return std::uniform_int_distribution<std::size_t>(0, count - 1)(rng);
  }

  /// @brief Generate a function or terminal at a legal genome position.
  /// @param terminal Whether that position allows only terminals.
  /// @return Random function or terminal.
  Symbol symbol(bool terminal) {
    if (!terminal && uniform(rng) < 0.55) return {static_cast<ProgramOperation>(1 + pick(6))};
    if (uniform(rng) < 0.6) return {ProgramOperation::constant, true, pick(options.input_count), 0};
    return {ProgramOperation::constant, false, 0, normal(rng)};
  }

  /// @brief Generate a prefix-encoded tree with a height bound.
  /// @param symbols Output array, appended in prefix order.
  /// @param depth Maximum height in nodes, positive.
  void tree(std::vector<Symbol>& symbols, std::size_t depth) {
    const auto root = symbol(depth == 1);
    symbols.push_back(root);
    for (std::size_t child = 0; child < arity(root.op); ++child) tree(symbols, depth - 1);
  }

  /// @brief Generate one valid Cartesian graph instruction.
  /// @param earlier Number of available input and earlier node slots.
  /// @return Random operation and earlier operands.
  ProgramInstruction node(std::size_t earlier) {
    return {static_cast<ProgramOperation>(pick(7)), pick(earlier), pick(earlier), normal(rng)};
  }
};

/// @brief Compile one prefix subtree into postorder instructions.
/// @param tokens Valid prefix program.
/// @param index Next token, advanced through its subtree.
/// @param inputs Number of input slots.
/// @param instructions Compiled instructions, appended in postorder.
/// @return Slot holding the subtree's result.
std::size_t compile_tree(const std::vector<Symbol>& tokens, std::size_t& index, std::size_t inputs,
                         std::vector<ProgramInstruction>& instructions) {
  const auto& current = tokens[index++];
  if (current.variable) return current.input;
  const auto children = arity(current.op);
  const auto first = children > 0 ? compile_tree(tokens, index, inputs, instructions) : 0;
  const auto second = children == 2 ? compile_tree(tokens, index, inputs, instructions) : 0;
  const auto output = inputs + instructions.size();
  instructions.push_back({current.op, first, second, current.literal});
  return output;
}

/// @brief Decode a valid GEP gene from breadth-first symbols into a graph.
/// @param symbols Valid head-tail gene.
/// @param inputs Number of input terminals.
/// @return Compiled protected expression.
GeneticProgram compile_gene(const std::vector<Symbol>& symbols, std::size_t inputs) {
  std::vector<std::size_t> first(symbols.size());
  std::size_t end = 1;
  for (std::size_t i = 0; i < end; ++i) {
    first[i] = end;
    end += arity(symbols[i].op);
  }
  std::vector<std::size_t> slots(end);
  std::vector<ProgramInstruction> instructions;
  for (std::size_t i = end; i-- > 0;) {
    const auto& current = symbols[i];
    if (current.variable) {
      slots[i] = current.input;
      continue;
    }
    const auto count = arity(current.op);
    slots[i] = inputs + instructions.size();
    instructions.push_back({current.op, count > 0 ? slots[first[i]] : 0,
                             count == 2 ? slots[first[i] + 1] : 0, current.literal});
  }
  return GeneticProgram(inputs, std::move(instructions), slots.front());
}

/// @brief Compile any supported genotype to the common evaluator.
/// @param genome Valid encoded program.
/// @param options Representation and input shape.
/// @return Owned executable expression.
GeneticProgram compile(const Genome& genome, const ProgramSearchOptions& options) {
  if (options.method == ProgramSearchMethod::cartesian) return {options.input_count, genome.graph, genome.output};
  if (options.method == ProgramSearchMethod::gene_expression) return compile_gene(genome.symbols, options.input_count);
  std::vector<ProgramInstruction> instructions;
  std::size_t index = 0;
  const auto output = compile_tree(genome.symbols, index, options.input_count, instructions);
  return {options.input_count, std::move(instructions), output};
}

/// @brief Locate the end and height of one prefix subtree.
/// @param symbols Valid prefix tree.
/// @param start Offset of the subtree root.
/// @return Half-open end index and node-count height.
std::pair<std::size_t, std::size_t> tree_geometry(const std::vector<Symbol>& symbols, std::size_t start) {
  auto end = start + 1;
  std::size_t height = 0;
  for (std::size_t child = 0; child < arity(symbols[start].op); ++child) {
    const auto [next, child_height] = tree_geometry(symbols, end);
    end = next;
    height = std::max(height, child_height);
  }
  return {end, height + 1};
}

/// @brief Try one subtree replacement and enforce the complete tree height.
/// @param variation Local engine and height control.
/// @param original Existing valid tree.
/// @param donor Valid replacement subtree, with no unused symbols.
/// @return Resulting tree, with original retained if the height would exceed the limit.
std::vector<Symbol> replace_subtree(Variation& variation, const std::vector<Symbol>& original,
                                    const std::vector<Symbol>& donor) {
  const auto cut = variation.pick(original.size());
  const auto [end, ignored] = tree_geometry(original, cut);
  std::vector<Symbol> changed(original.begin(), original.begin() + cut);
  changed.insert(changed.end(), donor.begin(), donor.end());
  changed.insert(changed.end(), original.begin() + end, original.end());
  return tree_geometry(changed, 0).second <= variation.options.size ? std::move(changed) : original;
}

/// @brief Draw a genotype, using an explicit zero constant for the first trial.
/// @param variation Local engine and representation options.
/// @param zero Whether to express the zero constant.
/// @return Initialized genotype.
Genome initialize(Variation& variation, bool zero) {
  const auto& options = variation.options;
  Genome out;
  if (options.method == ProgramSearchMethod::cartesian) {
    for (std::size_t i = 0; i < options.size; ++i) out.graph.push_back(variation.node(options.input_count + i));
    if (zero) out.graph[0] = {};
    out.output = zero ? options.input_count : variation.pick(options.input_count + options.size);
  } else if (options.method == ProgramSearchMethod::gene_expression) {
    for (std::size_t i = 0; i < 2 * options.size + 1; ++i) out.symbols.push_back(variation.symbol(i >= options.size));
    if (zero) out.symbols.front() = {};
  } else if (zero) {
    out.symbols.push_back({});
  } else {
    variation.tree(out.symbols, 1 + variation.pick(options.size));
  }
  return out;
}

/// @brief Generate a new genotype with representation-specific crossover and mutation.
/// @param variation Local random engine and options.
/// @param parent Greedy or tournament-selected template.
/// @param donor Other parent for crossover when used.
/// @return Feasible unscored genotype.
Genome vary(Variation& variation, const Genome& parent, const Genome& donor) {
  const auto& options = variation.options;
  auto out = parent;
  if (options.method == ProgramSearchMethod::cartesian) {
    for (std::size_t i = 0; i < out.graph.size(); ++i) {
      auto& node = out.graph[i];
      const auto boundary = options.input_count + i;
      if (variation.uniform(variation.rng) < options.mutation) {
        node.operation = static_cast<ProgramOperation>(variation.pick(7));
        node.literal = variation.normal(variation.rng);
      }
      if (variation.uniform(variation.rng) < options.mutation) node.first = variation.pick(boundary);
      if (variation.uniform(variation.rng) < options.mutation) node.second = variation.pick(boundary);
    }
    if (variation.uniform(variation.rng) < options.mutation) out.output = variation.pick(options.input_count + options.size);
  } else if (options.method == ProgramSearchMethod::gene_expression) {
    if (variation.uniform(variation.rng) < options.crossover) {
      const auto cut = 1 + variation.pick(out.symbols.size() - 1);
      std::copy(donor.symbols.begin() + cut, donor.symbols.end(), out.symbols.begin() + cut);
    }
    for (std::size_t i = 0; i < out.symbols.size(); ++i) {
      if (variation.uniform(variation.rng) < options.mutation) out.symbols[i] = variation.symbol(i >= options.size);
    }
    if (options.size > 1 && variation.uniform(variation.rng) < options.mutation) {
      const auto from = variation.pick(options.size);
      const auto into = 1 + variation.pick(options.size - 1);
      const auto count = std::min({std::size_t{3}, options.size - from, options.size - into});
      auto segment = std::vector<Symbol>(out.symbols.begin() + from, out.symbols.begin() + from + count);
      std::move_backward(out.symbols.begin() + into, out.symbols.begin() + options.size - count,
                         out.symbols.begin() + options.size);
      std::copy(segment.begin(), segment.end(), out.symbols.begin() + into);
    }
  } else {
    if (variation.uniform(variation.rng) < options.crossover) {
      const auto cut = variation.pick(donor.symbols.size());
      const auto end = tree_geometry(donor.symbols, cut).first;
      out.symbols = replace_subtree(variation, out.symbols,
                                   {donor.symbols.begin() + cut, donor.symbols.begin() + end});
    }
    if (variation.uniform(variation.rng) < options.mutation) {
      std::vector<Symbol> replacement;
      variation.tree(replacement, 1 + variation.pick(options.size));
      out.symbols = replace_subtree(variation, out.symbols, replacement);
    }
  }
  return out;
}

} // namespace

GeneticProgram::GeneticProgram(std::size_t inputs, std::vector<ProgramInstruction> instructions,
                               std::size_t output)
    : inputs_(inputs), instructions_(std::move(instructions)), output_(output) {
  if (inputs_ == 0 || instructions_.size() > std::numeric_limits<std::size_t>::max() - inputs_ ||
      output_ >= inputs_ + instructions_.size()) {
    throw std::invalid_argument("Invalid genetic program input or output size");
  }
  for (std::size_t i = 0; i < instructions_.size(); ++i) {
    const auto count = arity(instructions_[i].operation);
    auto& instruction = instructions_[i];
    if ((count == 0 && !std::isfinite(instruction.literal)) ||
        (count > 0 && instruction.first >= inputs_ + i) ||
        (count > 1 && instruction.second >= inputs_ + i)) {
      throw std::invalid_argument("Invalid genetic program instruction");
    }
    if (count < 2) instruction.second = 0;
    if (count == 0) instruction.first = 0;
  }
}

double GeneticProgram::evaluate(std::span<const double> input) const {
  if (input.size() != inputs_ || !std::ranges::all_of(input, [](double x) { return std::isfinite(x); })) {
    throw std::invalid_argument("Invalid genetic program input values");
  }
  std::vector<double> values(input.begin(), input.end());
  values.reserve(inputs_ + instructions_.size());
  for (const auto& instruction : instructions_) values.push_back(execute(instruction, values));
  return values.at(output_);
}

ProgramSearchResult genetic_programming_search(const ProgramObjective& objective, const ProgramSearchOptions& options) {
  const bool cartesian = options.method == ProgramSearchMethod::cartesian;
  const bool tree = options.method == ProgramSearchMethod::tree;
  if (!objective || options.evaluations == 0 || options.input_count == 0 || options.population < 2 ||
      options.size == 0 || options.size > (std::numeric_limits<std::size_t>::max() - 1) / 2 ||
      (tree && options.size > 12) || (!tree && !cartesian && options.method != ProgramSearchMethod::gene_expression) ||
      options.input_count > std::numeric_limits<std::size_t>::max() - 2 * options.size - 1 ||
      !(options.mutation >= 0 && options.mutation <= 1 && options.crossover >= 0 && options.crossover <= 1)) {
    throw std::invalid_argument("Invalid genetic programming options or objective");
  }
  Variation variation{options, std::mt19937_64(options.seed), {}, {}};
  ProgramSearchResult result{GeneticProgram(options.input_count, {}, 0),
                             std::numeric_limits<double>::infinity(), 0};
  /// Evaluate one actual expressed program and keep the earliest incumbent.
  const auto evaluate = [&](Genome genome) {
    auto program = compile(genome, options);
    genome.loss = objective(program);
    if (!std::isfinite(genome.loss)) throw std::domain_error("Genetic program objective must be finite");
    ++result.evaluations;
    if (genome.loss < result.value) {
      result.program = std::move(program);
      result.value = genome.loss;
    }
    return genome;
  };
  if (cartesian) {
    auto parent = evaluate(initialize(variation, true));
    while (result.evaluations < options.evaluations) {
      auto retained = parent;
      for (std::size_t i = 0; i < options.population && result.evaluations < options.evaluations; ++i) {
        auto candidate = evaluate(vary(variation, parent, parent));
        if (candidate.loss <= retained.loss) retained = std::move(candidate);
      }
      parent = std::move(retained);
    }
    return result;
  }

  std::vector<Genome> population;
  const auto count = std::min(options.evaluations, options.population);
  for (std::size_t i = 0; i < count; ++i) population.push_back(evaluate(initialize(variation, i == 0)));
  /// Select a minimum-loss parent from a three-way sampled tournament.
  const auto choose = [&]() -> const Genome& {
    auto winner = variation.pick(population.size());
    for (int i = 0; i < 2; ++i) {
      const auto candidate = variation.pick(population.size());
      if (population[candidate].loss < population[winner].loss) winner = candidate;
    }
    return population[winner];
  };
  while (result.evaluations < options.evaluations) {
    std::vector<Genome> next{*std::ranges::min_element(population, {}, &Genome::loss)};
    while (next.size() < count && result.evaluations < options.evaluations) {
      const auto& parent = choose();
      const auto& donor = choose();
      next.push_back(evaluate(vary(variation, parent, donor)));
    }
    population = std::move(next);
  }
  return result;
}

} // namespace datamunge::optim
