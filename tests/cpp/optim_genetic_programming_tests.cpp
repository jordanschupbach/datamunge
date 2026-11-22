#include <datamunge/optim/genetic_programming.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace o = datamunge::optim;

TEST(GeneticProgram, ProtectsArithmeticAndRejectsInvalidGraph) {
  const std::array input{8.0, 0.0};
  const o::GeneticProgram zero_divide(2, {{o::ProgramOperation::divide, 0, 1}}, 2);
  EXPECT_EQ(zero_divide.evaluate(input), 8);
  EXPECT_THROW((void)zero_divide.evaluate(std::array{1.0}), std::invalid_argument);
  EXPECT_THROW(o::GeneticProgram(1, {{o::ProgramOperation::add, 0, 1}}, 1), std::invalid_argument);
  const o::GeneticProgram square(1, {{o::ProgramOperation::multiply, 0, 0}}, 1);
  const auto maximum = std::numeric_limits<double>::max();
  EXPECT_EQ(square.evaluate(std::array{maximum}), maximum);
}

TEST(GeneticProgramming, AllRepresentationsExpressValidProgramsAndReplayAcrossPartialGenerations) {
  for (const auto method : {o::ProgramSearchMethod::tree, o::ProgramSearchMethod::cartesian,
                            o::ProgramSearchMethod::gene_expression}) {
    for (const auto budget : {1u, 5u, 97u}) {
      std::vector<double> history;
      const auto loss = [](const o::GeneticProgram& program) {
        double error = 0;
        for (const double point : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
          const auto remainder = std::clamp(program.evaluate(std::array{point}), -1e50, 1e50) - point * point;
          error += remainder * remainder;
        }
        return error;
      };
      const auto objective = [&](const o::GeneticProgram& program) {
        const auto value = loss(program);
        history.push_back(value);
        return value;
      };
      const o::ProgramSearchOptions options{budget, 7, method, 1, 5, 8, 0.3, 0.9};
      const auto result = o::genetic_programming_search(objective, options);
      ASSERT_EQ(history.size(), budget);
      EXPECT_EQ(result.evaluations, budget);
      EXPECT_EQ(result.value, *std::ranges::min_element(history));
      EXPECT_EQ(loss(result.program), result.value);
      const auto prior = history;
      history.clear();
      const auto again = o::genetic_programming_search(objective, options);
      EXPECT_EQ(again.value, result.value);
      EXPECT_EQ(prior, history);
    }
  }
}
