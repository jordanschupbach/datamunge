#include <datamunge/optim/linkage_search.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace o = datamunge::optim;

TEST(LinkageSearch, ScoresDistinctMixesAndProvidesDiversificationWhenUniform) {
  for (const auto method : {o::LinkageMixingMethod::parent_pair, o::LinkageMixingMethod::gene_pool}) {
    std::vector<o::CategoricalPoint> history;
    const auto objective = [&](const o::CategoricalPoint& point) {
      history.push_back(point);
      EXPECT_TRUE(std::ranges::all_of(point, [](auto bit) { return bit <= 1; }));
      return double(point[0] != 1 || point[1] != 1);
    };
    const o::LinkageSearchOptions options{81, 7, 8, method};
    const auto result = o::linkage_search(objective, 2, options);
    EXPECT_EQ(history.size(), 81);
    EXPECT_EQ(result.evaluations, 81);
    EXPECT_EQ(result.value, 0);
    EXPECT_EQ(result.point, (o::CategoricalPoint{1,1}));
    const auto prior = history;
    history.clear();
    const auto replay = o::linkage_search(objective, 2, options);
    EXPECT_EQ(replay.point, result.point);
    EXPECT_EQ(prior, history);
  }
}
