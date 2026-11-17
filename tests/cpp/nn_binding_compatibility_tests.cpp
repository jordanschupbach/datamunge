#include <gtest/gtest.h>

#include "common/memory_components.hpp"
namespace n = datamunge::nn;
TEST(NnBindingCompatibility, EveryMemoryKindRetainsLiveGradientsAndSavedArrays) {
  for (auto kind :
       {memory_components::Kind::Ntm,
        memory_components::Kind::Dnc,
        memory_components::Kind::MemNet,
        memory_components::Kind::KeyValue}) {
    bool copy = kind == memory_components::Kind::Ntm || kind == memory_components::Kind::Dnc;
    n::NamedTensors batch;
    for (const auto& [key, t] : memory_data::Make(copy))
      if (key.starts_with("fit_"))
        batch.emplace_back(key.substr(4), n::take_rows(t, {0, 1}));
    memory_components::Model model(kind, batch);
    auto                     leaves = model.Leaves(batch);
    auto                     out    = model.Forward(leaves, batch);
    auto                     loss   = model.Loss(out, batch);
    EXPECT_TRUE(std::isfinite(loss.item<double>()));
    auto grads = n::grad(loss, model.parameters().parameters());
    EXPECT_FALSE(grads.empty());
    double total = 0;
    for (const auto& g : grads)
      if (g.defined()) {
        EXPECT_TRUE(n::all(n::is_finite(g)).item<bool>());
        total += n::sum(n::abs(g)).item<double>();
      }
    EXPECT_GT(total, 0);
    auto prediction = model.Predict(batch);
    EXPECT_TRUE(n::all(n::is_finite(prediction)).item<bool>());
    EXPECT_FALSE(model.Arrays().empty());
  }
}
