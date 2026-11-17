// Canonical numerical fixtures shared by C++, R and Python. No fitting loop.
#pragma once
#include <cmath>

#include "common/dnc.hpp"
#include "common/ntm.hpp"
namespace memory_validation {
using namespace memory_nets;
inline void NtmFixtures(const std::string& directory) {
  const fs::path     root(directory);
  std::vector<float> memory(32), parameters(20);
  for (int i = 0; i < 32; ++i)
    memory[i] = float(.3 * std::sin(i * .7) + .05);
  for (int i = 0; i < 20; ++i)
    parameters[i] = float(.4 * std::cos(i * .3));
  Tensor m = nn::from_vector(memory, {2, 4, 4}), p = nn::from_vector(parameters, {2, 10});
  Tensor v = nn::from_vector(std::vector<float>{.7f, .1f, .15f, .05f, .1f, .2f, .3f, .4f}, {2, 4});
  Tensor probe =
      nn::from_vector(std::vector<float>{.2f, -.8f, .5f, .9f, .8f, -.3f, .7f, -.4f}, {2, 4});
  m.requires_grad_(true), p.requires_grad_(true), v.requires_grad_(true);
  const auto address = NtmAddressing(m, p, v);
  const auto g = nn::grad(std::vector<Tensor>{nn::sum(nn::mul(address.weights, probe))}, {m, p, v});
  const fs::path file = root / "address_fixture.npz";
  nn::save_npz(
      file.string(),
      {{"memory", m.detach()},
       {"parameters", p.detach()},
       {"previous", v.detach()},
       {"probe", probe},
       {"content", address.content.detach()},
       {"interpolated", address.interpolated.detach()},
       {"shifted", address.shifted.detach()},
       {"weights", address.weights.detach()},
       {"memory_gradient", g[0]},
       {"parameters_gradient", g[1]},
       {"previous_gradient", g[2]}});
  double total = 0;
  for (float w : address.weights.to_vector<float>())
    total += w;
  check::Near(total, 2, 1e-5, "address weights are distributions");
}

inline Tensor Vec(const std::vector<float>& v, const nn::Shape& shape) {
  return nn::from_vector(v, shape);
}
inline void DncFixtures(const std::string& directory) {
  const fs::path root(directory);
  { // allocation
    std::vector<float> probe(12);
    for (int j = 0; j < 12; ++j)
      probe[j] = float(.3 * std::sin(j * .7));
    Tensor       usage = Vec({.2f, .7f, .1f, .5f, .8f, .3f, .6f, .1f, .01f, .2f, .4f, .9f}, {3, 4});
    const Tensor ties  = Vec({0, 0, 0, 0, .2f, .2f, .8f, .4f, 1, 1, 1, 1}, {3, 4}),
                 p     = Vec(probe, {3, 4});
    usage.requires_grad_(true);
    const Tensor allocation = DncAllocation(usage);
    const auto   g = nn::grad(std::vector<Tensor>{nn::sum(nn::mul(allocation, p))}, {usage});
    Tensor       tie;
    {
      nn::NoGradGuard guard;
      tie = DncAllocation(ties);
    }
    nn::save_npz(
        (root / "allocation_fixture.npz").string(),
        {{"usage", usage.detach()},
         {"probe", p},
         {"allocation", allocation.detach()},
         {"gradient", g[0]},
         {"ties", ties},
         {"tie_allocation", tie}});
  }
  { // one access step with every partial
    std::map<std::string, Tensor> inputs;
    for (auto [name, width] : std::vector<std::pair<std::string, int>>{
             {"memory", 16},
             {"read_weights", 4},
             {"write_weights", 4},
             {"usage", 4},
             {"precedence", 4},
             {"links", 16},
             {"interface", 24}}) {
      std::vector<float> v(2 * width);
      for (int j = 0; j < 2 * width; ++j)
        v[j] = name == "memory" || name == "interface" ? float(.3 * std::sin(j * .43) + .1)
                                                       : float(.05 + .03 * (j % 5));
      inputs[name] = width == 16 ? Vec(v, {2, 4, 4}) : Vec(v, {2, width});
    }
    inputs["usage"] = Vec({.1f, .4f, .7f, .9f, .2f, .8f, .35f, .6f}, {2, 4});
    {
      auto links = inputs["links"].to_vector<float>();
      for (int b = 0; b < 2; ++b)
        for (int j = 0; j < 4; ++j)
          links[b * 16 + j * 5] = 0;
      inputs["links"] = Vec(links, {2, 4, 4});
    }
    std::vector<Tensor>      leaves;
    std::vector<std::string> leafnames;
    for (auto& [name, value] : inputs) {
      value.requires_grad_(true);
      leaves.push_back(value);
      leafnames.push_back(name);
    }
    const Tensor zero = nn::zeros({2, 8});
    DncState     previous{
        zero,
        zero,
        inputs["memory"],
        nn::zeros({2, 4}),
        inputs["read_weights"],
        inputs["write_weights"],
        inputs["usage"],
        inputs["precedence"],
        inputs["links"]};
    const auto                    access = DncMemory(inputs["interface"], previous);
    const auto                    values = access.state.Values();
    std::map<std::string, Tensor> fields{
        {"allocation", access.allocation},
        {"forward", access.forward},
        {"backward", access.backward}};
    for (int j = 2; j < 9; ++j)
      fields[DncNames()[j]] = values[j];
    Tensor           objective = nn::sum(nn::mul(inputs["write_weights"], 0.0));
    nn::NamedTensors saved;
    int              field = 0;
    for (const auto& [name, value] : fields) {
      const bool         matrix = name == "memory" || name == "links";
      std::vector<float> probe(matrix ? 32 : 8);
      for (std::size_t j = 0; j < probe.size(); ++j)
        probe[j] = float(.2 * std::sin(j * .37 + field * .61));
      const Tensor p = matrix ? Vec(probe, {2, 4, 4}) : Vec(probe, {2, 4});
      saved.emplace_back("probe_" + name, p);
      objective = nn::add(objective, nn::sum(nn::mul(value, p)));
      saved.emplace_back(name, value.detach());
      ++field;
    }
    nn::GradOptions options;
    options.allow_unused = true;
    const auto g         = nn::grad(std::vector<Tensor>{objective}, leaves, {}, options);
    saved.emplace_back("objective", objective.detach());
    for (std::size_t j = 0; j < g.size(); ++j)
      saved.emplace_back(
          "gradient_" + leafnames[j], g[j].defined() ? g[j] : nn::zeros_like(leaves[j].detach()));
    for (const auto& [name, value] : inputs)
      saved.emplace_back("input_" + name, value.detach());
    nn::save_npz((root / "access_fixture.npz").string(), saved);
  }
  { // exact temporal write chain into slots 2, 0, 3
    Tensor              p = nn::zeros({1, 4}), links = nn::zeros({1, 4, 4});
    std::vector<Tensor> trace{links};
    for (int slot : {2, 0, 3}) {
      std::vector<float> w(4, 0.f);
      w[slot]         = 1;
      const auto next = DncTemporal(links, p, Vec(w, {1, 4}));
      links           = next.links;
      p               = next.precedence;
      trace.push_back(links);
    }
    nn::save_npz(
        (root / "temporal_fixture.npz").string(), {{"links", nn::stack(trace)}, {"precedence", p}});
    const auto l = links.to_vector<float>();
    // After writes 2 -> 0 -> 3: L[0][2] = L[3][0] = 1 and nothing else.
    for (int i = 0; i < 16; ++i)
      check::Near(l[i], i == 2 || i == 12 ? 1 : 0, 0, "exact temporal write chain");
  }
}
} // namespace memory_validation
