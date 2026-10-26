// CMD: just run-nn dnc
// Differentiable neural computer binary copy with allocation and temporal
// linkage.
//
// Port of tflow's dnc (common/nn/dnc.h -> examples/nn/common/dnc.hpp, shared
// training flow in common/memory_task.hpp). Data: output/datasets/ntm/data.npz
// (the NTM copy task; tflow's scripts/fetch_datasets.py, found through
// paths::Dataset), copied into the run directory like tflow does. 799
// parameters in 7 tensors, Adam .005, batch 8 with BatchOrder(128, 8, 45207),
// three epochs (48 updates). Fixtures: allocation (untied usage + gradient,
// stable ties / full / empty usage), one full access step with probe-weighted
// partials of every input, and an exact temporal write chain; each is compared
// with tflow's fixture when present. Checks: parameter counts, finite
// validation loss, bit-exact .npz restore, label-free repeat prediction, one
// native DncCell step from every recorded state reproduces all nine state
// traces and the access fields (3e-5), input padding cannot change any state
// or gradient, weights immutable under inference.
// Deviations: checkpoints are final.npz (parameter store) instead of SaveV2;
// weights initialise from nn's generator (statistical parity) unless
// init=PATH (tflow's output/models/dnc/initial_arrays.npz) replays tflow's run,
// in which case every record is compared with tflow's run directory (tf=PATH).
// Options: epochs=3 out=PATH init=PATH tf=PATH. Output: output/nn/models/dnc/.
#include <algorithm>
#include <cmath>
#include <sstream>

#include "common/dnc.hpp"

// Experiment workflow from examples/nn/common/memory_task.hpp.
namespace memory_nets {
using namespace ::memory_nets;
template <typename F>
int Main(int argc, char** argv, const std::string& name, F&& body) {
  try {
    nn::set_num_threads(4);
    body(rnn_cells::ParseDemoOptions(argc, argv, name));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
}  // namespace memory_nets



// Experiment workflow from examples/nn/common/memory_task.hpp.
namespace memory_nets {
using namespace ::memory_nets;
inline TaskRun Train(const TaskSpec& spec, const rnn_cells::DemoOptions& options,
                     const std::function<void(const fs::path&)>& fixtures = {}) {
  TaskRun run;
  run.start   = std::chrono::steady_clock::now();
  run.options = options;
  run.root    = options.out;
  const fs::path& root = run.root;
  fs::create_directories(root);
  const fs::path source = paths::Dataset(spec.dataset);
  check::Require(fs::exists(source / "data.npz"), "dataset " + (source / "data.npz").string() +
                                                       " (tflow's scripts/fetch_datasets.py)");
  for (const char* f : {"data.npz", "manifest.json"})
    if (fs::exists(source / f)) fs::copy_file(source / f, root / f, fs::copy_options::overwrite_existing);
  if (fixtures) fixtures(root);
  const auto all = nn::load_npz((root / "data.npz").string());
  run.fit        = Split(all, "fit", spec.keys);
  run.validation = Split(all, "validation", spec.keys);
  run.test       = Split(all, "test", spec.keys);
  const Batch example = Rows(run.fit, {0});
  run.model           = spec.make(true, example);
  auto& m             = *run.model;
  check::Require(m.store.count() == spec.parameters && static_cast<std::int64_t>(m.store.size()) == spec.tensors,
                 spec.count_message);
  if (!options.init.empty()) m.LoadArrays(options.init);
  nn::save_npz((root / "initial_arrays.npz").string(), m.Arrays());
  SaveMap(root / "initial_outputs.npz", m.Evaluate(run.test));
  const auto fit_rows = static_cast<std::size_t>(run.fit.begin()->second.shape()[0]);
  nn::BatchOrder order(fit_rows, 8, spec.order_seed);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history << "epoch\t" << spec.history_column << "\n";
  batches << "epoch\tstep\trow\n";
  for (int e = 1; e <= options.epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const Batch b = Rows(run.fit, rows);
      if (!run.updates) {
        run.first = b;
        SaveMap(root / "first_batch.npz", b);
        nn::save_npz((root / "initial_gradients.npz").string(), m.Gradient(b));
        if (spec.first_forward) SaveMap(root / "first_forward.npz", m.Evaluate(b));
      }
      m.Update(b);
      if (!run.updates) nn::save_npz((root / "first_after.npz").string(), m.Arrays());
      for (auto row : rows) batches << e << '\t' << run.updates << '\t' << row << '\n';
      ++run.updates;
    }
    const float ce = m.Evaluate(run.validation).at("loss").item<float>();
    check::Require(std::isfinite(ce), "finite CE");
    history << e << '\t' << ce << '\n';
    std::cout << "epoch=" << e << " " << spec.print_name << "=" << ce << std::endl;
  }
  history.close();
  batches.close();
  nn::save_npz((root / "final_gradients.npz").string(), m.Gradient(run.first));
  if (spec.first_forward) SaveMap(root / "final_first_forward.npz", m.Evaluate(run.first));
  m.Save(root / "final.npz");
  run.deployed = spec.make(false, example);
  run.deployed->Restore(root / "final.npz");
  run.weights = run.deployed->Arrays();
  check::Require(SameBits(m.Arrays(), run.weights), "restore bitexact");
  run.out        = run.deployed->Evaluate(run.test);
  run.prediction = run.deployed->Predict(run.test);
  check::Require(SameBits(run.prediction, run.out.at("logits")) && SameBits(run.prediction, run.deployed->Predict(run.test)),
                 "label-free repeat inference");
  return run;
}
}  // namespace memory_nets

// Experiment workflow from examples/nn/common/memory_task.hpp.
namespace memory_nets {
using namespace ::memory_nets;
inline void Finish(const TaskSpec& spec, TaskRun& run, const std::string& metrics_json, const std::string& summary,
                   const std::string& html_body) {
  const fs::path& root = run.root;
  SaveMap(root / "outputs.npz", run.out);
  nn::save_npz((root / "model_arrays.npz").string(), run.weights);
  check::Require(SameBits(run.deployed->Arrays(), run.weights), "immutable inference");
  {
    std::ofstream metrics(root / "metrics.json");
    metrics << metrics_json << "\n";
    std::ofstream html(root / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>" << spec.name
         << "</title><style>body{font:16px system-ui;margin:30px;max-width:1000px}</style><h1>" << spec.name << "</h1>"
         << html_body << "<p>" << summary << "</p><p>See metrics.json, history.tsv and the .npz records.</p>";
    check::Require(bool(metrics) && bool(html), "complete records");
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - run.start).count();
  std::cout << summary << " (" << seconds << " s)\nArtifacts:" << (root / "index.html") << std::endl;
  std::vector<std::string> records{"first_batch.npz", "initial_arrays.npz", "initial_outputs.npz", "initial_gradients.npz",
                                   "first_after.npz", "final_gradients.npz", "model_arrays.npz", "outputs.npz"};
  if (spec.first_forward) {
    records.push_back("first_forward.npz");
    records.push_back("final_first_forward.npz");
  }
  rnn_cells::CompareWithTflow(root, run.options, records, spec.test_key);
}
}  // namespace memory_nets



using namespace memory_nets;

namespace {

Tensor Vec(const std::vector<float>& v, const nn::Shape& shape) { return nn::from_vector(v, shape); }

void Fixtures(const fs::path& root) {
  {  // allocation
    std::vector<float> probe(12);
    for (int j = 0; j < 12; ++j) probe[j] = float(.3 * std::sin(j * .7));
    Tensor usage = Vec({.2f, .7f, .1f, .5f, .8f, .3f, .6f, .1f, .01f, .2f, .4f, .9f}, {3, 4});
    const Tensor ties = Vec({0, 0, 0, 0, .2f, .2f, .8f, .4f, 1, 1, 1, 1}, {3, 4}), p = Vec(probe, {3, 4});
    usage.requires_grad_(true);
    const Tensor allocation = DncAllocation(usage);
    const auto   g          = nn::grad(std::vector<Tensor>{nn::sum(nn::mul(allocation, p))}, {usage});
    Tensor       tie;
    {
      nn::NoGradGuard guard;
      tie = DncAllocation(ties);
    }
    nn::save_npz((root / "allocation_fixture.npz").string(), {{"usage", usage.detach()},
                                                               {"probe", p},
                                                               {"allocation", allocation.detach()},
                                                               {"gradient", g[0]},
                                                               {"ties", ties},
                                                               {"tie_allocation", tie}});
  }
  {  // one access step with every partial
    std::map<std::string, Tensor> inputs;
    for (auto [name, width] : std::vector<std::pair<std::string, int>>{{"memory", 16}, {"read_weights", 4}, {"write_weights", 4},
                                                                       {"usage", 4},   {"precedence", 4},   {"links", 16},
                                                                       {"interface", 24}}) {
      std::vector<float> v(2 * width);
      for (int j = 0; j < 2 * width; ++j)
        v[j] = name == "memory" || name == "interface" ? float(.3 * std::sin(j * .43) + .1) : float(.05 + .03 * (j % 5));
      inputs[name] = width == 16 ? Vec(v, {2, 4, 4}) : Vec(v, {2, width});
    }
    inputs["usage"] = Vec({.1f, .4f, .7f, .9f, .2f, .8f, .35f, .6f}, {2, 4});
    {
      auto links = inputs["links"].to_vector<float>();
      for (int b = 0; b < 2; ++b)
        for (int j = 0; j < 4; ++j) links[b * 16 + j * 5] = 0;
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
    DncState previous{zero, zero, inputs["memory"], nn::zeros({2, 4}), inputs["read_weights"], inputs["write_weights"],
                      inputs["usage"], inputs["precedence"], inputs["links"]};
    const auto access = DncMemory(inputs["interface"], previous);
    const auto values = access.state.Values();
    std::map<std::string, Tensor> fields{{"allocation", access.allocation}, {"forward", access.forward}, {"backward", access.backward}};
    for (int j = 2; j < 9; ++j) fields[DncNames()[j]] = values[j];
    Tensor           objective = nn::sum(nn::mul(inputs["write_weights"], 0.0));
    nn::NamedTensors saved;
    int              field = 0;
    for (const auto& [name, value] : fields) {
      const bool         matrix = name == "memory" || name == "links";
      std::vector<float> probe(matrix ? 32 : 8);
      for (std::size_t j = 0; j < probe.size(); ++j) probe[j] = float(.2 * std::sin(j * .37 + field * .61));
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
      saved.emplace_back("gradient_" + leafnames[j], g[j].defined() ? g[j] : nn::zeros_like(leaves[j].detach()));
    for (const auto& [name, value] : inputs) saved.emplace_back("input_" + name, value.detach());
    nn::save_npz((root / "access_fixture.npz").string(), saved);
  }
  {  // exact temporal write chain into slots 2, 0, 3
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
    nn::save_npz((root / "temporal_fixture.npz").string(), {{"links", nn::stack(trace)}, {"precedence", p}});
    const auto l = links.to_vector<float>();
    // After writes 2 -> 0 -> 3: L[0][2] = L[3][0] = 1 and nothing else.
    for (int i = 0; i < 16; ++i) check::Near(l[i], i == 2 || i == 12 ? 1 : 0, 0, "exact temporal write chain");
  }
}

}  // namespace

int main(int argc, char** argv) {
  return Main(argc, argv, "dnc", [](const rnn_cells::DemoOptions& options) {
    TaskSpec spec;
    spec.name           = "dnc";
    spec.dataset        = "ntm";
    spec.keys           = {"input", "target", "lengths", "source_ids"};
    spec.history_column = "validation_bit_ce";
    spec.print_name     = "validationBitCE";
    spec.test_key       = "test_bit_ce";
    spec.order_seed     = 45207;
    spec.parameters     = 799;
    spec.tensors        = 7;
    spec.count_message  = "799 parameters seven tensors";
    spec.first_forward  = true;
    spec.make           = DncModel;
    auto run            = Train(spec, options, Fixtures);
    const auto& test    = run.test;
    const auto& out     = run.out;

    const auto lengths = test.at("lengths").to_vector<std::int32_t>();
    for (int t = 0; t < 9; ++t) {
      std::vector<Tensor> values;
      for (const auto& name : DncNames()) values.push_back(nn::select(out.at(name), 1, t));
      DncStep step;
      {
        nn::NoGradGuard guard;
        step = DncCell(run.deployed->store, nn::select(test.at("input"), 1, t), DncState::From(values));
      }
      std::map<std::string, Tensor> fields{{"logits", step.logits},
                                           {"interface", step.interface},
                                           {"allocation", step.allocation},
                                           {"forward", step.forward},
                                           {"backward", step.backward}};
      const auto state = step.state.Values();
      for (std::size_t j = 0; j < state.size(); ++j) fields[DncNames()[j]] = state[j];
      for (const auto& [name, value] : fields) {
        const bool        matrix = name == "memory" || name == "links";
        const int         next   = fields.count(name) && std::find(DncNames().begin(), DncNames().end(), name) != DncNames().end();
        const auto        a = value.to_vector<float>(), e = nn::select(out.at(name), 1, t + next).to_vector<float>();
        const std::size_t width = a.size() / 32;
        for (int b = 0; b < 32; ++b)
          if (t <= 2 * lengths[b])
            for (std::size_t j = 0; j < width; ++j)
              check::Near(a[b * width + j], e[b * width + j], 3e-5,
                          matrix ? "DNC streaming matrix parity" : "DNC streaming state/access parity");
      }
    }
    const Batch pad    = PadCopyInputs(test, .93f);
    const auto  normal = run.deployed->Evaluate(Rows(test, {0, 1})), padded = run.deployed->Evaluate(pad);
    check::Require(SameMaps(normal, padded), "all DNC state/gradient padding invariance");
    SaveMap(run.root / "padding_inputs.npz", pad);
    SaveMap(run.root / "padding_outputs.npz", padded);

    const auto         score = ScoreCopy(run.prediction, test);
    const float        ce    = out.at("loss").item<float>();
    std::ostringstream metrics, summary;
    metrics << "{\"parameters\":799,\"epochs\":" << options.epochs << ",\"updates\":" << run.updates << ",\"test_bit_ce\":" << ce
            << ",\"bit_accuracy\":" << double(score.correct) / score.total << ",\"exact\":" << score.exact << "}";
    summary << "testBitCE=" << ce << " bitAccuracy=" << double(score.correct) / score.total << " exact=" << score.exact
            << "/32;restore/nativeAllStateStreaming/padding/labelFree/immutable PASS";
    Finish(spec, run, metrics.str(), summary.str(),
           "<p>Binary copy (the NTM task). A differentiable neural computer (LSTM(8) controller, 4x4 memory, dynamic "
           "allocation from usage with free gates, temporal link matrix, read modes backward / forward / content) is "
           "trained with masked bit cross-entropy on the recall steps for three epochs.</p>");
    for (const char* f : {"allocation_fixture.npz", "access_fixture.npz", "temporal_fixture.npz"})
      if (fs::exists(options.tf / f)) {
        const double d = rnn_cells::CompareNpz(run.root / f, options.tf / f);
        std::cout << "  " << f << " max|diff| vs tflow=" << d << "\n";
        check::Require(d <= 1e-5, std::string(f) + " matches tflow");
      }
  });
}
