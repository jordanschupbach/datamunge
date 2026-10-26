// CMD: just run-nn ntm
// Neural Turing Machine binary copy with learned content/location heads.
//
// Port of tflow's ntm (common/nn/ntm.h -> examples/nn/common/ntm.hpp, shared
// training flow in common/memory_task.hpp). Data: output/datasets/ntm/data.npz
// (tflow's scripts/fetch_datasets.py; found through paths::Dataset), copied
// into the run directory like tflow does. 851 parameters in 8 tensors, Adam
// .005, batch 8 with BatchOrder(128, 8, 45107), three epochs (48 updates).
// Checks: parameter counts, finite validation CE, bit-exact .npz restore,
// label-free repeat inference, a single native NtmCell step from every recorded
// state reproduces the unrolled path (2e-5), arbitrary input padding after the
// recall window cannot change any output or gradient, weights immutable under
// inference. address_fixture.npz records the addressing pipeline and its
// gradients (compared with tflow's fixture when present).
// Deviations: checkpoints are final.npz (parameter store) instead of SaveV2;
// weights initialise from nn's generator (statistical parity) unless
// init=PATH (tflow's output/models/ntm/initial_arrays.npz) replays tflow's run,
// in which case every record is compared with tflow's run directory (tf=PATH).
// Options: epochs=3 out=PATH init=PATH tf=PATH. Output: output/nn/models/ntm/.
#include <cmath>
#include <sstream>

#include "common/ntm.hpp"

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

void AddressFixture(const fs::path& root) {
  std::vector<float> memory(32), parameters(20);
  for (int i = 0; i < 32; ++i) memory[i] = float(.3 * std::sin(i * .7) + .05);
  for (int i = 0; i < 20; ++i) parameters[i] = float(.4 * std::cos(i * .3));
  Tensor m = nn::from_vector(memory, {2, 4, 4}), p = nn::from_vector(parameters, {2, 10});
  Tensor v     = nn::from_vector(std::vector<float>{.7f, .1f, .15f, .05f, .1f, .2f, .3f, .4f}, {2, 4});
  Tensor probe = nn::from_vector(std::vector<float>{.2f, -.8f, .5f, .9f, .8f, -.3f, .7f, -.4f}, {2, 4});
  m.requires_grad_(true), p.requires_grad_(true), v.requires_grad_(true);
  const auto address = NtmAddressing(m, p, v);
  const auto g       = nn::grad(std::vector<Tensor>{nn::sum(nn::mul(address.weights, probe))}, {m, p, v});
  const fs::path file = root / "address_fixture.npz";
  nn::save_npz(file.string(), {{"memory", m.detach()},
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
  for (float w : address.weights.to_vector<float>()) total += w;
  check::Near(total, 2, 1e-5, "address weights are distributions");
}

}  // namespace

int main(int argc, char** argv) {
  return Main(argc, argv, "ntm", [](const rnn_cells::DemoOptions& options) {
    TaskSpec spec;
    spec.name           = "ntm";
    spec.dataset        = "ntm";
    spec.keys           = {"input", "target", "lengths", "source_ids"};
    spec.history_column = "validation_bit_ce";
    spec.print_name     = "validationBitCE";
    spec.test_key       = "test_bit_ce";
    spec.order_seed     = 45107;
    spec.parameters     = 851;
    spec.tensors        = 8;
    spec.count_message  = "851 parameters eight tensors";
    spec.make           = NtmModel;
    auto run            = Train(spec, options, AddressFixture);
    const auto& test    = run.test;
    const auto& out     = run.out;

    // Native streaming: one NtmCell step from the recorded state at t must
    // reproduce the unrolled path at t (logits, interface) and t + 1 (state).
    const auto lengths = test.at("lengths").to_vector<std::int32_t>();
    const std::vector<std::string> names{"logits", "h", "c", "memory", "read", "read_weights", "write_weights", "interface"};
    for (int t = 0; t < 9; ++t) {
      auto     at = [&](const std::string& n, int i) { return nn::select(out.at(n), 1, i); };
      NtmState state{at("h", t), at("c", t), at("memory", t), at("read", t), at("read_weights", t), at("write_weights", t)};
      NtmStep  step;
      {
        nn::NoGradGuard guard;
        step = NtmCell(run.deployed->store, nn::select(test.at("input"), 1, t), state);
      }
      const std::vector<Tensor> fields{step.logits,       step.state.h,             step.state.c,
                                       step.state.memory, step.state.read,          step.state.read_weights,
                                       step.state.write_weights, step.interface};
      for (int f = 0; f < 8; ++f) {
        const int  next = f > 0 && f < 7;
        const auto a = fields[f].to_vector<float>(), e = at(names[f], t + next).to_vector<float>();
        const std::size_t width = a.size() / 32;
        for (int b = 0; b < 32; ++b)
          if (t <= 2 * lengths[b])
            for (std::size_t j = 0; j < width; ++j)
              check::Near(a[b * width + j], e[b * width + j], 2e-5,
                          f == 3 ? "native memory streaming parity" : "native state/head streaming parity");
      }
    }
    const Batch padded  = PadCopyInputs(test, .9f);
    const auto  regular = run.deployed->Evaluate(Rows(test, {0, 1})), padout = run.deployed->Evaluate(padded);
    check::Require(SameMaps(regular, padout), "padding state/loss/gradient invariant");
    SaveMap(run.root / "padding_inputs.npz", padded);
    SaveMap(run.root / "padding_outputs.npz", padout);

    const auto        score = ScoreCopy(run.prediction, test);
    const float       ce    = out.at("loss").item<float>();
    std::ostringstream metrics, summary;
    metrics << "{\"parameters\":851,\"epochs\":" << options.epochs << ",\"updates\":" << run.updates << ",\"test_bit_ce\":" << ce
            << ",\"bit_accuracy\":" << double(score.correct) / score.total << ",\"exact\":" << score.exact << "}";
    summary << "testBitCE=" << ce << " bitAccuracy=" << double(score.correct) / score.total << " exact=" << score.exact
            << "/32;restore/nativeStreaming/padding/labelFree/immutable PASS";
    Finish(spec, run, metrics.str(), summary.str(),
           "<p>Binary copy: a sequence of up to four 3-bit vectors, a delimiter, then recall. A Neural Turing Machine "
           "(LSTM(8) controller, 4x4 memory, one content+location read head and one write head with erase/add) is trained "
           "with masked bit cross-entropy on the recall steps for three epochs.</p>");
    const fs::path tf_fixture = options.tf / "address_fixture.npz";
    if (fs::exists(tf_fixture)) {
      const double d = rnn_cells::CompareNpz(run.root / "address_fixture.npz", tf_fixture);
      std::cout << "  address_fixture.npz max|diff| vs tflow=" << d << "\n";
      check::Require(d <= 1e-5, "addressFixtureMatchesTflow");
    }
  });
}
