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
// Local assets: just nn-memory-assets; set DATAMUNGE_DATA_DIR to build/datasets.
#include "common/memory_data.hpp"
#include "common/dnc.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

using namespace memory_nets;

#include "common/memory_validation.hpp"

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    const auto options = rnn_cells::ParseDemoOptions(argc, argv, "dnc");
    TaskSpec   spec;
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
    TaskRun run;
    run.start            = std::chrono::steady_clock::now();
    run.options          = options;
    run.root             = options.out;
    const fs::path& root = run.root;
    fs::create_directories(root);
    const fs::path source = paths::Dataset(spec.dataset);
    memory_data::PrepareDataset(source.string(), true);
    for (const char* f : {"data.npz", "manifest.json"})
      if (fs::exists(source / f))
        fs::copy_file(source / f, root / f, fs::copy_options::overwrite_existing);
    memory_validation::DncFixtures(root.string());
    const auto all      = nn::load_npz((root / "data.npz").string());
    run.fit             = Split(all, "fit", spec.keys);
    run.validation      = Split(all, "validation", spec.keys);
    run.test            = Split(all, "test", spec.keys);
    const Batch example = Rows(run.fit, {0});
    run.model           = DncModel(true, example);
    auto& m             = *run.model;
    check::Require(
        m.store.count() == spec.parameters &&
            static_cast<std::int64_t>(m.store.size()) == spec.tensors,
        spec.count_message);
    auto named_gradients = [&](const Batch& b) {
      const auto      out        = m.forward(m.store, m.leaves(b), b);
      const auto      parameters = m.store.parameters();
      nn::GradOptions grad_options;
      grad_options.allow_unused = true;
      auto gradients = nn::grad(std::vector<Tensor>{m.loss(out, b)}, parameters, {}, grad_options);
      nn::NamedTensors named;
      for (std::size_t i = 0; i < gradients.size(); ++i) {
        if (!gradients[i].defined())
          gradients[i] = nn::zeros_like(parameters[i].detach());
        named.emplace_back(m.store.names()[i], gradients[i].detach());
      }
      return named;
    };
    if (!options.init.empty())
      m.LoadArrays(options.init);
    nn::save_npz((root / "initial_arrays.npz").string(), m.Arrays());
    SaveMap(root / "initial_outputs.npz", m.Evaluate(run.test));
    const auto     fit_rows = static_cast<std::size_t>(run.fit.begin()->second.shape()[0]);
    nn::BatchOrder order(fit_rows, 8, spec.order_seed);
    std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
    history << "epoch\t" << spec.history_column << "\n";
    batches << "epoch\tstep\trow\n";
    for (int e = 1; e <= options.epochs; ++e) {
      order.begin_epoch();
      for (;;) {
        const auto rows = order.next();
        if (rows.empty())
          break;
        const Batch b = Rows(run.fit, rows);
        if (!run.updates) {
          run.first = b;
          SaveMap(root / "first_batch.npz", b);
          nn::save_npz((root / "initial_gradients.npz").string(), named_gradients(b));
          if (spec.first_forward)
            SaveMap(root / "first_forward.npz", m.Evaluate(b));
        }
        const auto      out        = m.forward(m.store, m.leaves(b), b);
        const auto      parameters = m.store.parameters();
        nn::GradOptions grad_options;
        grad_options.allow_unused = true;
        auto gradients =
            nn::grad(std::vector<Tensor>{m.loss(out, b)}, parameters, {}, grad_options);
        for (std::size_t i = 0; i < gradients.size(); ++i)
          if (!gradients[i].defined())
            gradients[i] = nn::zeros_like(parameters[i].detach());
        m.optimizer->step(gradients);
        if (!run.updates)
          nn::save_npz((root / "first_after.npz").string(), m.Arrays());
        for (auto row : rows)
          batches << e << '\t' << run.updates << '\t' << row << '\n';
        ++run.updates;
      }
      const float ce = m.Evaluate(run.validation).at("loss").item<float>();
      check::Require(std::isfinite(ce), "finite CE");
      history << e << '\t' << ce << '\n';
      std::cout << "epoch=" << e << " " << spec.print_name << "=" << ce << std::endl;
    }
    history.close();
    batches.close();
    nn::save_npz((root / "final_gradients.npz").string(), named_gradients(run.first));
    if (spec.first_forward)
      SaveMap(root / "final_first_forward.npz", m.Evaluate(run.first));
    m.Save(root / "final.npz");
    run.deployed = DncModel(false, example);
    run.deployed->Restore(root / "final.npz");
    run.weights = run.deployed->Arrays();
    check::Require(SameBits(m.Arrays(), run.weights), "restore bitexact");
    run.out        = run.deployed->Evaluate(run.test);
    run.prediction = run.deployed->Predict(run.test);
    check::Require(
        SameBits(run.prediction, run.out.at("logits")) &&
            SameBits(run.prediction, run.deployed->Predict(run.test)),
        "label-free repeat inference");
    const auto& test = run.test;
    const auto& out  = run.out;

    const auto lengths = test.at("lengths").to_vector<std::int32_t>();
    for (int t = 0; t < 9; ++t) {
      std::vector<Tensor> values;
      for (const auto& name : DncNames())
        values.push_back(nn::select(out.at(name), 1, t));
      DncStep step;
      {
        nn::NoGradGuard guard;
        step = DncCell(
            run.deployed->store, nn::select(test.at("input"), 1, t), DncState::From(values));
      }
      std::map<std::string, Tensor> fields{
          {"logits", step.logits},
          {"interface", step.interface},
          {"allocation", step.allocation},
          {"forward", step.forward},
          {"backward", step.backward}};
      const auto state = step.state.Values();
      for (std::size_t j = 0; j < state.size(); ++j)
        fields[DncNames()[j]] = state[j];
      for (const auto& [name, value] : fields) {
        const bool matrix = name == "memory" || name == "links";
        const int  next   = fields.count(name) &&
                         std::find(DncNames().begin(), DncNames().end(), name) != DncNames().end();
        const auto a            = value.to_vector<float>(),
                   e            = nn::select(out.at(name), 1, t + next).to_vector<float>();
        const std::size_t width = a.size() / 32;
        for (int b = 0; b < 32; ++b)
          if (t <= 2 * lengths[b])
            for (std::size_t j = 0; j < width; ++j)
              check::Near(
                  a[b * width + j],
                  e[b * width + j],
                  3e-5,
                  matrix ? "DNC streaming matrix parity" : "DNC streaming state/access parity");
      }
    }
    const Batch pad    = PadCopyInputs(test, .93f);
    const auto  normal = run.deployed->Evaluate(Rows(test, {0, 1})),
               padded  = run.deployed->Evaluate(pad);
    check::Require(SameMaps(normal, padded), "all DNC state/gradient padding invariance");
    SaveMap(run.root / "padding_inputs.npz", pad);
    SaveMap(run.root / "padding_outputs.npz", padded);

    const auto score           = ScoreCopy(run.prediction, test);
    int        confusion[2][2] = {};
    const auto predicted_bits  = run.prediction.to_vector<float>();
    const auto target_bits     = test.at("target").to_vector<float>();
    for (std::size_t b = 0; b < lengths.size(); ++b)
      for (int t = lengths[b] + 1; t <= 2 * lengths[b]; ++t)
        for (int bit = 0; bit < 3; ++bit) {
          const auto index = (b * 9 + t) * 3 + bit;
          ++confusion[target_bits[index] > 0][predicted_bits[index] > 0];
        }
    check::Require(
        confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1] == score.total &&
            confusion[0][0] + confusion[1][1] == score.correct,
        "all held-out recall bits counted");
    std::cout << "confusion (truth rows, predicted columns; recall bits):\n";
    for (const auto& row : confusion)
      std::cout << row[0] << ' ' << row[1] << '\n';

    const float        ce = out.at("loss").item<float>();
    std::ostringstream metrics, summary;
    metrics << "{\"parameters\":799,\"epochs\":" << options.epochs << ",\"updates\":" << run.updates
            << ",\"test_bit_ce\":" << ce
            << ",\"bit_accuracy\":" << double(score.correct) / score.total
            << ",\"exact\":" << score.exact << "}";
    summary << "testBitCE=" << ce << " bitAccuracy=" << double(score.correct) / score.total
            << " exact=" << score.exact
            << "/32;restore/nativeAllStateStreaming/padding/labelFree/immutable PASS";
    {
      const fs::path& root = run.root;
      SaveMap(root / "outputs.npz", run.out);
      nn::save_npz((root / "model_arrays.npz").string(), run.weights);
      check::Require(SameBits(run.deployed->Arrays(), run.weights), "immutable inference");
      {
        std::ofstream metrics_file(root / "metrics.json");
        metrics_file << metrics.str() << "\n";
        std::ofstream html(root / "index.html");
        html << "<!doctype html><meta charset='utf-8'><title>" << spec.name
             << "</title><style>body{font:16px system-ui;margin:30px;max-width:1000px}</style><h1>"
             << spec.name << "</h1>"
             << "<p>Binary copy (the NTM task). A differentiable neural computer (LSTM(8) "
                "controller, 4x4 memory, dynamic "
                "allocation from usage with free gates, temporal link matrix, read modes backward "
                "/ forward / content) is "
                "trained with masked bit cross-entropy on the recall steps for three epochs.</p>"
             << "<p>" << summary.str()
             << "</p><p>See metrics.json, history.tsv and the .npz records.</p>";
        check::Require(bool(metrics_file) && bool(html), "complete records");
      }
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - run.start).count();
      std::cout << summary.str() << " (" << seconds << " s)\nArtifacts:" << (root / "index.html")
                << std::endl;
      std::vector<std::string> records{
          "first_batch.npz",
          "initial_arrays.npz",
          "initial_outputs.npz",
          "initial_gradients.npz",
          "first_after.npz",
          "final_gradients.npz",
          "model_arrays.npz",
          "outputs.npz"};
      if (spec.first_forward) {
        records.push_back("first_forward.npz");
        records.push_back("final_first_forward.npz");
      }
      rnn_cells::CompareWithTflow(root, run.options, records, spec.test_key);
    }
    for (const char* f : {"allocation_fixture.npz", "access_fixture.npz", "temporal_fixture.npz"})
      if (fs::exists(options.tf / f)) {
        const double d = rnn_cells::CompareNpz(run.root / f, options.tf / f);
        std::cout << "  " << f << " max|diff| vs tflow=" << d << "\n";
        check::Require(d <= 1e-5, std::string(f) + " matches tflow");
      }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
