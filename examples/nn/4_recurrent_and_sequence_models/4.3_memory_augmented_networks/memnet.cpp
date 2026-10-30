// CMD: just run-nn memnet
// Two-hop end-to-end memory network (adjacent embedding, answer tying) on
// synthetic fact/query question answering.
//
// Port of tflow's memnet (common/nn/memnet.h -> examples/nn/common/memnet.hpp,
// shared training flow in common/memory_task.hpp). Data:
// output/datasets/memnet/data.npz (tflow's scripts/fetch_datasets.py; found
// through paths::Dataset), copied into the run directory like tflow does. 312
// parameters in 3 tied embedding tables, Adam .01, batch 8 with
// BatchOrder(128, 8, 45307), three epochs (48 updates). Checks: parameter
// counts, bit-exact .npz restore, label-free repeat prediction, padding facts
// cannot change any field or fact/query gradient, empty memories read zero,
// fact order cannot change the answer (2e-6), weights immutable under
// inference. (tflow's header comment for this file is a copy of dnc.cc's.)
// Deviations: checkpoints are final.npz (parameter store) instead of SaveV2;
// weights initialise from nn's generator (statistical parity) unless
// init=PATH (tflow's output/models/memnet/initial_arrays.npz) replays tflow's
// run, in which case every record is compared with tflow's run directory (tf=PATH).
// Options: epochs=3 out=PATH init=PATH tf=PATH. Output: output/nn/models/memnet/.
// Local assets: just nn-memory-assets; set DATAMUNGE_DATA_DIR to build/datasets.
#include "common/memory_data.hpp"
#include "common/memnet.hpp"

using namespace memory_nets;
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    const auto options = rnn_cells::ParseDemoOptions(argc, argv, "memnet");
    TaskSpec   spec;
    spec.name                                   = "memnet";
    spec.order_seed                             = 45307;
    spec.parameters                             = 312;
    spec.tensors                                = 3;
    spec.count_message                          = "312 parameters three tied embedding tables";
    const std::vector<std::string> empty_fields = {
        "m0", "m1", "m2", "attention", "reads", "input_gradient"};
    const std::string description =
        "<p>Synthetic question answering over up to six two-token facts. A two-hop end-to-end "
        "memory network with "
        "position-weighted sentence embeddings, adjacent weight tying (C^k = A^{k+1}) and answers "
        "tied to the last "
        "table is trained with softmax cross-entropy for three epochs.</p>";
    spec.dataset        = "memnet";
    spec.keys           = {"facts", "query", "target", "lengths", "source_ids"};
    spec.history_column = "validation_ce";
    spec.print_name     = "validationCE";
    spec.test_key       = "test_ce";
    TaskRun run;
    run.start            = std::chrono::steady_clock::now();
    run.options          = options;
    run.root             = options.out;
    const fs::path& root = run.root;
    fs::create_directories(root);
    const fs::path source = paths::Dataset(spec.dataset);
    memory_data::PrepareDataset(source.string(), false);
    for (const char* f : {"data.npz", "manifest.json"})
      if (fs::exists(source / f))
        fs::copy_file(source / f, root / f, fs::copy_options::overwrite_existing);

    const auto all      = nn::load_npz((root / "data.npz").string());
    run.fit             = Split(all, "fit", spec.keys);
    run.validation      = Split(all, "validation", spec.keys);
    run.test            = Split(all, "test", spec.keys);
    const Batch example = Rows(run.fit, {0});
    run.model           = MemNetModel(true, example);
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
    run.deployed = MemNetModel(false, example);
    run.deployed->Restore(root / "final.npz");
    run.weights = run.deployed->Arrays();
    check::Require(SameBits(m.Arrays(), run.weights), "restore bitexact");
    run.out        = run.deployed->Evaluate(run.test);
    run.prediction = run.deployed->Predict(run.test);
    check::Require(
        SameBits(run.prediction, run.out.at("logits")) &&
            SameBits(run.prediction, run.deployed->Predict(run.test)),
        "label-free repeat inference");
    auto&       deployed  = *run.deployed;
    const Batch normal_in = Rows(run.test, {0, 1});
    const auto  normal    = deployed.Evaluate(normal_in);
    const auto  lengths   = normal_in.at("lengths").to_vector<std::int32_t>();
    const auto  facts     = normal_in.at("facts").to_vector<std::int32_t>();
    auto        at        = [](int b, int t, int j) { return (b * 6 + t) * 2 + j; };

    Batch pad    = normal_in;
    auto  padded = facts;
    for (int b = 0; b < 2; ++b)
      for (int t = lengths[b]; t < 6; ++t)
        for (int j = 0; j < 2; ++j)
          padded[at(b, t, j)] = 11;
    pad["facts"]          = nn::from_vector(padded, {2, 6, 2});
    const auto pad_output = deployed.Evaluate(pad);
    for (const auto& [name, value] : normal)
      check::Require(
          SameBits(value, pad_output.at(name)), "all masked fields/partials invariant: " + name);
    SaveMap(run.root / "padding_inputs.npz", pad);
    SaveMap(run.root / "padding_outputs.npz", pad_output);

    Batch empty      = normal_in;
    empty["lengths"] = nn::zeros({2}, nn::DType::Int32);
    const auto eo    = deployed.Evaluate(empty);
    for (const auto& name : empty_fields)
      for (float v : eo.at(name).to_vector<float>())
        check::Require(v == 0, "empty memory reads/partials zero");
    SaveMap(run.root / "empty_inputs.npz", empty);
    SaveMap(run.root / "empty_outputs.npz", eo);

    Batch shuffled = normal_in;
    auto  swapped  = facts;
    for (int b = 0; b < 2; ++b)
      for (int t = 0; t < lengths[b] / 2; ++t)
        for (int j = 0; j < 2; ++j)
          std::swap(swapped[at(b, t, j)], swapped[at(b, lengths[b] - 1 - t, j)]);
    shuffled["facts"] = nn::from_vector(swapped, {2, 6, 2});
    const auto so     = deployed.Evaluate(shuffled);
    for (const char* name : {"logits", "states", "reads", "loss"}) {
      const auto a = so.at(name).to_vector<float>(), e = normal.at(name).to_vector<float>();
      for (std::size_t j = 0; j < a.size(); ++j)
        check::Near(a[j], e[j], 2e-6, "fact-order invariance");
    }
    SaveMap(run.root / "shuffled_inputs.npz", shuffled);
    SaveMap(run.root / "shuffled_outputs.npz", so);

    const auto pred    = run.prediction.to_vector<float>();
    const auto target  = run.test.at("target").to_vector<std::int32_t>();
    int        correct = 0;
    for (int b = 0; b < 32; ++b) {
      int best = 0;
      for (int j = 1; j < 4; ++j)
        if (pred[b * 4 + j] > pred[b * 4 + best])
          best = j;
      correct += best == target[b];
    }
    nn::ConfusionMatrix confusion(4);
    confusion.add(run.test.at("target"), nn::argmax(run.prediction, 1));
    std::cout << "confusion (truth rows, predicted columns):\n";
    for (const auto& row : confusion.cells()) {
      for (auto count : row)
        std::cout << count << ' ';
      std::cout << '\n';
    }
    const float        ce = run.out.at("loss").item<float>();
    std::ostringstream metrics, summary;
    metrics << "{\"parameters\":" << spec.parameters << ",\"epochs\":" << options.epochs
            << ",\"updates\":" << run.updates << ",\"test_ce\":" << ce << ",\"correct\":" << correct
            << "}";
    summary << "testCE=" << ce << " correct=" << correct
            << "/32;restore/padding/empty/factOrder/labelFree/immutable PASS";
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
             << spec.name << "</h1>" << description << "<p>" << summary.str()
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
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
