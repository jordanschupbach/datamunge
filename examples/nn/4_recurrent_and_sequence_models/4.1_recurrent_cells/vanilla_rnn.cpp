// CMD: just run-nn vanilla_rnn
// Explicit shared-weight unrolled tanh RNN on a variable-length marked-token
// memory task, three cheap epochs.
//
// Port of tflow's vanilla_rnn (common/ml/unrolled_rnn.h). No dataset: the
// 512/128/128 marked sequences are generated with tflow's std::mt19937 recipe
// (bit-identical data.npz). h_t = tanh(x_t Wx + h_{t-1} Wh + b), 206 parameters
// in 5 tensors, 12 hidden units, 12->2 head; Adam .01, batch 32, three epochs
// (48 updates). Checks: tflow's marked-sequence audit (counts, bit-exact .npz
// restore, label-free prediction == evaluation, repeat/single parity, padding
// garbage cannot change any output or gradient, last-token causality,
// zero-length sequences, weights immutable under inference).
// Weights initialise from nn's generator (statistical parity); init=PATH
// (tflow's output/models/vanilla_rnn/initial_arrays.npz) replays tflow's run.
// Options: epochs=3 out=PATH init=PATH tf=PATH. Output: output/nn/models/vanilla_rnn/.
#include "common/marked_sequence_demo.hpp"
#include "common/unrolled_rnn.hpp"

using namespace rnn_cells;

int main(int argc, char** argv) {
  try {
    MarkedDemo d;
    d.name = "vanilla_rnn", d.parameters = 206, d.tensors = 5;
    d.count_message   = "206SharedTanhRNNParameters5Tensors";
    d.restore_message = "all206WeightsBitexactRestore";
    d.make            = [](bool training) { return UnrolledRnnModel(training); };
    d.title           = "Vanilla tanh RNN from an explicit unrolled graph";
    d.description =
        "h_t = tanh(x_t Wx + h_previous Wh + b), the same 206 parameters across 12 steps, 12 "
        "hidden units, final 12->2 linear "
        "head. Primitive nn ops, full BPTT through all valid steps, no opaque RNN op. The explicit "
        "length mask retains the state "
        "after each sequence ends; zero initial state per sequence; prediction feeds only inputs "
        "and lengths.";
    nn::set_num_threads(4);
    const auto options = ParseDemoOptions(argc, argv, d.name);
    // Raw loss gradients are computed here, before the example applies its
    // gradient policy and optimizer. Diagnostics reuse this same calculation.
    auto loss_gradients = [](RecurrentModel& network, const Feed& feed) {
      const auto outputs = network.forward(
          network.store, feed.inputs, feed.lengths, network.Initials(feed.inputs.shape()[0]));
      const auto      parameters = network.store.parameters();
      nn::GradOptions gradient_options;
      gradient_options.allow_unused = true;
      auto gradients                = nn::grad(
          std::vector<Tensor>{network.loss_fn(outputs, feed.labels).first},
          parameters,
          {},
          gradient_options);
      for (std::size_t i = 0; i < gradients.size(); ++i)
        gradients[i] = DefinedGradient(gradients[i], parameters[i]);
      return gradients;
    };
    auto named_gradients = [&](RecurrentModel& network, const Feed& feed) {
      const auto       gradients = loss_gradients(network, feed);
      nn::NamedTensors result;
      for (std::size_t i = 0; i < gradients.size(); ++i)
        result.emplace_back(network.store.names()[i], gradients[i]);
      return result;
    };

    const auto     start = std::chrono::steady_clock::now();
    const fs::path root  = options.out;
    fs::create_directories(root);
    nn::NamedTensors data;
    auto             pools = MarkedPools(&data);
    nn::save_npz((root / "data.npz").string(), data);
    auto model = d.make(true);
    check::Require(
        model->store.count() == d.parameters &&
            static_cast<std::int64_t>(model->store.size()) == d.tensors,
        d.count_message);
    if (!options.init.empty())
      model->LoadArrays(options.init);
    DemoContext ctx{root, pools, *model, nullptr, {}, {}, {}, model->Arrays(), {}};
    nn::save_npz((root / "initial_arrays.npz").string(), ctx.initial_arrays);
    SaveOutputs(root / "initial_outputs.npz", model->Evaluate(pools[2].feed()));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history.precision(17);
    history << "epoch\tvalidation_ce\tvalidation_accuracy\n";
    batches << "epoch\tstep\trow\n";
    nn::BatchOrder order(512, 32, 26019);
    int            updates = 0;
    for (int epoch = 1; epoch <= options.epochs; ++epoch) {
      order.begin_epoch();
      for (;;) {
        const auto rows = order.next();
        if (rows.empty())
          break;
        const auto x = MarkedRows(pools[0], rows);
        if (d.before_batch)
          d.before_batch(*model, x, root, updates);
        if (!updates) {
          SaveData(root / "first_batch.npz", x);
          SaveOutputs(root / "first_forward.npz", model->Evaluate(x.feed()));
          nn::save_npz(
              (root / "initial_gradients.npz").string(), named_gradients(*model, x.feed()));
        }
        if (d.before_update)
          d.before_update(*model, x, root, updates);
        auto gradients = loss_gradients(*model, x.feed());
        if (model->gradient_policy)
          gradients = model->gradient_policy(gradients);
        model->optimizer->step(gradients);
        if (model->after_update)
          model->after_update(model->store);
        if (d.after_update)
          d.after_update(*model, x, root, updates);
        if (!updates)
          nn::save_npz((root / "first_after.npz").string(), model->Arrays());
        for (auto row : rows)
          batches << epoch << '\t' << updates << '\t' << row << '\n';
        ++updates;
      }
      const auto   valid    = model->Evaluate(pools[1].feed());
      const double ce       = valid.at("loss").item<double>(),
                   accuracy = Accuracy(valid.at("logits"), pools[1].labels);
      history << epoch << '\t' << float(ce) << '\t' << accuracy << '\n';
      nn::save_npz(
          (root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), model->Arrays());
      SaveOutputs(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), valid);
      std::cout << "epoch" << epoch << " validationCE=" << float(ce) << " accuracy=" << accuracy
                << std::endl;
    }
    batches.flush();
    if (d.after_training)
      d.after_training(ctx);
    nn::save_npz(
        (root / "final_gradients.npz").string(),
        named_gradients(*model, LoadData(root / "first_batch.npz").feed()));
    model->Save(root / "final.npz");
    auto deployed = d.make(false);
    deployed->Restore(root / "final.npz");
    ctx.deployed = deployed.get();
    check::Require(SameBits(model->Arrays(), deployed->Arrays()), d.restore_message);
    const auto before      = deployed->Arrays();
    auto&      test        = pools[2];
    ctx.output             = deployed->Evaluate(test.feed());
    const auto  prediction = deployed->Predict(test.feed());
    const auto& output     = ctx.output;
    for (const auto& [k, v] : prediction)
      check::Require(SameBits(v, output.at(k)), "labelFreePredictionMatchesEvaluation");
    nn::save_npz((root / "model_arrays.npz").string(), deployed->Arrays());
    SaveOutputs(root / "outputs.npz", output);
    if (d.after_outputs)
      d.after_outputs(ctx);
    const auto repeat = deployed->Predict(test.feed());
    for (const auto& [k, v] : prediction)
      check::Require(SameBits(v, repeat.at(k)), "RNNPredictRepeatBitexact");
    const auto single        = deployed->Predict(MarkedRows(test, {0}).feed());
    const auto single_logits = single.at("logits").to_vector<float>(),
               logits        = output.at("logits").to_vector<float>();
    for (int c = 0; c < 2; ++c)
      check::Near(single_logits[c], logits[c], 2e-6, "singleBatchRNNParityWithResetState");

    // Counterfactual copies: padding garbage, flipped marked value, flipped last valid value.
    const auto test_inputs    = test.inputs.to_vector<float>();
    const auto lengths        = test.lengths.to_vector<std::int32_t>(),
               labels         = test.labels.to_vector<std::int32_t>();
    const auto input_gradient = output.at("input_gradient").to_vector<float>(),
               states         = output.at("states").to_vector<float>();
    auto padded = test_inputs, cue_flip = test_inputs, last_flip = test_inputs;
    auto cue_labels     = labels;
    int  oracle_correct = 0, last_correct = 0;
    auto idx  = [](int b, int t, int c) { return (static_cast<std::size_t>(b) * 12 + t) * 2 + c; };
    auto sidx = [](int b, int t, int j) { return (static_cast<std::size_t>(b) * 13 + t) * 12 + j; };
    for (int b = 0; b < 128; ++b) {
      const int length   = lengths[b];
      double    selected = 0;
      for (int t = 0; t < 12; ++t) {
        if (t >= length) {
          padded[idx(b, t, 0)] = 123.f + b + t;
          padded[idx(b, t, 1)] = -57.f;
          for (int c = 0; c < 2; ++c)
            check::Require(
                input_gradient[idx(b, t, c)] == 0, "ALLpaddedInputLogitGradientExactlyZero");
          for (int j = 0; j < 12; ++j)
            check::Require(
                states[sidx(b, t + 1, j)] == states[sidx(b, t, j)],
                "stateRetainedAcrossEveryPaddingStep");
        }
        if (test_inputs[idx(b, t, 1)] > 0) {
          selected += test_inputs[idx(b, t, 0)];
          cue_flip[idx(b, t, 0)] *= -1;
        }
      }
      oracle_correct += (selected > 0) == (labels[b] != 0);
      last_correct += (test_inputs[idx(b, length - 1, 0)] > 0) == (labels[b] != 0);
      cue_labels[b] = 1 - labels[b];
      last_flip[idx(b, length - 1, 0)] *= -1;
    }
    check::Require(oracle_correct == 128, "deterministicMarkerValueOracle");
    MarkedSequences pad_data = Copy(test), cue_data = Copy(test), last_data = Copy(test);
    pad_data.inputs       = nn::from_vector(padded, {128, 12, 2});
    cue_data.inputs       = nn::from_vector(cue_flip, {128, 12, 2});
    cue_data.labels       = nn::from_vector(cue_labels);
    last_data.inputs      = nn::from_vector(last_flip, {128, 12, 2});
    const auto pad_output = deployed->Evaluate(pad_data.feed());
    ctx.cue_output        = deployed->Evaluate(cue_data.feed());
    ctx.last_output       = deployed->Evaluate(last_data.feed());
    for (const auto& [k, v] : output)
      check::Require(
          SameBits(v, pad_output.at(k)),
          "arbitraryPaddingValuesCannotAffectAnyOutputOrGradient:" + k);
    const auto last_states = ctx.last_output.at("states").to_vector<float>();
    for (int b = 0; b < 128; ++b)
      for (int t = 0; t < lengths[b]; ++t)
        for (int j = 0; j < 12; ++j)
          check::Require(
              last_states[sidx(b, t, j)] == states[sidx(b, t, j)],
              "changingLastValidTokenCannotChangeEarlierHiddenStates");
    if (d.after_counterfactuals)
      d.after_counterfactuals(ctx);
    SaveData(root / "padding_inputs.npz", pad_data);
    SaveData(root / "cue_flip_inputs.npz", cue_data);
    SaveData(root / "last_flip_inputs.npz", last_data);
    SaveOutputs(root / "padding_outputs.npz", pad_output);
    SaveOutputs(root / "cue_flip_outputs.npz", ctx.cue_output);
    SaveOutputs(root / "last_flip_outputs.npz", ctx.last_output);
    auto empty              = MarkedRows(test, {0, 1, 2, 3});
    empty.lengths           = nn::zeros({4}, nn::DType::Int32);
    const auto empty_output = deployed->Evaluate(empty.feed());
    for (float v : empty_output.at("states").to_vector<float>())
      check::Require(v == 0, "zeroLengthSequenceRetainsZeroInitialState");
    for (float v : empty_output.at("input_gradient").to_vector<float>())
      check::Require(v == 0, "zeroLengthAllInputGradientsZero");
    SaveData(root / "zero_length_inputs.npz", empty);
    SaveOutputs(root / "zero_length_outputs.npz", empty_output);
    Heatmap(root / "sequences.png", test.inputs);
    Heatmap(root / "input_gradients.png", output.at("input_gradient"), true);
    std::ofstream predictions(root / "predictions.tsv");
    predictions << "source_id\tlength\tlabel\tpredicted\tmarked_oracle\tlast_value_baseline\n";
    const auto ids = test.source_ids.to_vector<std::int32_t>();
    for (int b = 0; b < 128; ++b)
      predictions << ids[b] << '\t' << lengths[b] << '\t' << labels[b] << '\t'
                  << (logits[b * 2 + 1] > logits[b * 2]) << '\t' << labels[b] << '\t'
                  << (test_inputs[idx(b, lengths[b] - 1, 0)] > 0) << '\n';
    if (d.after_predictions)
      d.after_predictions(ctx);
    check::Require(
        SameBits(before, deployed->Arrays()),
        "allRecurrentInferenceAndGradientControlsWeightsImmutable");
    nn::ConfusionMatrix confusion(2);
    confusion.add(test.labels, nn::argmax(output.at("logits"), 1));
    std::uint64_t evaluated = 0;
    std::cout << "  confusion (truth rows):\n";
    for (const auto& row : confusion.cells()) {
      std::cout << "   ";
      for (auto count : row) {
        std::cout << ' ' << count;
        evaluated += count;
      }
      std::cout << '\n';
    }
    check::Require(evaluated == 128, "all held-out recurrent sequences evaluated");
    check::Near(
        confusion.accuracy(),
        Accuracy(output.at("logits"), test.labels),
        1e-7,
        "recurrent confusion agrees with accuracy");
    const double test_ce        = output.at("loss").item<double>(),
                 accuracy       = Accuracy(output.at("logits"), test.labels);
    const double  cue_accuracy  = Accuracy(ctx.cue_output.at("logits"), cue_data.labels);
    const double  last_accuracy = Accuracy(ctx.last_output.at("logits"), test.labels);
    std::ofstream meta(root / "metrics.json");
    meta.precision(17);
    meta << "{" << d.metrics_prefix << "\"parameters\":" << d.parameters
         << ",\"parameter_tensors\":" << d.tensors
         << ",\"hidden_units\":12,\"maximum_steps\":12,\"epochs\":" << options.epochs
         << ",\"optimizer_updates\":" << updates << ",\"test_ce\":" << float(test_ce)
         << ",\"test_accuracy\":" << accuracy << ",\"cue_flip_accuracy\":" << cue_accuracy
         << ",\"last_distractor_flip_accuracy\":" << last_accuracy
         << ",\"last_value_baseline_accuracy\":" << last_correct / 128.
         << ",\"majority_baseline_accuracy\":0.5,\"marked_value_oracle_accuracy\":1"
         << ctx.metrics.str() << "}\n";
    std::ofstream html(root / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>" << d.title
         << "</title><style>body{font:16px "
            "system-ui;margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</"
            "style>"
         << "<h1>" << d.title << "</h1><p>" << d.description << "</p>"
         << "<p>Synthetic marked-token memory task: 512 fit / 128 validation / 128 test, balanced "
            "labels, valid length 4..12, one "
            "marker before the last token whose +/-1 value sets the binary target; padding zeros. "
            "Three epochs, 48 updates. The task "
            "has a perfect deterministic marker-times-value oracle; it is not a natural-language "
            "benchmark.</p>"
         << "<h2>First 24 held-out sequences</h2><p>12 time columns; blue=-1, red=+1, "
            "white=padding, black top strip=marked "
            "token.</p><img src='sequences.png'><h2>Target-logit input-value gradients</h2><p>Same "
            "cases, global symmetric colour "
            "peak. Masked padding derivatives are exactly zero.</p><img src='input_gradients.png'>"
         << "<p>Test CE " << float(test_ce) << ", accuracy " << accuracy << ", cue-flip accuracy "
         << cue_accuracy << ", last-distractor-flip accuracy " << last_accuracy
         << ", last-value baseline " << last_correct / 128.
         << ". See metrics.json and predictions.tsv.</p>";
    check::Require(
        bool(meta) && bool(html) && bool(history) && bool(batches) && bool(predictions),
        "completeUnrolledRNNRecords");
    meta.close(), html.close(), history.close(), batches.close(), predictions.close();
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "testCE=" << float(test_ce) << " accuracy=" << accuracy
              << " cueFlip=" << cue_accuracy << " lastBaseline=" << last_correct / 128. << ";"
              << d.pass_tag << " PASS (" << seconds << " s)\nArtifacts:" << (root / "index.html")
              << std::endl;
    CompareWithTflow(
        root,
        options,
        {"initial_outputs.npz",
         "first_forward.npz",
         "initial_gradients.npz",
         "first_after.npz",
         "epoch1_arrays.npz",
         "epoch" + std::to_string(options.epochs) + "_arrays.npz",
         "final_gradients.npz",
         "outputs.npz",
         "cue_flip_outputs.npz",
         "last_flip_outputs.npz"});
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
