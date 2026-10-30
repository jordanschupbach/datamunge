// CMD: just run-nn recurrent_dropout
// Per-time-step versus sequence-shared (variational) hidden-to-hidden dropout
// masks on the explicit unrolled tanh RNN; three cheap epochs each.
//
// Port of tflow's recurrent_dropout (common/ml/recurrent_dropout.h). No
// dataset: the 512/128/128 marked sequences are generated with tflow's
// std::mt19937 recipe (bit-identical data.npz). h_t = tanh(x_t Wx +
// (h_{t-1} * mask_t) Wh + b), 206 parameters, dropout p .2 with inverted scale
// 1.25 on the recurrent input only. Two runs from the same initial weights:
// per_step/ (fresh mask per time step) and variational/ (one mask per
// sequence repeated across time); masks come from std::mt19937 seeds
// 3000051 + step exactly as in tflow (bit-identical training_masks.npz).
// Inference feeds an all-ones mask. Adam .01, batch 32, three epochs.
// Checks: tflow's marked-sequence audit for both runs, a changed training mask
// actually changes outputs, a fixed mask replays bit-exactly, inference masks
// are all ones, both runs start from bit-identical parameters.
// Weights initialise from nn's generator (statistical parity); init=PATH
// (tflow's output/models/recurrent_dropout/per_step/initial_arrays.npz) replays
// tflow's runs (tf=PATH is tflow's recurrent_dropout directory).
// Options: epochs=3 out=PATH init=PATH tf=PATH.
// Output: output/nn/models/recurrent_dropout/{per_step,variational}/ + index.html.
#include "common/recurrent_dropout.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

#include "common/marked_sequence_demo.hpp"

// Experiment workflow from examples/nn/common/marked_sequence_demo.hpp.
namespace rnn_cells {
using namespace ::rnn_cells;
} // namespace rnn_cells

using namespace rnn_cells;

namespace {

void MaskPng(const fs::path& file, const Tensor& mask) {
  const auto         m = mask.to_vector<float>();
  std::vector<float> rgb(416 * 96 * 3, 220.f);
  const int          drop[] = {30, 90, 190};
  for (int b = 0; b < 4; ++b)
    for (int t = 0; t < 12; ++t)
      for (int j = 0; j < 12; ++j)
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x)
            for (int c = 0; c < 3; ++c)
              rgb[(static_cast<std::size_t>(b * 104 + t * 8 + y) * 96 + j * 8 + x) * 3 + c] =
                  m[(static_cast<std::size_t>(b) * 12 + t) * 12 + j] > 0 ? 250.f : float(drop[c]);
  nn::save_png(nn::from_vector(rgb, {416, 96, 3}), file.string(), 0, 255);
}

MarkedDemo Spec(bool shared) {
  auto       randomness = std::make_shared<nn::NamedTensors>();
  MarkedDemo d;
  d.name = "recurrent_dropout", d.parameters = 206, d.tensors = 5;
  d.count_message   = "206SharedTanhRNNParameters5Tensors";
  d.restore_message = "all206WeightsBitexactRestore";
  d.make            = [](bool training) { return RecurrentDropoutModel(training); };
  d.title           = shared ? "Sequence-shared variational recurrent mask"
                             : "Fresh recurrent mask at each time step";
  d.description =
      "h_t = tanh(x_t Wx + (h_previous * recurrent_mask_t) Wh + b), 206 parameters across 12 "
      "steps, 12 hidden units, final "
      "12->2 head, full BPTT. Dropout ONLY on the hidden-to-hidden input, probability .2, inverted "
      "scale 1.25. The per-time "
      "mask is fresh for each valid time and sequence; the variational mask is one 12-vector per "
      "sequence repeated across time, "
      "resampled per batch. Same initial weights, data, batches and master uniform bank for both "
      "modes. Inference feeds an "
      "ALL-ONES mask (dropout disabled). A restricted time-sharing illustration (Gal and "
      "Ghahramani motivate temporal sharing), "
      "not a Bayesian language-model replication. first_mask.png: first 4 training masks, 12 time "
      "rows x 12 unit columns, white "
      "keep, blue drop.";
  d.before_batch =
      [shared, randomness](
          RecurrentModel& model, const MarkedSequences& x, const fs::path& root, int step) {
        auto [mask, uniform] = DropoutMask(static_cast<int>(x.size()), 3000051 + step, shared);
        randomness->emplace_back("step" + std::to_string(step) + "_uniform", uniform);
        randomness->emplace_back("step" + std::to_string(step) + "_mask", mask);
        if (!step) {
          nn::save_npz((root / "first_mask.npz").string(), {{"mask", mask}, {"uniform", uniform}});
          MaskPng(root / "first_mask.png", mask);
        }
        model.side_inputs["recurrent_mask"] = mask;
      };
  d.after_update = [](RecurrentModel& model, const MarkedSequences&, const fs::path&, int) {
    model.side_inputs.clear();
  };
  d.after_training = [randomness](DemoContext& ctx) {
    nn::save_npz((ctx.root / "training_masks.npz").string(), *randomness);
    // final_gradients.npz uses the first batch with its training mask, as in tflow.
    ctx.model.side_inputs["recurrent_mask"] =
        Find(nn::load_npz((ctx.root / "first_mask.npz").string()), "mask");
  };
  d.after_counterfactuals = [shared](DemoContext& ctx) {
    auto&      deployed   = *ctx.deployed;
    const auto first_data = LoadData(ctx.root / "first_batch.npz");
    const auto first_mask = Find(nn::load_npz((ctx.root / "first_mask.npz").string()), "mask");
    auto       evaluate   = [&](const Tensor& mask) {
      if (mask.defined())
        deployed.side_inputs["recurrent_mask"] = mask;
      auto out = deployed.Evaluate(first_data.feed());
      deployed.side_inputs.clear();
      return out;
    };
    const auto train_probe               = evaluate(first_mask);
    auto [changed_mask, changed_uniform] = DropoutMask(32, 30101, shared);
    const auto changed = evaluate(changed_mask), without = evaluate(Tensor());
    const auto a       = train_probe.at("logits").to_vector<float>(),
               b       = changed.at("logits").to_vector<float>();
    double mask_effect = 0;
    for (int i = 0; i < 64; ++i)
      mask_effect = std::max(mask_effect, std::abs(double(a[i]) - b[i]));
    check::Require(mask_effect > 1e-5, "explicitRecurrentMaskActuallyChangesTrainingModeOutputs");
    const auto train_repeat = evaluate(first_mask);
    for (const auto& [key, value] : train_probe)
      check::Require(SameBits(value, train_repeat.at(key)), "fixedDropoutMaskReplayBitexact");
    SaveOutputs(ctx.root / "training_mode_probe.npz", train_probe);
    SaveOutputs(ctx.root / "changed_mask_probe.npz", changed);
    SaveOutputs(ctx.root / "without_dropout_probe.npz", without);
    nn::save_npz(
        (ctx.root / "changed_mask.npz").string(),
        {{"mask", changed_mask}, {"uniform", changed_uniform}});
    // Prediction equals evaluation bit for bit (checked by the demo), so this is the label-free
    // mask.
    for (float v : ctx.output.at("recurrent_mask").to_vector<float>())
      check::Require(v == 1, "inferenceDropoutAlwaysDisabledAllOnesMask");
    ctx.metrics
        << ",\"dropout_probability\":0.2,\"inverted_scale\":1.25,\"mask_shared_across_time\":"
        << (shared ? "true" : "false")
        << ",\"changed_training_mask_max_logit_effect\":" << mask_effect;
  };
  return d;
}

} // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(4);
    const DemoOptions options = ParseDemoOptions(argc, argv, "recurrent_dropout");
    for (const auto& [sub, shared] :
         {std::pair<const char*, bool>{"per_step", false}, {"variational", true}}) {
      DemoOptions o = options;
      o.out         = options.out / sub;
      o.tf          = options.tf / sub;
      std::cout << "== " << sub << "\n";
      {
        const auto     d       = Spec(shared);
        const auto&    options = o;
        const auto     start   = std::chrono::steady_clock::now();
        const fs::path root    = options.out;
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
              nn::save_npz((root / "initial_gradients.npz").string(), model->Gradient(x.feed()));
            }
            if (d.before_update)
              d.before_update(*model, x, root, updates);
            const auto feed    = x.feed();
            const auto outputs = model->forward(
                model->store, feed.inputs, feed.lengths, model->Initials(feed.inputs.shape()[0]));
            const auto      parameters = model->store.parameters();
            nn::GradOptions grad_options;
            grad_options.allow_unused = true;
            auto gradients            = nn::grad(
                std::vector<Tensor>{model->loss_fn(outputs, feed.labels).first},
                parameters,
                {},
                grad_options);
            for (std::size_t i = 0; i < gradients.size(); ++i)
              gradients[i] = DefinedGradient(gradients[i], parameters[i]);
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
            model->Gradient(LoadData(root / "first_batch.npz").feed()));
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
        auto idx = [](int b, int t, int c) {
          return (static_cast<std::size_t>(b) * 12 + t) * 2 + c;
        };
        auto sidx = [](int b, int t, int j) {
          return (static_cast<std::size_t>(b) * 13 + t) * 12 + j;
        };
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
                "system-ui;margin:30px;max-width:1100px}img{max-width:100%;image-rendering:"
                "pixelated}</style>"
             << "<h1>" << d.title << "</h1><p>" << d.description << "</p>"
             << "<p>Synthetic marked-token memory task: 512 fit / 128 validation / 128 test, "
                "balanced labels, valid length 4..12, one "
                "marker before the last token whose +/-1 value sets the binary target; padding "
                "zeros. Three epochs, 48 updates. The task "
                "has a perfect deterministic marker-times-value oracle; it is not a "
                "natural-language benchmark.</p>"
             << "<h2>First 24 held-out sequences</h2><p>12 time columns; blue=-1, red=+1, "
                "white=padding, black top strip=marked "
                "token.</p><img src='sequences.png'><h2>Target-logit input-value "
                "gradients</h2><p>Same cases, global symmetric colour "
                "peak. Masked padding derivatives are exactly zero.</p><img "
                "src='input_gradients.png'>"
             << "<p>Test CE " << float(test_ce) << ", accuracy " << accuracy
             << ", cue-flip accuracy " << cue_accuracy << ", last-distractor-flip accuracy "
             << last_accuracy << ", last-value baseline " << last_correct / 128.
             << ". See metrics.json and predictions.tsv.</p>";
        check::Require(
            bool(meta) && bool(html) && bool(history) && bool(batches) && bool(predictions),
            "completeUnrolledRNNRecords");
        meta.close(), html.close(), history.close(), batches.close(), predictions.close();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "testCE=" << float(test_ce) << " accuracy=" << accuracy
                  << " cueFlip=" << cue_accuracy << " lastBaseline=" << last_correct / 128. << ";"
                  << d.pass_tag << " PASS (" << seconds
                  << " s)\nArtifacts:" << (root / "index.html") << std::endl;
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
      }
    }
    check::Require(
        SameBits(
            nn::load_npz((options.out / "per_step/initial_arrays.npz").string()),
            nn::load_npz((options.out / "variational/initial_arrays.npz").string())),
        "bothDropoutModelsStartWithBitexactSameParameters");
    std::ofstream index(options.out / "index.html");
    index << "<!doctype html><meta charset=utf-8><title>Recurrent dropout</title><h1>Recurrent "
             "dropout: per-time versus "
             "sequence-shared masks</h1><p>Matched source, initial weights, batches and "
             "three-epoch budgets, dropout p .2; "
             "inference uses an all-ones mask.</p><p><a href=per_step/index.html>Fresh per-step "
             "masks</a></p><p><a "
             "href=variational/index.html>Sequence-shared variational masks</a></p>";
    check::Require(bool(index), "recurrentDropoutIndexWritten");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
