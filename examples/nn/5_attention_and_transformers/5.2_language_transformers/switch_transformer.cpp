// Switch sparse Transformer: top1 router, actual dispatch, capacity and
// balance. CMD: just run-nn switch_transformer
// Eager port preserving gathered expert execution and all source controls.
#include "common/switch_transformer.hpp"

#include <fstream>
namespace nn = datamunge::nn;
namespace {
using Model = switch_lm::Model;
using A     = switch_lm::A;
nn::Tensor Lens(int n, int length = 16) {
  return nn::full({n}, length, nn::DType::Int32);
}
void Save(const std::filesystem::path& file, const A& a) {
  nn::save_npz(file.string(), replay::ToNamed(a));
}
void Compare(const A& a, const std::string& name, double tol = 5e-5) {
  auto file = replay::TfModels("switch_transformer") / name;
  if (std::filesystem::exists(file)) {
    double e = replay::MaxDiff(a, file);
    std::cout << "TF " << name << " max difference=" << e << std::endl;
    check::Near(e, 0, tol, "TF Switch " + name);
  }
}
void Fixtures(const std::filesystem::path& root) {
  nn::ParameterStore p(47711);
  {
    nn::NoGradGuard guard;
    switch_lm::Switch(p, nn::zeros({1, 8, 8}), nn::ones({1, 8}, nn::DType::Bool), 2);
  }
  auto tf = replay::TfModels("switch_transformer");
  if (std::filesystem::exists(tf / "fixture_arrays.npz"))
    replay::LoadInto(p, tf / "fixture_arrays.npz");
  Save(root / "fixture_arrays.npz", replay::Arrays(p));
  auto original = p.at("switch/router/W").detach().clone();
  for (const std::string name :
       {"balanced",
        "overflow",
        "tie",
        "empty",
        "holes",
        "zero_capacity",
        "default_capacity",
        "two_groups"}) {
    int                n = name == "two_groups" ? 2 : 1, capacity = name == "zero_capacity" ? 0
                                                                    : name == "holes"       ? 1
                                                                    : name == "default_capacity" ? -1
                                                                                                 : 2;
    std::vector<float> x(n * 8 * 8), router(32);
    std::vector<int> valid(n * 8);
    for (int d = 0; d < 4; ++d)
      router[d * 4 + d] = 2;
    if (name == "tie")
      std::fill(router.begin(), router.end(), 0);
    for (int b = 0; b < n; ++b)
      for (int t = 0; t < 8; ++t) {
        valid[b * 8 + t] = name != "empty" && (name != "holes" || (t != 1 && t != 4 && t != 6));
        int expert       = name == "overflow" || (name == "two_groups" && b == 0) ? 0 : t % 4;
        x[(b * 8 + t) * 8 + expert] = 1;
      }
    auto input = nn::from_vector(x, {n, 8, 8}).requires_grad_(),
         mask = nn::from_vector(valid, {n, 8}), rw = nn::from_vector(router, {8, 4});
    p.set("switch/router/W", rw);
    auto path = switch_lm::Switch(p, input, nn::cast(mask, nn::DType::Bool), capacity);
    auto objective =
        nn::sum(nn::square(path.fields.at("output"))) + path.fields.at("auxiliary") * .01;
    auto leaves = p.parameters();
    leaves.push_back(input);
    auto grad                = nn::grad(objective, leaves);
    A    out                 = path.fields;
    out["objective"]         = objective;
    out["input"]             = input;
    out["valid"]             = mask;
    out["capacity_override"] = nn::scalar(capacity, nn::DType::Int32);
    out["router_override"]   = rw;
    int accepted             = 0;
    for (int e = 0; e < 4; ++e) {
      auto prefix                 = "expert" + std::to_string(e);
      out[prefix + "_indices"]    = path.indices[e];
      out[prefix + "_dispatched"] = path.dispatched[e];
      out[prefix + "_outputs"]    = path.expert_outputs[e];
      accepted += path.indices[e].numel();
    }
    for (size_t i = 0; i < p.size(); ++i)
      out[p.names()[i]] = grad[i];
    out["input_gradient"] = grad.back();
    Save(root / ("fixture_" + name + ".npz"), out);
    Compare(out, "fixture_" + name + ".npz", 2e-6);
    int expected = name == "empty" || name == "zero_capacity" ? 0
                   : name == "overflow" || name == "tie"      ? 2
                   : name == "holes"                          ? 4
                   : name == "two_groups"                     ? 10
                                                              : 8;
    check::Require(accepted == expected, "actual gathered expert rows/capacity");
    std::cout << "fixture=" << name << " actualExpertRows=" << accepted << '/' << n * 8
              << std::endl;
  }
  p.set("switch/router/W", original);
}
void Run(int epochs) {
  auto root = paths::OutputDir("models/switch_transformer");
  Fixtures(root);
  replay::CopyAssets(
      paths::Dataset("switch_transformer"), root, {"data.npz", "manifest.json", "vocabulary.txt"});
  auto  data = replay::ToMap(nn::load_npz((root / "data.npz").string()));
  Model model;
  check::Require(
      model.parameters.count() == 2689 && model.parameters.size() == 31,
      "2689 parameters31 tensors/four actual experts");
  auto tf = replay::TfModels("switch_transformer");
  if (std::filesystem::exists(tf / "initial_arrays.npz"))
    replay::LoadInto(model.parameters, tf / "initial_arrays.npz");
  Save(root / "initial_arrays.npz", model.Arrays());
  auto init = model.Run(data.at("test_input"), data.at("test_lengths"), &data.at("test_target"));
  Save(root / "initial_outputs.npz", init);
  Compare(init, "initial_outputs.npz");
  nn::BatchOrder order(128, 8, 47707);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history << "epoch\tvalidation_ce\tvalidation_auxiliary\taccepted\n";
  batches << "epoch\tstep\trow\tsource_start\n";
  int        updates = 0;
  nn::Tensor firstx, firsty;
  for (int e = 1; e <= epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      auto rows = order.next();
      if (rows.empty())
        break;
      auto x = nn::take_rows(data.at("fit_input"), rows),
           y = nn::take_rows(data.at("fit_target"), rows);
      if (!updates) {
        firstx = x;
        firsty = y;
        Save(root / "first_batch.npz", {{"input", x}, {"target", y}, {"lengths", Lens(8)}});
        auto g = model.Run(x, Lens(8), &y, true);
        Save(root / "initial_gradients.npz", g);
        Compare(g, "initial_gradients.npz");
      }
      model.Update(x, y, Lens(8));
      if (!updates)
        Save(root / "first_after.npz", model.Arrays());
      auto start = data.at("fit_start").to_vector<int>();
      for (auto row : rows)
        batches << e << '\t' << updates << '\t' << row << '\t' << start[row] << '\n';
      ++updates;
    }
    auto out = model.Run(
        data.at("validation_input"), data.at("validation_lengths"), &data.at("validation_target"));
    Save(root / ("epoch" + std::to_string(e) + "_validation.npz"), out);
    Save(root / ("epoch" + std::to_string(e) + "_arrays.npz"), model.Arrays());
    Compare(out, "epoch" + std::to_string(e) + "_validation.npz");
    double accepted = nn::sum(out.at("used")).item();
    history << e << '\t' << out.at("ce").item() << '\t' << out.at("auxiliary").item() << '\t'
            << accepted << '\n';
    std::cout << "epoch=" << e << " validationCE=" << out.at("ce").item()
              << " aux=" << out.at("auxiliary").item() << " accepted=" << accepted << "/512\n";
  }
  auto final = model.Arrays(), grad = model.Run(firstx, Lens(8), &firsty, true);
  Save(root / "model_arrays.npz", final);
  Save(root / "final_gradients.npz", grad);
  Compare(grad, "final_gradients.npz");
  model.parameters.save((root / "final.npz").string());
  Model deployed(false);
  deployed.parameters.load((root / "final.npz").string());
  check::Require(replay::SameBits(final, deployed.Arrays()), "all31Switch tensors bitexact");
  auto out = deployed.Run(data.at("test_input"), data.at("test_lengths"), &data.at("test_target"));
  Save(root / "outputs.npz", out);
  Compare(out, "outputs.npz");
  auto one=nn::take_rows(data.at("test_input"),{0});auto original=deployed.Run(one,Lens(1));
  auto changed = one.to_vector<int>();
  for (int t = 9; t < 16; ++t)
    changed[t] = (changed[t] + 17) % 65;
  auto future_input = nn::from_vector(changed, {1, 16});
  auto future       = deployed.Run(future_input, Lens(1));
  check::Near(
      replay::MaxAbsDiff(
          nn::slice(original.at("logits"), {0, 0, 0}, {1, 9, 65}),
          nn::slice(future.at("logits"), {0, 0, 0}, {1, 9, 65})),
      0,
      0,
      "fixed-capacity future cannot change prefix");
  check::Near(
      nn::max(nn::abs(nn::slice(original.at("prefix_gradient"), {0, 9, 0}, {1, 7, 65}))).item(),
      0,
      0,
      "causal prefix derivative0");
  Save(
      root / "causality.npz",
      {{"input", one},
       {"future_input", future_input},
       {"logits", original.at("logits")},
       {"future_logits", future.at("logits")},
       {"prefix_gradient", original.at("prefix_gradient")}});
  auto pad     = nn::take_rows(data.at("test_input"), {0, 1, 2}),
       pady    = nn::take_rows(data.at("test_target"), {0, 1, 2}),
       plen    = nn::from_vector(std::vector<int>{0, 7, 16}, {3});
  auto base    = deployed.Run(pad, plen, &pady);
  auto altered = pad.to_vector<int>(), lengths = plen.to_vector<int>();
  for (int b = 0; b < 3; ++b)
    for (int t = lengths[b]; t < 16; ++t)
      altered[b * 16 + t] = (altered[b * 16 + t] + 13) % 65;
  auto altered_input = nn::from_vector(altered, {3, 16});
  auto invariant     = deployed.Run(altered_input, plen, &pady);
  for (const auto& [key, value] : base)
    if (value.dtype() == nn::DType::Float32)
      check::Near(
          replay::MaxAbsDiff(value, invariant.at(key)),
          0,
          0,
          "Switch all fields/partials padding exclusion");
  Save(
      root / "padding_inputs.npz", {{"input", altered_input}, {"target", pady}, {"lengths", plen}});
  Save(root / "padding_outputs.npz", invariant);
  auto empty = deployed.Run(altered_input, Lens(3, 0), &pady);
  check::Near(empty.at("loss").item(), 0, 0, "all-empty Switch loss0");
  Save(root / "empty_outputs.npz", empty);
  for (int b = 0; b < 3; ++b) {
    auto single = nn::take_rows(data.at("test_input"), {b});
    auto z      = deployed.Run(single, Lens(1));
    check::Near(
        replay::MaxAbsDiff(z.at("logits"), nn::slice(out.at("logits"), {b, 0, 0}, {1, 16, 65})),
        0,
        2e-6,
        "independent routing groups under batching");
  }
  double fd    = 0;
  int    count = 0;
  for (auto name : model.parameters.names()) {
    auto value  = final.at(name);
    auto values = value.to_vector<float>();
    for (int i : {0, int(values.size() / 2), int(values.size() - 1)}) {
      float old = values[i];
      values[i] = old + .0001f;
      auto plus = model.Run(
          firstx, Lens(8), &firsty, false, {{name, nn::from_vector(values, value.shape())}});
      values[i]  = old - .0001f;
      auto minus = model.Run(
          firstx, Lens(8), &firsty, false, {{name, nn::from_vector(values, value.shape())}});
      values[i] = old;
      for (auto key : {"chosen", "accepted"})
        check::Require(
            replay::SameBits({{key, plus.at(key)}}, {{key, minus.at(key)}}) &&
                replay::SameBits({{key, plus.at(key)}}, {{key, grad.at(key)}}),
            "finite differences stay inside fixed routing branch");
      fd = std::max(
          fd,
          std::abs(
              (plus.at("loss").item() - minus.at("loss").item()) / .0002 -
              grad.at(name).to_vector<float>()[i]));
      ++count;
    }
  }
  check::Require(fd < .005, "piecewise native Switch finite differences");
  std::ifstream      vf(root / "vocabulary.txt");
  std::string        vocab((std::istreambuf_iterator<char>(vf)), {});
  std::vector<int>   codes(65), contexts(64 * 16), context_lengths(64);
  std::vector<float> scores(64 * 65);
  codes[0] = vocab.find('\n');
  std::ofstream sample(root / "sample.txt");
  sample << '\n';
  for (int t = 0; t < 64; ++t) {
    int              n = std::min(t + 1, 16), start = t + 1 - n;
    std::vector<int> context(16);
    std::copy_n(codes.begin() + start, n, context.begin());
    std::copy(context.begin(), context.end(), contexts.begin() + t * 16);
    context_lengths[t] = n;
    auto z =
        deployed.Run(nn::from_vector(context, {1, 16}), Lens(1, n)).at("logits").to_vector<float>();
    int best = 0;
    for (int c = 0; c < 65; ++c) {
      scores[t * 65 + c] = z[(n - 1) * 65 + c];
      if (scores[t * 65 + c] > scores[t * 65 + best])
        best = c;
    }
    codes[t + 1] = best;
    sample << vocab.at(best);
  }
  Save(
      root / "generation.npz",
      {{"codes", nn::from_vector(codes, {65})},
       {"logits", nn::from_vector(scores, {64, 65})},
       {"contexts", nn::from_vector(contexts, {64, 16})},
       {"lengths", nn::from_vector(context_lengths, {64})}});
  check::Require(replay::SameBits(final, deployed.Arrays()), "Switch inference immutable");
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"parameters\":2689,\"variables\":31,\"epochs\":" << epochs
          << ",\"updates\":" << updates << ",\"test_ce\":" << out.at("ce").item()
          << ",\"test_auxiliary\":" << out.at("auxiliary").item()
          << ",\"native_fd_count\":" << count << ",\"native_fd_error\":" << fd << "}\n";
  std::cout
      << "testCE=" << out.at("ce").item() << "; " << count << " FD max=" << fd
      << "; actualDispatch/capacity/balance/causal/padding/generation/restore/immutable PASS\n";
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    int epochs = 3;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      check::Require(a.rfind("epochs=", 0) == 0, "epochs=N");
      epochs = std::stoi(a.substr(7));
    }
    check::Require(epochs > 0 && epochs <= 100, "positive bounded epochs");
    Run(epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
