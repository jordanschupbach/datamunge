// Published Small BERT SavedModel: actual C++ serving and independent audit.
// CMD: just run-nn pretrained_bert_serve
// Native adaptation: import actual published weights exported as NPZ by the
// source example, reconstruct its encoder/pooler equations, and compare all
// five outputs. SavedModel/GraphDef loading is outside this plan's scope.
// Options: text=ASCII; arrays=PATH selects the exported44-tensor NPZ.
#include <chrono>

#include "common/pretrained_bert.hpp"
namespace nn = datamunge::nn;
namespace {
using A         = pretrained_bert::A;
using Model     = pretrained_bert::Model;
using Tokenizer = wordpiece::AsciiWordPiece;
void Save(const std::filesystem::path& file, const A& a) {
  nn::save_npz(file.string(), replay::ToNamed(a));
}
A Batch(const std::vector<Tokenizer::Packed>& rows) {
  check::Require(!rows.empty(), "nonempty WordPiece batch");
  std::vector<nn::Tensor> ids, mask, types;
  for (const auto& row : rows) {
    ids.push_back(row.ids);
    mask.push_back(row.mask);
    types.push_back(row.types);
  }
  return {
      {"input_word_ids", nn::concat(ids, 0)},
      {"input_mask", nn::concat(mask, 0)},
      {"input_type_ids", nn::concat(types, 0)}};
}
A Copy(const A& b) {
  A out;
  for (const auto& [k, v] : b)
    out[k] = v.clone();
  return out;
}
double Delta(const A& a, const A& b) {
  double e = 0;
  for (const auto& [k, v] : a)
    e = std::max(e, replay::MaxAbsDiff(v, b.at(k)));
  return e;
}
double ValidDelta(const A& a, const A& b, const nn::Tensor& mask) {
  double error = 0;
  auto   m     = mask.to_vector<int>();
  int    n = mask.size(0), t = mask.size(1);
  for (const auto& [key, v] : a) {
    auto w = b.at(key);
    auto x = v.to_vector<float>(), y = w.to_vector<float>();
    for (int row = 0; row < n; ++row)
      for (int j = 0; j < (v.dim() == 2 ? 1 : t); ++j)
        if (v.dim() == 2 || m[row * t + j])
          for (int d = 0; d < 128; ++d) {
            int i = v.dim() == 2 ? row * 128 + d : (row * t + j) * 128 + d;
            int k = v.dim() == 2 ? row * 128 + d : (row * w.size(1) + j) * 128 + d;
            error = std::max(error, double(std::abs(x[i] - y[k])));
          }
  }
  return error;
}
void Edit(A& b, const std::string& key, const std::function<void(std::vector<int>&)>& fn) {
  auto x = b.at(key).to_vector<int>();
  fn(x);
  b[key] = nn::from_vector(x, b.at(key).shape());
}
void Run(const std::filesystem::path& arrays, const std::string& request) {
  auto root = paths::OutputDir("models/pretrained_bert_serve"),
       tf   = replay::TfModels("pretrained_bert_serve");
  replay::CopyAssets(tf, root, {"vocab.txt", "source_manifest.json"});
  Tokenizer tokenizer(root / "vocab.txt");
  Model     model(arrays);
  auto      frozen = model.Arrays();
  Save(root / "pretrained_arrays.npz", frozen);
  check::Require(
      tokenizer.Tokenize("Hello TensorFlow!") == std::vector<int>({7592, 23435, 12314, 999}),
      "official WordPiece fixture");
  check::Require(
      tokenizer.Tokenize("HELLO tensorflow!") == tokenizer.Tokenize("Hello TensorFlow!"),
      "uncased tokenizer");
  check::Require(
      tokenizer.Tokenize(std::string(101, 'q')) == std::vector<int>({100}),
      "whole unknown token for >100ASCIIcharacters");
  check::Require(
      tokenizer.Tokenize("hello\tworld\n!") == std::vector<int>({7592, 2088, 999}),
      "BasicTokenizer whitespace");
  const std::vector<std::string> names{
      "hello", "hello_case", "cat", "cat_rephrase", "weather", "pair", "empty", "wordpieces"},
      first{
          "Hello TensorFlow!",
          "HELLO tensorflow!",
          "The cat sits on the mat.",
          "A cat is sitting on a mat.",
          "Heavy rain is expected tomorrow.",
          "The cat sits on the mat.",
          "",
          "unaffable tokenization."};
  std::vector<Tokenizer::Packed> packed;
  for (size_t i = 0; i < first.size(); ++i)
    packed.push_back(
        tokenizer.Pack(first[i], i == 5 ? "A cat is sitting on a mat." : "", i == 5, 32));
  auto batch = Batch(packed), out = model.Run(batch);
  Save(root / "inputs.npz", batch);
  Save(root / "outputs.npz", out);
  std::ofstream cases(root / "cases.tsv");
  cases << "row\tname\tfirst\tsecond\n";
  for (int i = 0; i < 8; ++i)
    cases << i << '\t' << names[i] << '\t' << first[i] << '\t'
          << (i == 5 ? "A cat is sitting on a mat." : "") << '\n';
  A fixtures;
  for (const auto& [name, p] : std::vector<std::pair<std::string, Tokenizer::Packed>>{
           {"pair", tokenizer.Pack("hello tensorflow", "goodbye tensorflow", true, 16)},
           {"truncate_single",
            tokenizer.Pack("hello hello hello hello hello hello hello hello", {}, false, 8)},
           {"truncate_pair",
            tokenizer.Pack(
                "hello hello hello hello hello",
                "goodbye goodbye goodbye goodbye goodbye",
                true,
                8)},
           {"unknown", tokenizer.Pack(std::string(101, 'q'), {}, false, 8)},
           {"empty", tokenizer.Pack("", {}, false, 8)},
           {"punctuation", tokenizer.Pack("Can't, stop.", {}, false, 16)}}) {
    fixtures[name + "_ids"]   = p.ids;
    fixtures[name + "_mask"]  = p.mask;
    fixtures[name + "_types"] = p.types;
  }
  Save(root / "tokenizer_fixtures.npz", fixtures);
  // Integer tokenizer fixtures are compared separately, without float casting.
  auto expected_fixtures = replay::ToMap(nn::load_npz((tf / "tokenizer_fixtures.npz").string()));
  check::Require(replay::SameBits(fixtures, expected_fixtures), "source tokenizer packing exact");
  double batch_error = 0;
  for (int row = 0; row < 8; ++row)
    batch_error = std::max(
        batch_error, Delta(replay::Rows(out, {row}), model.Run(replay::Rows(batch, {row}))));
  check::Require(batch_error < 2e-5, "batch/single serving parity");
  check::Near(Delta(out, model.Run(batch)), 0, 0, "repeated serving deterministic");
  auto padded = Copy(batch);
  auto mask   = batch.at("input_mask").to_vector<int>();
  for (auto key : {"input_word_ids", "input_type_ids"})
    Edit(padded, key, [&](auto& x) {
      for (int i = 0; i < 256; ++i)
        if (!mask[i])
          x[i] = std::string(key) == "input_word_ids" ? 1000 + i % 100 : 1;
    });
  auto   padout        = model.Run(padded);
  double mask_error    = ValidDelta(out, padout, batch.at("input_mask")),
         masked_effect = Delta(out, padout);
  check::Require(mask_error < 2e-5, "masked keys cannot affect visible states/pooled output");
  check::Require(masked_effect > .01, "masked QUERY states not forced zero");
  Save(root / "masked_inputs.npz", padded);
  Save(root / "masked_outputs.npz", padout);
  auto changed = Copy(batch);
  Edit(changed, "input_word_ids", [&](auto& x) { x[2 * 32 + 2] = tokenizer.lookup.at("dog"); });
  auto   changedout   = model.Run(changed);
  double token_effect = Delta(out, changedout);
  check::Require(token_effect > .01, "visible word changes pretrained outputs");
  Save(root / "changed_inputs.npz", changed);
  Save(root / "changed_outputs.npz", changedout);
  auto types              = Copy(batch);
  types["input_type_ids"] = nn::zeros({8, 32}, nn::DType::Int32);
  auto   typeout          = model.Run(types);
  double type_effect      = Delta(out, typeout);
  check::Require(type_effect > .01, "sentence-pair type embeddings matter");
  Save(root / "types_inputs.npz", types);
  Save(root / "types_outputs.npz", typeout);
  auto hole = Copy(batch);
  Edit(hole, "input_mask", [](auto& x) { x[2 * 32 + 2] = 0; });
  auto   holeout     = model.Run(hole);
  double hole_effect = Delta(out, holeout);
  check::Require(hole_effect > .01, "removing visible attention key changes context");
  Save(root / "hole_inputs.npz", hole);
  Save(root / "hole_outputs.npz", holeout);
  auto short_batch   = Batch({tokenizer.Pack("Hello TensorFlow!", {}, false, 8)}),
       short_out     = model.Run(short_batch);
  double width_error = ValidDelta(short_out, replay::Rows(out, {0}), short_batch.at("input_mask"));
  check::Require(width_error < 2e-5, "dynamic width preserves valid states");
  Save(root / "short_inputs.npz", short_batch);
  Save(root / "short_outputs.npz", short_out);
  int  guards = 0;
  auto reject = [&](auto fn) {
    bool rejected = false;
    try {
      fn();
    } catch (const std::exception&) {
      rejected = true;
    }
    check::Require(rejected, "invalid serving request rejected");
    ++guards;
  };
  reject([&] { tokenizer.Tokenize("caf\xc3\xa9"); });
  reject([&] { tokenizer.Pack("a", "b", true, 2); });
  reject([&] { tokenizer.Pack("a", {}, false, 513); });
  for (auto field : {"input_word_ids", "input_mask", "input_type_ids"}) {
    auto bad = Copy(batch);
    Edit(bad, field, [&](auto& x) { x[0] = std::string(field) == "input_word_ids" ? 30522 : 2; });
    reject([&] { model.Run(bad); });
  }
  auto bad              = Copy(batch);
  bad["input_word_ids"] = nn::zeros({8, 32});
  reject([&] { model.Run(bad); });
  bad                   = Copy(batch);
  bad["input_type_ids"] = nn::zeros({8, 31}, nn::DType::Int32);
  reject([&] { model.Run(bad); });
  bad               = Copy(batch);
  bad["input_mask"] = nn::zeros({8, 32}, nn::DType::Int32);
  reject([&] { model.Run(bad); });
  bad = Copy(batch);
  bad.erase("input_mask");
  reject([&] { model.Run(bad); });
  Model reloaded(arrays);
  check::Near(Delta(out, reloaded.Run(batch)), 0, 0, "fresh NPZ reload exact");
  check::Require(replay::SameBits(frozen, reloaded.Arrays()), "fresh NPZ weights identical");
  double tf_error = 0;
  for (auto prefix : {"", "masked_", "changed_", "types_", "hole_", "short_"}) {
    auto inputs =
        replay::ToMap(nn::load_npz((root / (std::string(prefix) + "inputs.npz")).string()));
    auto result = model.Run(inputs);
    auto expected =
        replay::ToMap(nn::load_npz((tf / (std::string(prefix) + "outputs.npz")).string()));
    double e = Delta(result, expected);
    tf_error = std::max(tf_error, e);
    check::Near(e, 0, 2e-5, "published TF outputs " + std::string(prefix));
  }
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; ++i)
    model.Run(batch);
  double milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
      10;
  if (!request.empty()) {
    auto input = Batch({tokenizer.Pack(request)}), result = model.Run(input);
    Save(root / "request_inputs.npz", input);
    Save(root / "request_outputs.npz", result);
    std::ofstream(root / "request.txt") << request;
    std::cout << "request pooled first4:";
    auto pooled = result.at("bert_encoder").to_vector<float>();
    for (int i = 0; i < 4; ++i)
      std::cout << ' ' << pooled[i];
    std::cout << '\n';
  }
  check::Require(replay::SameBits(frozen, model.Arrays()), "all44pretrained variables immutable");
  std::ofstream metrics(root / "metrics.json");
  metrics.precision(17);
  metrics << "{\"stored_variables\":44,\"stored_parameters\":4433210,\"encoder_pooler_parameters\":"
             "4385920,\"unused_mlm_parameters\":47290,\"requests_in_batch\":8,\"signature_"
             "outputs\":5,\"guard_checks\":"
          << guards << ",\"batch_error\":" << batch_error
          << ",\"masked_valid_error\":" << mask_error
          << ",\"masked_query_effect\":" << masked_effect
          << ",\"visible_token_effect\":" << token_effect << ",\"type_effect\":" << type_effect
          << ",\"hole_mask_effect\":" << hole_effect << ",\"dynamic_width_error\":" << width_error
          << ",\"reload_error\":0,\"tf_output_error\":" << tf_error
          << ",\"warm_batch8_milliseconds\":" << milliseconds << "}\n";
  std::cout << "Published Small BERT NPZ:44vars/4433210params;8requests/10guards; TF max="
            << tf_error << " batch=" << batch_error << " maskedValid=" << mask_error
            << " dynamicWidth=" << width_error << "; immutable/reload/tokenization/serving PASS\n";
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    auto        arrays = replay::TfModels("pretrained_bert_serve") / "pretrained_arrays.npz";
    std::string text   = "Hello TensorFlow!";
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("text=", 0) == 0)
        text = a.substr(5);
      else if (a.rfind("arrays=", 0) == 0)
        arrays = a.substr(7);
      else
        throw std::invalid_argument("text=ASCII arrays=PATH");
    }
    Run(arrays, text);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
