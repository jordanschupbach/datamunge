/// @file id3_diagnostics.cpp
/// @brief Split accounting, safe categorical traversal and frozen frame dictionaries.
#include <datamunge/algorithms/id3_diagnostics.hpp>
#include <functional>
#include <limits>
#include <numeric>
namespace datamunge::algorithms {
/// @brief Internal input checks and categorical topology walks.
namespace {
/// @brief Check supervised categorical shapes and make an all-row selection.
/// @param x Rectangular feature rows.
/// @param y Matching labels.
/// @return Consecutive row indices, empty when inputs are empty.
/// @throws std::invalid_argument Misaligned labels or ragged rows.
std::vector<std::size_t> rows_for(
    const std::vector<std::vector<int>>& x,
    const std::vector<int>&              y) {
  if (x.size() != y.size())
    throw std::invalid_argument("id3: label count mismatch");
  const auto d = x.empty() ? 0 : x[0].size();
  for (const auto& row : x)
    if (row.size() != d)
      throw std::invalid_argument("id3: ragged matrix");
  std::vector<std::size_t> rows(x.size());
  std::iota(rows.begin(), rows.end(), 0);
  return rows;
}
/// @brief Validate a categorical tree and determine root-first node depths iteratively.
/// @param tree Categorical topology; empty is permitted.
/// @return Depth indexed by native node index.
/// @throws std::invalid_argument Invalid/shared/cyclic/unreachable nodes or leaf children.
std::vector<double> depths(const DecisionTree& tree) {
  std::vector<double> depth(tree.nodes.size(), -1);
  if (tree.nodes.empty())
    return depth;
  depth[0] = 0;
  std::vector<std::size_t> stack{0};
  while (!stack.empty()) {
    const auto i = stack.back();
    stack.pop_back();
    const auto& n = tree.nodes[i];
    if ((n.leaf && !n.children.empty()) || (!n.leaf && n.children.empty()) || n.left != -1 ||
        n.right != -1)
      throw std::invalid_argument("id3: invalid categorical topology");
    for (const auto& [v, c] : n.children) {
      if (c < 0 || static_cast<std::size_t>(c) >= depth.size() || depth[c] >= 0)
        throw std::invalid_argument("id3: invalid or repeated child");
      depth[c] = depth[i] + 1;
      stack.push_back(c);
    }
  }
  if (std::find(depth.begin(), depth.end(), -1) != depth.end())
    throw std::invalid_argument("id3: unreachable node");
  return depth;
}
/// @brief Read finite numeric integer labels from a named response column.
/// @param frame Source frame, possibly empty.
/// @param name Numeric response column.
/// @return Owned integer class labels.
/// @throws std::invalid_argument Missing/wrong-type response or invalid cell.
std::vector<int> frame_labels(const dstruct::DataFrame& frame, const std::string& name) {
  using DF = dstruct::DataFrame;
  if (!frame.has_column(name) || frame.column_type(name) != DF::ColumnType::Numeric)
    throw std::invalid_argument("id3: numeric response required");
  std::vector<int> y;
  for (std::size_t i = 0; i < frame.nrows(); ++i) {
    if (frame.is_null(name, i))
      throw std::invalid_argument("id3: missing response");
    const double v = frame.double_at(name, i);
    if (!std::isfinite(v) || v != std::trunc(v) || v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
      throw std::invalid_argument("id3: response must fit int");
    y.push_back(static_cast<int>(v));
  }
  return y;
}
/// @brief Read one valid categorical cell, retaining its original storage type.
/// @param frame Source frame.
/// @param name Existing predictor name.
/// @param row Valid row index.
/// @return String or finite double variant; null is rejected.
/// @throws std::invalid_argument Null or nonfinite numeric value.
dstruct::DataFrame::cell_type
category(const dstruct::DataFrame& frame, const std::string& name, std::size_t row) {
  const auto v = frame.value(name, row);
  if (std::holds_alternative<dstruct::DataFrame::null_type>(v) ||
      (std::holds_alternative<double>(v) && !std::isfinite(std::get<double>(v))))
    throw std::invalid_argument("id3: complete finite categorical predictors required");
  return v;
}
} // namespace
ID3Data id3_tennis() {
  return {
      {{0, 0, 0, 0},
       {0, 0, 0, 1},
       {1, 0, 0, 0},
       {2, 1, 0, 0},
       {2, 2, 1, 0},
       {2, 2, 1, 1},
       {1, 2, 1, 1},
       {0, 1, 0, 0},
       {0, 2, 1, 0},
       {2, 1, 1, 0},
       {0, 1, 1, 1},
       {1, 1, 0, 1},
       {1, 0, 1, 0},
       {2, 1, 0, 1}},
      {0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0},
      {"outlook", "temperature", "humidity", "windy"}};
}
dstruct::DataFrame id3_split_table(
    const std::vector<std::vector<int>>& x,
    const std::vector<int>&              y,
    const std::vector<std::string>&      names) {
  const auto rows = rows_for(x, y);
  const auto d    = x.empty() ? 0 : x[0].size();
  if (!names.empty() && names.size() != d)
    throw std::invalid_argument("id3: wrong names count");
  const double             base = detail::class_entropy(y, rows);
  std::vector<double>      feature, branches, entropy, conditional, gain;
  std::vector<std::string> name;
  for (std::size_t f = 0; f < d; ++f) {
    std::map<int, std::vector<std::size_t>> parts;
    for (auto i : rows)
      parts[x[i][f]].push_back(i);
    double h = 0;
    for (const auto& [v, sub] : parts)
      h += double(sub.size()) / rows.size() * detail::class_entropy(y, sub);
    feature.push_back(f);
    branches.push_back(parts.size());
    entropy.push_back(base);
    conditional.push_back(h);
    gain.push_back(base - h);
    name.push_back(names.empty() ? "f" + std::to_string(f) : names[f]);
  }
  dstruct::DataFrame out;
  out.add_column("feature", feature);
  out.add_column("name", name);
  out.add_column("branches", branches);
  out.add_column("entropy", entropy);
  out.add_column("conditional_entropy", conditional);
  out.add_column("gain", gain);
  return out;
}
dstruct::DataFrame id3_branch_table(
    const std::vector<std::vector<int>>& x,
    const std::vector<int>&              y,
    std::size_t                          feature) {
  const auto rows = rows_for(x, y);
  if (x.empty() || feature >= x[0].size())
    throw std::invalid_argument("id3: invalid branch feature");
  std::map<int, std::vector<std::size_t>> parts;
  for (auto i : rows)
    parts[x[i][feature]].push_back(i);
  std::vector<double> value, count, weight, entropy, weighted, majority;
  for (const auto& [v, sub] : parts) {
    const double w = double(sub.size()) / rows.size(), h = detail::class_entropy(y, sub);
    value.push_back(v);
    count.push_back(sub.size());
    weight.push_back(w);
    entropy.push_back(h);
    weighted.push_back(w * h);
    majority.push_back(detail::majority_class(y, sub));
  }
  dstruct::DataFrame out;
  out.add_column("value", value);
  out.add_column("rows", count);
  out.add_column("weight", weight);
  out.add_column("entropy", entropy);
  out.add_column("weighted_entropy", weighted);
  out.add_column("majority", majority);
  return out;
}
dstruct::DataFrame id3_node_table(const DecisionTree& tree) {
  const auto          depth = depths(tree);
  std::vector<double> index, leaf, feature, prediction, branches;
  for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
    const auto& n = tree.nodes[i];
    index.push_back(i);
    leaf.push_back(n.leaf);
    feature.push_back(n.leaf ? -1 : double(n.feature));
    prediction.push_back(n.prediction);
    branches.push_back(n.children.size());
  }
  dstruct::DataFrame out;
  out.add_column("node", index);
  out.add_column("depth", depth);
  out.add_column("leaf", leaf);
  out.add_column("feature", feature);
  out.add_column("prediction", prediction);
  out.add_column("branches", branches);
  return out;
}
dstruct::DataFrame id3_trace(const DecisionTree& tree, const std::vector<int>& x) {
  (void)id3_predict(tree, x); // central visited-path contract, including cycle checks
  std::vector<double> node, feature, value, prediction, leaf, unseen;
  int                 i = 0;
  for (;;) {
    const auto& n       = tree.nodes[i];
    const auto  it      = n.leaf ? n.children.end() : n.children.find(x[n.feature]);
    const bool  missing = !n.leaf && it == n.children.end();
    node.push_back(i);
    feature.push_back(n.leaf ? -1 : double(n.feature));
    value.push_back(n.leaf ? -1 : x[n.feature]);
    prediction.push_back(n.prediction);
    leaf.push_back(n.leaf);
    unseen.push_back(missing);
    if (n.leaf || missing)
      break;
    i = it->second;
  }
  dstruct::DataFrame out;
  out.add_column("node", node);
  out.add_column("feature", feature);
  out.add_column("value", value);
  out.add_column("prediction", prediction);
  out.add_column("leaf", leaf);
  out.add_column("unseen", unseen);
  return out;
}
dstruct::DataFrame id3_evaluate(
    const DecisionTree&                  tree,
    const std::vector<std::vector<int>>& x,
    const std::vector<int>&              y) {
  const auto rows = rows_for(x, y);
  if (rows.empty() || tree.nodes.empty())
    throw std::invalid_argument("id3: evaluation needs rows and a tree");
  std::size_t correct = 0, base = 0, fallback = 0;
  for (auto i : rows) {
    const auto trace = id3_trace(tree, x[i]);
    const auto last  = trace.nrows() - 1;
    correct += trace.double_at("prediction", last) == y[i];
    fallback += trace.double_at("unseen", last) != 0;
    base += tree.nodes[0].prediction == y[i];
  }
  dstruct::DataFrame out;
  out.add_column("rows", std::vector<double>{double(rows.size())});
  out.add_column("correct", std::vector<double>{double(correct)});
  out.add_column("accuracy", std::vector<double>{double(correct) / rows.size()});
  out.add_column("fallbacks", std::vector<double>{double(fallback)});
  out.add_column("baseline_accuracy", std::vector<double>{double(base) / rows.size()});
  return out;
}
graphics::RPlot plot_id3_tree(const DecisionTree& tree, const std::vector<std::string>& names) {
  const auto depth = depths(tree);
  if (depth.empty() || depth.size() > 31 || *std::max_element(depth.begin(), depth.end()) > 7)
    throw std::invalid_argument("id3: plot needs 1..31 nodes and at most eight levels");
  for (const auto& n : tree.nodes)
    if (!n.leaf && !names.empty() && n.feature >= names.size())
      throw std::invalid_argument("id3: too few plot names");
  std::vector<double> xpos(depth.size());
  double              next = 0;
  // Postorder placement uses leaf order; topology checks bound recursion to eight levels.
  std::function<void(std::size_t)> place = [&](std::size_t i) {
    if (tree.nodes[i].leaf) {
      xpos[i] = next++;
      return;
    }
    for (const auto& [v, c] : tree.nodes[i].children) {
      place(c);
      xpos[i] += xpos[c];
    }
    xpos[i] /= tree.nodes[i].children.size();
  };
  place(0);
  auto p = graphics::RPlot::create();
  for (std::size_t i = 0; i < depth.size(); ++i) {
    const auto& n = tree.nodes[i];
    for (const auto& [v, c] : n.children) {
      p.line({xpos[i], xpos[c]}, {-depth[i] - .13, -depth[c] + .13}, "", {140, 150, 165});
      p.text(
          (xpos[i] + xpos[c]) / 2,
          (-depth[i] - depth[c]) / 2 + .08,
          std::to_string(v),
          {80, 80, 80},
          15);
    }
    const std::string label =
        n.leaf ? "class " + std::to_string(n.prediction)
               : (names.empty() ? "f" + std::to_string(n.feature) : names[n.feature]);
    p.text(
        xpos[i],
        -depth[i],
        label,
        n.leaf ? graphics::RGB{25, 125, 65} : graphics::RGB{30, 70, 160},
        17);
  }
  return p.title("Categorical decision tree")
      .x_label("leaf order")
      .y_label("negative depth")
      .x_limits(-.6, std::max(.6, next - .4))
      .y_limits(-*std::max_element(depth.begin(), depth.end()) - .45, .45);
}
ID3FrameClassifier::ID3FrameClassifier(
    const dstruct::DataFrame&       f,
    const std::vector<std::string>& predictors,
    const std::string&              response,
    std::size_t                     max_depth)
    : names_(predictors), response_(response) {
  if (f.nrows() == 0 || predictors.empty() ||
      std::set<std::string>(predictors.begin(), predictors.end()).size() != predictors.size())
    throw std::invalid_argument("id3: nonempty training rows and distinct predictors required");
  const auto y = frame_labels(f, response);
  levels_.resize(names_.size());
  for (std::size_t j = 0; j < names_.size(); ++j) {
    const auto& name = names_[j];
    if (name == response || !f.has_column(name))
      throw std::invalid_argument("id3: invalid predictor name");
    types_.push_back(f.column_type(name));
    for (std::size_t i = 0; i < f.nrows(); ++i) {
      const auto v = category(f, name, i);
      if (!levels_[j].contains(v)) {
        if (levels_[j].size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::invalid_argument("id3: too many levels");
        const auto code = static_cast<int>(levels_[j].size());
        levels_[j].emplace(v, code);
      }
    }
  }
  tree_ = id3_train(encode(f), y, max_depth);
}
std::vector<std::vector<int>> ID3FrameClassifier::encode(const dstruct::DataFrame& f) const {
  for (std::size_t j = 0; j < names_.size(); ++j)
    if (!f.has_column(names_[j]) || f.column_type(names_[j]) != types_[j])
      throw std::invalid_argument("id3: predictor schema mismatch");
  std::vector<std::vector<int>> out(f.nrows(), std::vector<int>(names_.size()));
  for (std::size_t i = 0; i < f.nrows(); ++i)
    for (std::size_t j = 0; j < names_.size(); ++j) {
      const auto it = levels_[j].find(category(f, names_[j], i));
      out[i][j]     = it == levels_[j].end() ? -1 : it->second;
    }
  return out;
}
std::vector<int> ID3FrameClassifier::predict(const dstruct::DataFrame& f) const {
  const auto       x = encode(f);
  std::vector<int> out;
  out.reserve(x.size());
  for (const auto& row : x)
    out.push_back(id3_predict(tree_, row));
  return out;
}
dstruct::DataFrame ID3FrameClassifier::evaluate(const dstruct::DataFrame& f) const {
  return id3_evaluate(tree_, encode(f), frame_labels(f, response_));
}
} // namespace datamunge::algorithms
