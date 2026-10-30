#pragma once
/// @file id3_diagnostics.hpp
/// @brief Categorical ID3 split accounting, traversal diagnostics and frozen DataFrame encoding.
#include <datamunge/algorithms/decision_tree_induction.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
namespace datamunge::algorithms {
/// @brief Owned integer-coded categorical teaching data and predictor names.
/// @par Example
/// @code
/// const auto data=dm::algorithms::id3_tennis();
/// std::print("{}\n",data.labels.size()); // 14
/// @endcode
/// @see id3_tennis(), id3_train()
struct ID3Data {
  std::vector<std::vector<int>> features;      ///< Rows of categorical codes.
  std::vector<int>              labels;        ///< One integer class label per row.
  std::vector<std::string>      feature_names; ///< Display names in column order.
};
/// @brief Fourteen-row Play Tennis teaching fixture; a constructed example, not a survey.
/// Codes: outlook sunny/overcast/rain=0/1/2; temperature hot/mild/cool=0/1/2;
/// humidity high/normal=0/1; windy false/true=0/1; play no/yes=0/1.
/// @return Independent owned data with predictors outlook, temperature, humidity, windy.
/// @par Example
/// @code
/// const auto d=dm::algorithms::id3_tennis();
/// const auto t=dm::algorithms::id3_train(d.features,d.labels);
/// std::print("{}\n",t.nodes[0].feature); // 0
/// @endcode
/// @see ID3Data, id3_split_table()
ID3Data id3_tennis();
/// @brief Empirical entropy and conditional entropy for every candidate root feature.
/// Uses precisely the categorical partition/bit-entropy criterion of id3_train().
/// Zero-count classes are omitted. Names default to f0, f1, ... . Owns its output,
/// borrows inputs without mutation; O(n*d*log(n)) work and O(n+d) auxiliary storage.
/// @param features Rectangular integer-coded n-by-d matrix; empty input gives no rows.
/// @param labels Exactly n class labels.
/// @param names Empty (default) or exactly d predictor names.
/// @return feature/name/branches/entropy/conditional_entropy/gain in predictor order.
/// @throws std::invalid_argument Ragged matrix, label mismatch or wrong name count.
/// @par Example
/// @code
/// const auto s=dm::algorithms::id3_split_table({{0},{1}}, {0,1});
/// std::print("{:.0f}\n",s.double_at("gain",0)); // 1
/// @endcode
/// @par References
/// - J. R. Quinlan (1986), Induction of decision trees. https://doi.org/10.1007/BF00116251
/// @see id3_train(), id3_branch_table()
dstruct::DataFrame id3_split_table(
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels,
    const std::vector<std::string>&      names = {});
/// @brief Weighted branch entropies for one categorical test, including majority labels.
/// Owns results and does not mutate inputs. Work O(n*log(n)), storage O(n).
/// @param features Nonempty rectangular integer-coded design.
/// @param labels Exactly one label per row.
/// @param feature Valid column index.
/// @return value/rows/weight/entropy/weighted_entropy/majority in increasing category order.
/// @throws std::invalid_argument Empty/ragged input, mismatched labels or invalid feature.
/// @par Example
/// @code
/// const auto b=dm::algorithms::id3_branch_table({{0},{0},{1}}, {0,1,1},0);
/// std::print("{:.0f}\n",b.double_at("rows",0)); // 2
/// @endcode
/// @par References
/// - Quinlan (1986). https://doi.org/10.1007/BF00116251
/// @see id3_split_table()
dstruct::DataFrame id3_branch_table(
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels,
    std::size_t                          feature);
/// @brief Inspect every node of a valid categorical tree, with depth and fallback label.
/// Validates that all nodes are reachable once from root zero, with no cycles/shared
/// children and no children on leaves. Empty tree gives a typed zero-row table.
/// O(nodes+edges) work/storage; inputs are unchanged.
/// @param tree Categorical tree with valid child indices.
/// @return node/depth/leaf/feature/prediction/branches; feature=-1 for leaves.
/// @throws std::invalid_argument Malformed categorical node topology.
/// @par Example
/// @code
/// const auto t=dm::algorithms::id3_train({{0},{1}}, {0,1});
/// std::print("{}\n",dm::algorithms::id3_node_table(t).nrows()); // 3
/// @endcode
/// @see id3_trace(), plot_id3_tree()
dstruct::DataFrame id3_node_table(const DecisionTree& tree);
/// @brief Explain a categorical prediction through each visited test and final stop.
/// Same visited-path checks as id3_predict(); no mutation/shared state. O(h*log(b))
/// work and O(h) result storage. Unseen category stops at the current majority.
/// @param tree Nonempty categorical tree.
/// @param x Query with all visited features present; extra values are ignored.
/// @return node/feature/value/prediction/leaf/unseen; leaf feature/value use -1 sentinels.
/// @throws std::invalid_argument Empty tree, short query, invalid visited index or cycle.
/// @par Example
/// @code
/// const auto t=dm::algorithms::id3_train({{0},{0},{1}}, {2,2,7});
/// std::print("{:.0f}\n",dm::algorithms::id3_trace(t,{9}).double_at("unseen",0)); // 1
/// @endcode
/// @see id3_predict(), id3_evaluate()
dstruct::DataFrame id3_trace(const DecisionTree& tree, const std::vector<int>& x);
/// @brief Accuracy and unseen-branch counts against supplied categorical labels.
/// Baseline always predicts the fitted root majority, never a majority fitted to
/// evaluation labels. A fallback is counted once per query. No mutation/shared state.
/// Work O(n*h*log(b)); temporary storage O(h), plus the one-row output.
/// @param tree Nonempty categorical tree.
/// @param features Nonempty rectangular evaluation design.
/// @param labels One class label per evaluation row.
/// @return rows/correct/accuracy/fallbacks/baseline_accuracy, unweighted over rows.
/// @throws std::invalid_argument Empty/ragged/misaligned inputs or invalid traversed tree/query.
/// @par Example
/// @code
/// const auto t=dm::algorithms::id3_train({{0},{1}}, {0,1});
/// const auto e=dm::algorithms::id3_evaluate(t,{{0},{1}}, {0,1});
/// std::print("{:.0f}\n",e.double_at("accuracy",0)); // 1
/// @endcode
/// @see id3_trace(), ID3FrameClassifier
dstruct::DataFrame id3_evaluate(
    const DecisionTree&                  tree,
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels);
/// @brief Draw a small categorical tree, labelling tests, category edges and leaf classes.
/// Places leaves uniformly and internal nodes at the mean child position. Intended
/// for teaching trees with at most 31 nodes and eight levels; returns an editable
/// plot, never inline SVG bytes. O(nodes+edges) work/storage, no input mutation.
/// @param tree Nonempty valid categorical tree within the display limits.
/// @param names Empty (default, f0/f1/...) or enough names for every tested feature.
/// @return Owned plot with y positions equal to negative depth.
/// @throws std::invalid_argument Malformed/empty/oversized tree or too few names.
/// @par Example
/// @code
/// const auto t=dm::algorithms::id3_train({{0},{1}}, {0,1});
/// dm::algorithms::plot_id3_tree(t,{"category"}).save_svg("id3.svg");
/// @endcode
/// @see id3_node_table()
graphics::RPlot plot_id3_tree(const DecisionTree& tree, const std::vector<std::string>& names = {});
/// @brief ID3 with frozen categorical encoding of named numeric or string DataFrame columns.
/// All predictor values are categories, including numeric columns: no thresholds,
/// scaling or ordering assumptions. Dictionaries are learned only on training rows,
/// assigning nonnegative codes in first-occurrence order. New levels map to -1 and
/// use the native local-majority fallback. Nulls and nonfinite numeric predictors
/// are rejected, rather than silently omitted or imputed. Labels are numeric ints.
/// Models own their tree/schema/dictionaries. Const inference has no shared mutation.
/// @par Example
/// @code
/// const auto f=dm::datasets::titanic().select({"Sex","Pclass","Survived"});
/// const dm::algorithms::ID3FrameClassifier m(f,{"Sex","Pclass"},"Survived",2);
/// std::print("{}\n",m.predict(f).size()); // 891
/// @endcode
/// @par References
/// - Quinlan (1986). https://doi.org/10.1007/BF00116251
/// @see id3_train(), id3_evaluate()
class ID3FrameClassifier {
 public:
  /// @brief Fit a categorical tree and training-only level dictionaries.
  /// Work O(n*d*h*log(n)); dictionaries add O(n*d) worst-case storage.
  /// @param frame Nonempty complete training rows; extra columns ignored.
  /// @param predictors Nonempty distinct existing column names, excluding response.
  /// @param response Existing numeric column of finite integer labels in int range.
  /// @param max_depth Zero (default) for unlimited, otherwise maximum split depth.
  /// @throws std::invalid_argument Invalid schema, missing/nonfinite data or noninteger labels.
  /// @par Example
  /// @code
  /// const auto f=dm::datasets::titanic();
  /// const dm::algorithms::ID3FrameClassifier m(f,{"Sex"},"Survived",1);
  /// std::print("{}\n",m.tree().nodes.size()); // 3
  /// @endcode
  /// @see ID3FrameClassifier, encode()
  ID3FrameClassifier(
      const dstruct::DataFrame&       frame,
      const std::vector<std::string>& predictors,
      const std::string&              response,
      std::size_t                     max_depth = 0);
  /// @brief Read the fitted tree without copying; reference lives with this model.
  /// @return Borrowed immutable categorical tree.
  const DecisionTree& tree() const { return tree_; }
  /// @brief Encode a new frame using only fitted dictionaries; unseen levels become -1.
  /// Requires the same predictor names/types; extra columns ignored. Empty frame
  /// with matching schema returns no rows. O(n*d*log(L)) work, O(n*d) output.
  /// @param frame Complete finite predictors with fitted column types.
  /// @return Owned integer design in fitted predictor order.
  /// @throws std::invalid_argument Missing/wrong-type columns or invalid predictor cells.
  /// @par Example
  /// @code
  /// const auto f=dm::datasets::titanic();
  /// const dm::algorithms::ID3FrameClassifier m(f,{"Sex"},"Survived");
  /// std::print("{}\n",m.encode(f.head(2)).size()); // 2
  /// @endcode
  /// @see predict()
  std::vector<std::vector<int>> encode(const dstruct::DataFrame& frame) const;
  /// @brief Encode complete rows and predict, using local majority for new categories.
  /// Work is encoding plus native tree traversal per row. No model mutation.
  /// @param frame Frame satisfying encode()'s contract; empty rows are allowed.
  /// @return One owned integer prediction per input row, in input order.
  /// @throws std::invalid_argument Invalid query schema or predictor cells.
  /// @par Example
  /// @code
  /// const auto f=dm::datasets::titanic();
  /// const dm::algorithms::ID3FrameClassifier m(f,{"Sex"},"Survived");
  /// std::print("{}\n",m.predict(f.head(2)).size()); // 2
  /// @endcode
  /// @see encode(), id3_predict()
  std::vector<int> predict(const dstruct::DataFrame& frame) const;
  /// @brief Evaluate labelled complete rows with the fitted baseline and unseen counts.
  /// Response name is the one fitted at construction; encoding never learns levels.
  /// @param frame Nonempty frame with fitted predictor schema and valid integer response.
  /// @return One-row table from id3_evaluate().
  /// @throws std::invalid_argument Invalid predictor/response schema or cells, or no rows.
  /// @par Example
  /// @code
  /// const auto f=dm::datasets::titanic();
  /// const dm::algorithms::ID3FrameClassifier m(f,{"Sex"},"Survived");
  /// std::print("{:.0f}\n",m.evaluate(f).double_at("rows",0)); // 891
  /// @endcode
  /// @see id3_evaluate()
  dstruct::DataFrame evaluate(const dstruct::DataFrame& frame) const;

 private:
  DecisionTree                                tree_;     ///< Owned fitted native tree.
  std::vector<std::string>                    names_;    ///< Ordered fitted predictor names.
  std::string                                 response_; ///< Numeric response column name.
  std::vector<dstruct::DataFrame::ColumnType> types_;    ///< Fitted predictor types.
  std::vector<std::map<dstruct::DataFrame::cell_type, int>> levels_; ///< Training-only level codes.
};
} // namespace datamunge::algorithms
