classdef DecisionTreeClassifier < datamunge.NativeObject
  % DecisionTreeClassifier  SWIG-friendly facade for datamunge::stats::DecisionTreeClassifier — a CART-style classification tree.
  %
  %   obj = datamunge.DecisionTreeClassifier (data, formula)
  %   obj = datamunge.DecisionTreeClassifier (data, formula, max_depth)
  %   obj = datamunge.DecisionTreeClassifier (data, formula, max_depth, min_samples_split)
  %   obj = datamunge.DecisionTreeClassifier (data, formula, max_depth, min_samples_split, min_samples_leaf)
  %   obj = datamunge.DecisionTreeClassifier (data, formula, max_depth, min_samples_split, min_samples_leaf, criterion)
  %
  %   Methods:
  %     classes  C++ datamunge::DecisionTreeClassifier::classes.
  %     predictor_names  C++ datamunge::DecisionTreeClassifier::predictor_names.
  %     observations  C++ datamunge::DecisionTreeClassifier::observations.
  %     node_count  C++ datamunge::DecisionTreeClassifier::node_count.
  %     leaf_count  C++ datamunge::DecisionTreeClassifier::leaf_count.
  %     depth  C++ datamunge::DecisionTreeClassifier::depth.
  %     feature_importance  C++ datamunge::DecisionTreeClassifier::feature_importance.
  %     training_accuracy  C++ datamunge::DecisionTreeClassifier::training_accuracy.
  %     confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
  %     summary  C++ datamunge::DecisionTreeClassifier::summary.
  %     print_summary  C++ datamunge::DecisionTreeClassifier::print_summary.
  %     predict  C++ datamunge::DecisionTreeClassifier::predict.
  %     predict_frame  "class" column plus one numeric probability column per class (prob_<class>).
  %     plot_classification  Scatter of `data` in the (x_feature, y_feature) plane, colored by true class, with misclassified points overlaid in a distinct marker.
  %     plot_decision_regions  Background grid of predicted class regions plus training points; requires exactly 2 predictors.
  %
  %   C++: class datamunge::DecisionTreeClassifier
  methods
    function obj = DecisionTreeClassifier (varargin)
      % DecisionTreeClassifier  Construct a C++ datamunge::DecisionTreeClassifier.
      %
      %   obj = datamunge.DecisionTreeClassifier (data, formula)
      %   obj = datamunge.DecisionTreeClassifier (data, formula, max_depth)
      %   obj = datamunge.DecisionTreeClassifier (data, formula, max_depth, min_samples_split)
      %   obj = datamunge.DecisionTreeClassifier (data, formula, max_depth, min_samples_split, min_samples_leaf)
      %   obj = datamunge.DecisionTreeClassifier (data, formula, max_depth, min_samples_split, min_samples_leaf, criterion)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     max_depth  integer, default 5
      %     min_samples_split  integer, default 2
      %     min_samples_leaf  integer, default 1
      %     criterion  char row, default 'gini': "gini" (default) or "entropy".
      %
      %   C++: datamunge::DecisionTreeClassifier::DecisionTreeClassifier(const DataFrame & data, const std::string & formula, std::size_t max_depth = 5, std::size_t min_samples_split = 2, std::size_t min_samples_leaf = 1, const std::string & criterion = "gini")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2331, false, varargin{:});
    end
    function r = classes (obj)
      % classes  C++ datamunge::DecisionTreeClassifier::classes.
      %
      %   r = obj.classes ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::DecisionTreeClassifier::classes() const
      r = __datamunge__(2316, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::DecisionTreeClassifier::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::DecisionTreeClassifier::predictor_names() const
      r = __datamunge__(2317, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::DecisionTreeClassifier::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DecisionTreeClassifier::observations() const
      r = __datamunge__(2318, obj);
    end
    function r = node_count (obj)
      % node_count  C++ datamunge::DecisionTreeClassifier::node_count.
      %
      %   r = obj.node_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DecisionTreeClassifier::node_count() const
      r = __datamunge__(2319, obj);
    end
    function r = leaf_count (obj)
      % leaf_count  C++ datamunge::DecisionTreeClassifier::leaf_count.
      %
      %   r = obj.leaf_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DecisionTreeClassifier::leaf_count() const
      r = __datamunge__(2320, obj);
    end
    function r = depth (obj)
      % depth  C++ datamunge::DecisionTreeClassifier::depth.
      %
      %   r = obj.depth ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DecisionTreeClassifier::depth() const
      r = __datamunge__(2321, obj);
    end
    function r = feature_importance (obj)
      % feature_importance  C++ datamunge::DecisionTreeClassifier::feature_importance.
      %
      %   r = obj.feature_importance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::DecisionTreeClassifier::feature_importance() const
      r = __datamunge__(2322, obj);
    end
    function r = training_accuracy (obj)
      % training_accuracy  C++ datamunge::DecisionTreeClassifier::training_accuracy.
      %
      %   r = obj.training_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::DecisionTreeClassifier::training_accuracy() const
      r = __datamunge__(2323, obj);
    end
    function r = confusion_matrix (obj)
      % confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
      %
      %   r = obj.confusion_matrix ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DecisionTreeClassifier::confusion_matrix() const
      r = __datamunge__(2324, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::DecisionTreeClassifier::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::DecisionTreeClassifier::summary() const
      r = __datamunge__(2325, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::DecisionTreeClassifier::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::DecisionTreeClassifier::print_summary() const
      __datamunge__(2326, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::DecisionTreeClassifier::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::DecisionTreeClassifier::predict(const DataFrame & newdata) const
      r = __datamunge__(2327, obj, newdata);
    end
    function r = predict_frame (obj, newdata)
      % predict_frame  "class" column plus one numeric probability column per class (prob_<class>).
      %
      %   r = obj.predict_frame (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DecisionTreeClassifier::predict_frame(const DataFrame & newdata) const
      r = __datamunge__(2328, obj, newdata);
    end
    function r = plot_classification (obj, data, x_feature, y_feature)
      % plot_classification  Scatter of `data` in the (x_feature, y_feature) plane, colored by true class, with misclassified points overlaid in a distinct marker.
      %
      %   r = obj.plot_classification (data, x_feature, y_feature)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     x_feature  char row
      %     y_feature  char row
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::DecisionTreeClassifier::plot_classification(const DataFrame & data, const std::string & x_feature, const std::string & y_feature) const
      r = __datamunge__(2329, obj, data, x_feature, y_feature);
    end
    function r = plot_decision_regions (obj, varargin)
      % plot_decision_regions  Background grid of predicted class regions plus training points; requires exactly 2 predictors.
      %
      %   r = obj.plot_decision_regions (x_feature, y_feature)
      %   r = obj.plot_decision_regions (x_feature, y_feature, grid_resolution)
      %
      %   Inputs:
      %     x_feature  char row
      %     y_feature  char row
      %     grid_resolution  integer, default 60
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::DecisionTreeClassifier::plot_decision_regions(const std::string & x_feature, const std::string & y_feature, std::size_t grid_resolution = 60) const
      r = __datamunge__(2330, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::DecisionTreeClassifier. Do not edit.
