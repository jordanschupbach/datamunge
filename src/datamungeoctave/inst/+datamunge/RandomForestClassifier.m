classdef RandomForestClassifier < datamunge.NativeObject
  % RandomForestClassifier  SWIG-friendly facade for datamunge::stats::RandomForestClassifier — a bagged ensemble of CART trees, each fit on a bootstrap sample with a random subset of predictors considered at every split.
  %
  %   obj = datamunge.RandomForestClassifier (data, formula)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, criterion)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, criterion, bootstrap)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, criterion, bootstrap, sample_fraction)
  %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, criterion, bootstrap, sample_fraction, seed)
  %
  %   Methods:
  %     classes  C++ datamunge::RandomForestClassifier::classes.
  %     predictor_names  C++ datamunge::RandomForestClassifier::predictor_names.
  %     observations  C++ datamunge::RandomForestClassifier::observations.
  %     n_trees  C++ datamunge::RandomForestClassifier::n_trees.
  %     max_features_used  C++ datamunge::RandomForestClassifier::max_features_used.
  %     feature_importance  C++ datamunge::RandomForestClassifier::feature_importance.
  %     training_accuracy  C++ datamunge::RandomForestClassifier::training_accuracy.
  %     oob_accuracy  Out-of-bag accuracy estimate (majority vote among trees that did not train on each row).
  %     confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
  %     summary  C++ datamunge::RandomForestClassifier::summary.
  %     print_summary  C++ datamunge::RandomForestClassifier::print_summary.
  %     predict  C++ datamunge::RandomForestClassifier::predict.
  %     predict_frame  "class" column plus one numeric vote-share column per class (votes_<class>).
  %     plot_classification  Scatter of `data` in the (x_feature, y_feature) plane, colored by true class, with misclassified points overlaid in a distinct marker.
  %     plot_decision_regions  Background grid of majority-vote predicted class regions plus training points; requires exactly 2 predictors.
  %
  %   C++: class datamunge::RandomForestClassifier
  methods
    function obj = RandomForestClassifier (varargin)
      % RandomForestClassifier  Construct a C++ datamunge::RandomForestClassifier.
      %
      %   obj = datamunge.RandomForestClassifier (data, formula)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, criterion)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, criterion, bootstrap)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, criterion, bootstrap, sample_fraction)
      %   obj = datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, criterion, bootstrap, sample_fraction, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     n_trees  integer, default 100
      %     max_depth  integer, default 10
      %     min_samples_split  integer, default 2
      %     min_samples_leaf  integer, default 1
      %     max_features  integer, default 0: 0 = auto (floor(sqrt(number of predictors))).
      %     criterion  char row, default 'gini': "gini" (default) or "entropy".
      %     bootstrap  logical, default true
      %     sample_fraction  double, default 1.0
      %     seed  integer, default 42
      %
      %   C++: datamunge::RandomForestClassifier::RandomForestClassifier(const DataFrame & data, const std::string & formula, std::size_t n_trees = 100, std::size_t max_depth = 10, std::size_t min_samples_split = 2, std::size_t min_samples_leaf = 1, std::size_t max_features = 0, const std::string & criterion = "gini", bool bootstrap = true, double sample_fraction = 1.0, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2362, false, varargin{:});
    end
    function r = classes (obj)
      % classes  C++ datamunge::RandomForestClassifier::classes.
      %
      %   r = obj.classes ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::RandomForestClassifier::classes() const
      r = __datamunge__(2347, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::RandomForestClassifier::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::RandomForestClassifier::predictor_names() const
      r = __datamunge__(2348, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::RandomForestClassifier::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::RandomForestClassifier::observations() const
      r = __datamunge__(2349, obj);
    end
    function r = n_trees (obj)
      % n_trees  C++ datamunge::RandomForestClassifier::n_trees.
      %
      %   r = obj.n_trees ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::RandomForestClassifier::n_trees() const
      r = __datamunge__(2350, obj);
    end
    function r = max_features_used (obj)
      % max_features_used  C++ datamunge::RandomForestClassifier::max_features_used.
      %
      %   r = obj.max_features_used ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::RandomForestClassifier::max_features_used() const
      r = __datamunge__(2351, obj);
    end
    function r = feature_importance (obj)
      % feature_importance  C++ datamunge::RandomForestClassifier::feature_importance.
      %
      %   r = obj.feature_importance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::RandomForestClassifier::feature_importance() const
      r = __datamunge__(2352, obj);
    end
    function r = training_accuracy (obj)
      % training_accuracy  C++ datamunge::RandomForestClassifier::training_accuracy.
      %
      %   r = obj.training_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::RandomForestClassifier::training_accuracy() const
      r = __datamunge__(2353, obj);
    end
    function r = oob_accuracy (obj)
      % oob_accuracy  Out-of-bag accuracy estimate (majority vote among trees that did not train on each row).
      %
      %   r = obj.oob_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::RandomForestClassifier::oob_accuracy() const
      r = __datamunge__(2354, obj);
    end
    function r = confusion_matrix (obj)
      % confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
      %
      %   r = obj.confusion_matrix ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::RandomForestClassifier::confusion_matrix() const
      r = __datamunge__(2355, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::RandomForestClassifier::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::RandomForestClassifier::summary() const
      r = __datamunge__(2356, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::RandomForestClassifier::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::RandomForestClassifier::print_summary() const
      __datamunge__(2357, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::RandomForestClassifier::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::RandomForestClassifier::predict(const DataFrame & newdata) const
      r = __datamunge__(2358, obj, newdata);
    end
    function r = predict_frame (obj, newdata)
      % predict_frame  "class" column plus one numeric vote-share column per class (votes_<class>).
      %
      %   r = obj.predict_frame (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::RandomForestClassifier::predict_frame(const DataFrame & newdata) const
      r = __datamunge__(2359, obj, newdata);
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
      %   C++: datamunge::graphics::RPlot datamunge::RandomForestClassifier::plot_classification(const DataFrame & data, const std::string & x_feature, const std::string & y_feature) const
      r = __datamunge__(2360, obj, data, x_feature, y_feature);
    end
    function r = plot_decision_regions (obj, varargin)
      % plot_decision_regions  Background grid of majority-vote predicted class regions plus training points; requires exactly 2 predictors.
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
      %   C++: datamunge::graphics::RPlot datamunge::RandomForestClassifier::plot_decision_regions(const std::string & x_feature, const std::string & y_feature, std::size_t grid_resolution = 60) const
      r = __datamunge__(2361, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::RandomForestClassifier. Do not edit.
