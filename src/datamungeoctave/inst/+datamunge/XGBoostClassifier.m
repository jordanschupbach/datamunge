classdef XGBoostClassifier < datamunge.NativeObject
  % XGBoostClassifier  SWIG-friendly facade for datamunge::stats::XGBoostClassifier — regularized, second-order (gradient + Hessian) multiclass gradient boosting, using the same regularized-gain tree-growing objective as the XGBoost algorithm (L1/L2 leaf regularization plus a per-split complexity penalty).
  %
  %   obj = datamunge.XGBoostClassifier (data, formula)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree)
  %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
  %
  %   Methods:
  %     classes  C++ datamunge::XGBoostClassifier::classes.
  %     predictor_names  C++ datamunge::XGBoostClassifier::predictor_names.
  %     observations  C++ datamunge::XGBoostClassifier::observations.
  %     n_trees  C++ datamunge::XGBoostClassifier::n_trees.
  %     feature_importance  Gain-based importance (XGBoost's default "gain" metric), normalized to sum to 1.
  %     training_accuracy  C++ datamunge::XGBoostClassifier::training_accuracy.
  %     confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
  %     training_deviance  Multinomial deviance on the training set after each boosting round (length n_trees()).
  %     summary  C++ datamunge::XGBoostClassifier::summary.
  %     print_summary  C++ datamunge::XGBoostClassifier::print_summary.
  %     predict  C++ datamunge::XGBoostClassifier::predict.
  %     predict_frame  "class" column plus one numeric probability column per class (prob_<class>).
  %     plot_classification  C++ datamunge::XGBoostClassifier::plot_classification.
  %     plot_decision_regions  C++ datamunge::XGBoostClassifier::plot_decision_regions.
  %     plot_training_deviance  C++ datamunge::XGBoostClassifier::plot_training_deviance.
  %
  %   C++: class datamunge::XGBoostClassifier
  methods
    function obj = XGBoostClassifier (varargin)
      % XGBoostClassifier  Construct a C++ datamunge::XGBoostClassifier.
      %
      %   obj = datamunge.XGBoostClassifier (data, formula)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree)
      %   obj = datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     n_trees  integer, default 100
      %     learning_rate  double, default 0.3
      %     max_depth  integer, default 6
      %     lambda  double, default 1.0
      %     alpha  double, default 0.0
      %     gamma  double, default 0.0
      %     min_child_weight  double, default 1.0
      %     min_samples_leaf  integer, default 1
      %     subsample  double, default 1.0
      %     colsample_bytree  double, default 1.0
      %     seed  integer, default 42
      %
      %   C++: datamunge::XGBoostClassifier::XGBoostClassifier(const DataFrame & data, const std::string & formula, std::size_t n_trees = 100, double learning_rate = 0.29999999999999999, std::size_t max_depth = 6, double lambda = 1.0, double alpha = 0.0, double gamma = 0.0, double min_child_weight = 1.0, std::size_t min_samples_leaf = 1, double subsample = 1.0, double colsample_bytree = 1.0, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2679, false, varargin{:});
    end
    function r = classes (obj)
      % classes  C++ datamunge::XGBoostClassifier::classes.
      %
      %   r = obj.classes ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::XGBoostClassifier::classes() const
      r = __datamunge__(2664, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::XGBoostClassifier::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::XGBoostClassifier::predictor_names() const
      r = __datamunge__(2665, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::XGBoostClassifier::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::XGBoostClassifier::observations() const
      r = __datamunge__(2666, obj);
    end
    function r = n_trees (obj)
      % n_trees  C++ datamunge::XGBoostClassifier::n_trees.
      %
      %   r = obj.n_trees ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::XGBoostClassifier::n_trees() const
      r = __datamunge__(2667, obj);
    end
    function r = feature_importance (obj)
      % feature_importance  Gain-based importance (XGBoost's default "gain" metric), normalized to sum to 1.
      %
      %   r = obj.feature_importance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::XGBoostClassifier::feature_importance() const
      r = __datamunge__(2668, obj);
    end
    function r = training_accuracy (obj)
      % training_accuracy  C++ datamunge::XGBoostClassifier::training_accuracy.
      %
      %   r = obj.training_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::XGBoostClassifier::training_accuracy() const
      r = __datamunge__(2669, obj);
    end
    function r = confusion_matrix (obj)
      % confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
      %
      %   r = obj.confusion_matrix ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::XGBoostClassifier::confusion_matrix() const
      r = __datamunge__(2670, obj);
    end
    function r = training_deviance (obj)
      % training_deviance  Multinomial deviance on the training set after each boosting round (length n_trees()).
      %
      %   r = obj.training_deviance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::XGBoostClassifier::training_deviance() const
      r = __datamunge__(2671, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::XGBoostClassifier::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::XGBoostClassifier::summary() const
      r = __datamunge__(2672, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::XGBoostClassifier::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::XGBoostClassifier::print_summary() const
      __datamunge__(2673, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::XGBoostClassifier::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::XGBoostClassifier::predict(const DataFrame & newdata) const
      r = __datamunge__(2674, obj, newdata);
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
      %   C++: DataFrame * datamunge::XGBoostClassifier::predict_frame(const DataFrame & newdata) const
      r = __datamunge__(2675, obj, newdata);
    end
    function r = plot_classification (obj, data, x_feature, y_feature)
      % plot_classification  C++ datamunge::XGBoostClassifier::plot_classification.
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
      %   C++: datamunge::graphics::RPlot datamunge::XGBoostClassifier::plot_classification(const DataFrame & data, const std::string & x_feature, const std::string & y_feature) const
      r = __datamunge__(2676, obj, data, x_feature, y_feature);
    end
    function r = plot_decision_regions (obj, varargin)
      % plot_decision_regions  C++ datamunge::XGBoostClassifier::plot_decision_regions.
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
      %   C++: datamunge::graphics::RPlot datamunge::XGBoostClassifier::plot_decision_regions(const std::string & x_feature, const std::string & y_feature, std::size_t grid_resolution = 60) const
      r = __datamunge__(2677, obj, varargin{:});
    end
    function r = plot_training_deviance (obj)
      % plot_training_deviance  C++ datamunge::XGBoostClassifier::plot_training_deviance.
      %
      %   r = obj.plot_training_deviance ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::XGBoostClassifier::plot_training_deviance() const
      r = __datamunge__(2678, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::XGBoostClassifier. Do not edit.
