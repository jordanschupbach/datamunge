classdef GBMClassifier < datamunge.NativeObject
  % GBMClassifier  SWIG-friendly facade for datamunge::stats::GBMClassifier — multiclass gradient boosting (a sequence of shallow trees, one per class per round, fit to the current multinomial-deviance gradient).
  %
  %   obj = datamunge.GBMClassifier (data, formula)
  %   obj = datamunge.GBMClassifier (data, formula, n_trees)
  %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate)
  %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth)
  %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split)
  %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf)
  %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample)
  %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
  %
  %   Methods:
  %     classes  C++ datamunge::GBMClassifier::classes.
  %     predictor_names  C++ datamunge::GBMClassifier::predictor_names.
  %     observations  C++ datamunge::GBMClassifier::observations.
  %     n_trees  C++ datamunge::GBMClassifier::n_trees.
  %     feature_importance  C++ datamunge::GBMClassifier::feature_importance.
  %     training_accuracy  C++ datamunge::GBMClassifier::training_accuracy.
  %     confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
  %     training_deviance  Multinomial deviance on the training set after each boosting round (length n_trees()).
  %     summary  C++ datamunge::GBMClassifier::summary.
  %     print_summary  C++ datamunge::GBMClassifier::print_summary.
  %     predict  C++ datamunge::GBMClassifier::predict.
  %     predict_frame  "class" column plus one numeric probability column per class (prob_<class>).
  %     plot_classification  C++ datamunge::GBMClassifier::plot_classification.
  %     plot_decision_regions  C++ datamunge::GBMClassifier::plot_decision_regions.
  %     plot_training_deviance  C++ datamunge::GBMClassifier::plot_training_deviance.
  %
  %   C++: class datamunge::GBMClassifier
  methods
    function obj = GBMClassifier (varargin)
      % GBMClassifier  Construct a C++ datamunge::GBMClassifier.
      %
      %   obj = datamunge.GBMClassifier (data, formula)
      %   obj = datamunge.GBMClassifier (data, formula, n_trees)
      %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate)
      %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth)
      %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split)
      %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf)
      %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample)
      %   obj = datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     n_trees  integer, default 100
      %     learning_rate  double, default 0.1
      %     max_depth  integer, default 3
      %     min_samples_split  integer, default 2
      %     min_samples_leaf  integer, default 1
      %     subsample  double, default 1.0
      %     seed  integer, default 42
      %
      %   C++: datamunge::GBMClassifier::GBMClassifier(const DataFrame & data, const std::string & formula, std::size_t n_trees = 100, double learning_rate = 0.10000000000000001, std::size_t max_depth = 3, std::size_t min_samples_split = 2, std::size_t min_samples_leaf = 1, double subsample = 1.0, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2648, false, varargin{:});
    end
    function r = classes (obj)
      % classes  C++ datamunge::GBMClassifier::classes.
      %
      %   r = obj.classes ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GBMClassifier::classes() const
      r = __datamunge__(2633, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::GBMClassifier::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GBMClassifier::predictor_names() const
      r = __datamunge__(2634, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::GBMClassifier::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GBMClassifier::observations() const
      r = __datamunge__(2635, obj);
    end
    function r = n_trees (obj)
      % n_trees  C++ datamunge::GBMClassifier::n_trees.
      %
      %   r = obj.n_trees ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GBMClassifier::n_trees() const
      r = __datamunge__(2636, obj);
    end
    function r = feature_importance (obj)
      % feature_importance  C++ datamunge::GBMClassifier::feature_importance.
      %
      %   r = obj.feature_importance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GBMClassifier::feature_importance() const
      r = __datamunge__(2637, obj);
    end
    function r = training_accuracy (obj)
      % training_accuracy  C++ datamunge::GBMClassifier::training_accuracy.
      %
      %   r = obj.training_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GBMClassifier::training_accuracy() const
      r = __datamunge__(2638, obj);
    end
    function r = confusion_matrix (obj)
      % confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
      %
      %   r = obj.confusion_matrix ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::GBMClassifier::confusion_matrix() const
      r = __datamunge__(2639, obj);
    end
    function r = training_deviance (obj)
      % training_deviance  Multinomial deviance on the training set after each boosting round (length n_trees()).
      %
      %   r = obj.training_deviance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GBMClassifier::training_deviance() const
      r = __datamunge__(2640, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::GBMClassifier::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GBMClassifier::summary() const
      r = __datamunge__(2641, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::GBMClassifier::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::GBMClassifier::print_summary() const
      __datamunge__(2642, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::GBMClassifier::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GBMClassifier::predict(const DataFrame & newdata) const
      r = __datamunge__(2643, obj, newdata);
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
      %   C++: DataFrame * datamunge::GBMClassifier::predict_frame(const DataFrame & newdata) const
      r = __datamunge__(2644, obj, newdata);
    end
    function r = plot_classification (obj, data, x_feature, y_feature)
      % plot_classification  C++ datamunge::GBMClassifier::plot_classification.
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
      %   C++: datamunge::graphics::RPlot datamunge::GBMClassifier::plot_classification(const DataFrame & data, const std::string & x_feature, const std::string & y_feature) const
      r = __datamunge__(2645, obj, data, x_feature, y_feature);
    end
    function r = plot_decision_regions (obj, varargin)
      % plot_decision_regions  C++ datamunge::GBMClassifier::plot_decision_regions.
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
      %   C++: datamunge::graphics::RPlot datamunge::GBMClassifier::plot_decision_regions(const std::string & x_feature, const std::string & y_feature, std::size_t grid_resolution = 60) const
      r = __datamunge__(2646, obj, varargin{:});
    end
    function r = plot_training_deviance (obj)
      % plot_training_deviance  C++ datamunge::GBMClassifier::plot_training_deviance.
      %
      %   r = obj.plot_training_deviance ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GBMClassifier::plot_training_deviance() const
      r = __datamunge__(2647, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::GBMClassifier. Do not edit.
