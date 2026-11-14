classdef NaiveBayesClassifier < datamunge.NativeObject
  % NaiveBayesClassifier  SWIG-friendly facade for datamunge::stats::NaiveBayesClassifier — a Naive Bayes classifier that models numeric predictors with per-class Gaussians and categorical predictors with per-class frequency tables (each categorical predictor's levels modeled jointly as one variable, not as separate independent dummy features).
  %
  %   obj = datamunge.NaiveBayesClassifier (data, formula)
  %   obj = datamunge.NaiveBayesClassifier (data, formula, laplace_smoothing)
  %   obj = datamunge.NaiveBayesClassifier (data, formula, laplace_smoothing, var_smoothing)
  %
  %   Methods:
  %     classes  C++ datamunge::NaiveBayesClassifier::classes.
  %     predictor_names  C++ datamunge::NaiveBayesClassifier::predictor_names.
  %     observations  C++ datamunge::NaiveBayesClassifier::observations.
  %     class_priors  C++ datamunge::NaiveBayesClassifier::class_priors.
  %     training_accuracy  C++ datamunge::NaiveBayesClassifier::training_accuracy.
  %     confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
  %     summary  C++ datamunge::NaiveBayesClassifier::summary.
  %     print_summary  C++ datamunge::NaiveBayesClassifier::print_summary.
  %     predict  C++ datamunge::NaiveBayesClassifier::predict.
  %     predict_frame  "class" column plus one numeric probability column per class (prob_<class>).
  %     plot_classification  C++ datamunge::NaiveBayesClassifier::plot_classification.
  %     plot_decision_regions  Background grid of predicted class regions; requires exactly two predictors, both numeric.
  %
  %   C++: class datamunge::NaiveBayesClassifier
  methods
    function obj = NaiveBayesClassifier (varargin)
      % NaiveBayesClassifier  Construct a C++ datamunge::NaiveBayesClassifier.
      %
      %   obj = datamunge.NaiveBayesClassifier (data, formula)
      %   obj = datamunge.NaiveBayesClassifier (data, formula, laplace_smoothing)
      %   obj = datamunge.NaiveBayesClassifier (data, formula, laplace_smoothing, var_smoothing)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     laplace_smoothing  double, default 1.0
      %     var_smoothing  double, default 1e-09
      %
      %   C++: datamunge::NaiveBayesClassifier::NaiveBayesClassifier(const DataFrame & data, const std::string & formula, double laplace_smoothing = 1.0, double var_smoothing = 1.0000000000000001E-9)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2746, false, varargin{:});
    end
    function r = classes (obj)
      % classes  C++ datamunge::NaiveBayesClassifier::classes.
      %
      %   r = obj.classes ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::NaiveBayesClassifier::classes() const
      r = __datamunge__(2734, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::NaiveBayesClassifier::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::NaiveBayesClassifier::predictor_names() const
      r = __datamunge__(2735, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::NaiveBayesClassifier::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::NaiveBayesClassifier::observations() const
      r = __datamunge__(2736, obj);
    end
    function r = class_priors (obj)
      % class_priors  C++ datamunge::NaiveBayesClassifier::class_priors.
      %
      %   r = obj.class_priors ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::NaiveBayesClassifier::class_priors() const
      r = __datamunge__(2737, obj);
    end
    function r = training_accuracy (obj)
      % training_accuracy  C++ datamunge::NaiveBayesClassifier::training_accuracy.
      %
      %   r = obj.training_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::NaiveBayesClassifier::training_accuracy() const
      r = __datamunge__(2738, obj);
    end
    function r = confusion_matrix (obj)
      % confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
      %
      %   r = obj.confusion_matrix ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::NaiveBayesClassifier::confusion_matrix() const
      r = __datamunge__(2739, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::NaiveBayesClassifier::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::NaiveBayesClassifier::summary() const
      r = __datamunge__(2740, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::NaiveBayesClassifier::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::NaiveBayesClassifier::print_summary() const
      __datamunge__(2741, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::NaiveBayesClassifier::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::NaiveBayesClassifier::predict(const DataFrame & newdata) const
      r = __datamunge__(2742, obj, newdata);
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
      %   C++: DataFrame * datamunge::NaiveBayesClassifier::predict_frame(const DataFrame & newdata) const
      r = __datamunge__(2743, obj, newdata);
    end
    function r = plot_classification (obj, data, x_feature, y_feature)
      % plot_classification  C++ datamunge::NaiveBayesClassifier::plot_classification.
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
      %   C++: datamunge::graphics::RPlot datamunge::NaiveBayesClassifier::plot_classification(const DataFrame & data, const std::string & x_feature, const std::string & y_feature) const
      r = __datamunge__(2744, obj, data, x_feature, y_feature);
    end
    function r = plot_decision_regions (obj, varargin)
      % plot_decision_regions  Background grid of predicted class regions; requires exactly two predictors, both numeric.
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
      %   C++: datamunge::graphics::RPlot datamunge::NaiveBayesClassifier::plot_decision_regions(const std::string & x_feature, const std::string & y_feature, std::size_t grid_resolution = 60) const
      r = __datamunge__(2745, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::NaiveBayesClassifier. Do not edit.
