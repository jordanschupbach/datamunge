classdef KNNClassifier < datamunge.NativeObject
  % KNNClassifier  SWIG-friendly facade for datamunge::stats::KNNClassifier — k-nearest-neighbors classification. Since a training point's nearest neighbor is always itself, training_accuracy()/confusion_matrix() report leave-one-out performance rather than a trivial resubstitution fit.
  %
  %   obj = datamunge.KNNClassifier (data, formula)
  %   obj = datamunge.KNNClassifier (data, formula, k)
  %   obj = datamunge.KNNClassifier (data, formula, k, metric)
  %   obj = datamunge.KNNClassifier (data, formula, k, metric, weighted)
  %   obj = datamunge.KNNClassifier (data, formula, k, metric, weighted, standardize)
  %
  %   Methods:
  %     classes  C++ datamunge::KNNClassifier::classes.
  %     predictor_names  C++ datamunge::KNNClassifier::predictor_names.
  %     observations  C++ datamunge::KNNClassifier::observations.
  %     k  C++ datamunge::KNNClassifier::k.
  %     training_accuracy  C++ datamunge::KNNClassifier::training_accuracy.
  %     confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes(); leave-one-out.
  %     summary  C++ datamunge::KNNClassifier::summary.
  %     print_summary  C++ datamunge::KNNClassifier::print_summary.
  %     predict  C++ datamunge::KNNClassifier::predict.
  %     predict_frame  "class" column plus one numeric vote-share column per class (votes_<class>).
  %     plot_classification  Scatter of `data` in the (x_feature, y_feature) plane, colored by true class, with misclassified points overlaid in a distinct marker.
  %     plot_decision_regions  Background grid of predicted class regions plus training points; requires exactly 2 predictors.
  %
  %   C++: class datamunge::KNNClassifier
  methods
    function obj = KNNClassifier (varargin)
      % KNNClassifier  Construct a C++ datamunge::KNNClassifier.
      %
      %   obj = datamunge.KNNClassifier (data, formula)
      %   obj = datamunge.KNNClassifier (data, formula, k)
      %   obj = datamunge.KNNClassifier (data, formula, k, metric)
      %   obj = datamunge.KNNClassifier (data, formula, k, metric, weighted)
      %   obj = datamunge.KNNClassifier (data, formula, k, metric, weighted, standardize)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     k  integer, default 5
      %     metric  char row, default 'euclidean': "euclidean" (default) or "manhattan".
      %     weighted  logical, default false
      %     standardize  logical, default true
      %
      %   C++: datamunge::KNNClassifier::KNNClassifier(const DataFrame & data, const std::string & formula, std::size_t k = 5, const std::string & metric = "euclidean", bool weighted = false, bool standardize = true)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2460, false, varargin{:});
    end
    function r = classes (obj)
      % classes  C++ datamunge::KNNClassifier::classes.
      %
      %   r = obj.classes ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::KNNClassifier::classes() const
      r = __datamunge__(2448, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::KNNClassifier::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::KNNClassifier::predictor_names() const
      r = __datamunge__(2449, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::KNNClassifier::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KNNClassifier::observations() const
      r = __datamunge__(2450, obj);
    end
    function r = k (obj)
      % k  C++ datamunge::KNNClassifier::k.
      %
      %   r = obj.k ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KNNClassifier::k() const
      r = __datamunge__(2451, obj);
    end
    function r = training_accuracy (obj)
      % training_accuracy  C++ datamunge::KNNClassifier::training_accuracy.
      %
      %   r = obj.training_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::KNNClassifier::training_accuracy() const
      r = __datamunge__(2452, obj);
    end
    function r = confusion_matrix (obj)
      % confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes(); leave-one-out.
      %
      %   r = obj.confusion_matrix ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::KNNClassifier::confusion_matrix() const
      r = __datamunge__(2453, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::KNNClassifier::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::KNNClassifier::summary() const
      r = __datamunge__(2454, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::KNNClassifier::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::KNNClassifier::print_summary() const
      __datamunge__(2455, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::KNNClassifier::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::KNNClassifier::predict(const DataFrame & newdata) const
      r = __datamunge__(2456, obj, newdata);
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
      %   C++: DataFrame * datamunge::KNNClassifier::predict_frame(const DataFrame & newdata) const
      r = __datamunge__(2457, obj, newdata);
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
      %   C++: datamunge::graphics::RPlot datamunge::KNNClassifier::plot_classification(const DataFrame & data, const std::string & x_feature, const std::string & y_feature) const
      r = __datamunge__(2458, obj, data, x_feature, y_feature);
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
      %   C++: datamunge::graphics::RPlot datamunge::KNNClassifier::plot_decision_regions(const std::string & x_feature, const std::string & y_feature, std::size_t grid_resolution = 60) const
      r = __datamunge__(2459, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::KNNClassifier. Do not edit.
