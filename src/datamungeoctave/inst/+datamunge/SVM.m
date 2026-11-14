classdef SVM < datamunge.NativeObject
  % SVM  SWIG-friendly facade for datamunge::stats::SVM — R-`e1071::svm()`-style multi-class SVM classification.
  %
  %   obj = datamunge.SVM (data, formula)
  %   obj = datamunge.SVM (data, formula, kernel)
  %   obj = datamunge.SVM (data, formula, kernel, cost)
  %   obj = datamunge.SVM (data, formula, kernel, cost, gamma)
  %   obj = datamunge.SVM (data, formula, kernel, cost, gamma, coef0)
  %   obj = datamunge.SVM (data, formula, kernel, cost, gamma, coef0, degree)
  %   obj = datamunge.SVM (data, formula, kernel, cost, gamma, coef0, degree, scale)
  %
  %   Methods:
  %     classes  C++ datamunge::SVM::classes.
  %     predictor_names  C++ datamunge::SVM::predictor_names.
  %     observations  C++ datamunge::SVM::observations.
  %     num_support_vectors  C++ datamunge::SVM::num_support_vectors.
  %     training_accuracy  C++ datamunge::SVM::training_accuracy.
  %     confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
  %     summary  C++ datamunge::SVM::summary.
  %     print_summary  C++ datamunge::SVM::print_summary.
  %     predict  C++ datamunge::SVM::predict.
  %     predict_frame  "class" column plus one numeric vote-count column per class (votes_<class>).
  %
  %   C++: class datamunge::SVM
  methods
    function obj = SVM (varargin)
      % SVM  Construct a C++ datamunge::SVM.
      %
      %   obj = datamunge.SVM (data, formula)
      %   obj = datamunge.SVM (data, formula, kernel)
      %   obj = datamunge.SVM (data, formula, kernel, cost)
      %   obj = datamunge.SVM (data, formula, kernel, cost, gamma)
      %   obj = datamunge.SVM (data, formula, kernel, cost, gamma, coef0)
      %   obj = datamunge.SVM (data, formula, kernel, cost, gamma, coef0, degree)
      %   obj = datamunge.SVM (data, formula, kernel, cost, gamma, coef0, degree, scale)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     kernel  char row, default 'radial': One of "linear", "polynomial", "radial" (default), "sigmoid".
      %     cost  double, default 1.0
      %     gamma  double, default -1.0: <= 0 means "auto" = 1 / number of predictors.
      %     coef0  double, default 0.0
      %     degree  integer, default 3
      %     scale  logical, default true
      %
      %   C++: datamunge::SVM::SVM(const DataFrame & data, const std::string & formula, const std::string & kernel = "radial", double cost = 1.0, double gamma = -1.0, double coef0 = 0.0, int degree = 3, bool scale = true)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2315, false, varargin{:});
    end
    function r = classes (obj)
      % classes  C++ datamunge::SVM::classes.
      %
      %   r = obj.classes ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::SVM::classes() const
      r = __datamunge__(2305, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::SVM::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::SVM::predictor_names() const
      r = __datamunge__(2306, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::SVM::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::SVM::observations() const
      r = __datamunge__(2307, obj);
    end
    function r = num_support_vectors (obj)
      % num_support_vectors  C++ datamunge::SVM::num_support_vectors.
      %
      %   r = obj.num_support_vectors ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::SVM::num_support_vectors() const
      r = __datamunge__(2308, obj);
    end
    function r = training_accuracy (obj)
      % training_accuracy  C++ datamunge::SVM::training_accuracy.
      %
      %   r = obj.training_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::SVM::training_accuracy() const
      r = __datamunge__(2309, obj);
    end
    function r = confusion_matrix (obj)
      % confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
      %
      %   r = obj.confusion_matrix ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::SVM::confusion_matrix() const
      r = __datamunge__(2310, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::SVM::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::SVM::summary() const
      r = __datamunge__(2311, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::SVM::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::SVM::print_summary() const
      __datamunge__(2312, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::SVM::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::SVM::predict(const DataFrame & newdata) const
      r = __datamunge__(2313, obj, newdata);
    end
    function r = predict_frame (obj, newdata)
      % predict_frame  "class" column plus one numeric vote-count column per class (votes_<class>).
      %
      %   r = obj.predict_frame (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::SVM::predict_frame(const DataFrame & newdata) const
      r = __datamunge__(2314, obj, newdata);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::SVM. Do not edit.
