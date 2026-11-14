classdef LDA < datamunge.NativeObject
  % LDA  SWIG-friendly facade for datamunge::stats::LDA — R-`MASS::lda()`-style linear discriminant analysis.
  %
  %   obj = datamunge.LDA (data, formula)
  %   obj = datamunge.LDA (data, formula, priors)
  %
  %   Methods:
  %     classes  C++ datamunge::LDA::classes.
  %     predictor_names  C++ datamunge::LDA::predictor_names.
  %     observations  C++ datamunge::LDA::observations.
  %     num_discriminants  C++ datamunge::LDA::num_discriminants.
  %     priors  C++ datamunge::LDA::priors.
  %     group_means  One row per class, one numeric column per predictor.
  %     scaling  One row per predictor, one numeric column per linear discriminant (LD1, LD2, ...).
  %     proportion_of_trace  C++ datamunge::LDA::proportion_of_trace.
  %     training_accuracy  C++ datamunge::LDA::training_accuracy.
  %     confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
  %     summary  C++ datamunge::LDA::summary.
  %     print_summary  C++ datamunge::LDA::print_summary.
  %     to_md  The fit as Markdown tables (priors, group means, coefficients, proportion of trace).
  %     predict  C++ datamunge::LDA::predict.
  %     predict_frame  "class" column, LD1/LD2/... discriminant scores, and posterior_<class> probability columns.
  %     plot_discriminants  LD1/LD2 scatter; with `decision_regions`, class regions shaded and boundaries drawn.
  %     save_discriminant_plot  C++ datamunge::LDA::save_discriminant_plot.
  %     plot_decision_regions  Decision regions in predictor space, for a model with exactly two predictors.
  %
  %   C++: class datamunge::LDA
  methods
    function obj = LDA (varargin)
      % LDA  Construct a C++ datamunge::LDA.
      %
      %   obj = datamunge.LDA (data, formula)
      %   obj = datamunge.LDA (data, formula, priors)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     priors  double vector, default the C++ default: Optional class prior probabilities (must sum to 1, ordered as classes() once sorted alphabetically); pass an empty vector (the default) to use observed class proportions.
      %
      %   C++: datamunge::LDA::LDA(const DataFrame & data, const std::string & formula, const std::vector<double> & priors = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2304, false, varargin{:});
    end
    function r = classes (obj)
      % classes  C++ datamunge::LDA::classes.
      %
      %   r = obj.classes ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LDA::classes() const
      r = __datamunge__(2286, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::LDA::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LDA::predictor_names() const
      r = __datamunge__(2287, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::LDA::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LDA::observations() const
      r = __datamunge__(2288, obj);
    end
    function r = num_discriminants (obj)
      % num_discriminants  C++ datamunge::LDA::num_discriminants.
      %
      %   r = obj.num_discriminants ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LDA::num_discriminants() const
      r = __datamunge__(2289, obj);
    end
    function r = priors (obj)
      % priors  C++ datamunge::LDA::priors.
      %
      %   r = obj.priors ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LDA::priors() const
      r = __datamunge__(2290, obj);
    end
    function r = group_means (obj)
      % group_means  One row per class, one numeric column per predictor.
      %
      %   r = obj.group_means ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::LDA::group_means() const
      r = __datamunge__(2291, obj);
    end
    function r = scaling (obj)
      % scaling  One row per predictor, one numeric column per linear discriminant (LD1, LD2, ...).
      %
      %   r = obj.scaling ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::LDA::scaling() const
      r = __datamunge__(2292, obj);
    end
    function r = proportion_of_trace (obj)
      % proportion_of_trace  C++ datamunge::LDA::proportion_of_trace.
      %
      %   r = obj.proportion_of_trace ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LDA::proportion_of_trace() const
      r = __datamunge__(2293, obj);
    end
    function r = training_accuracy (obj)
      % training_accuracy  C++ datamunge::LDA::training_accuracy.
      %
      %   r = obj.training_accuracy ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LDA::training_accuracy() const
      r = __datamunge__(2294, obj);
    end
    function r = confusion_matrix (obj)
      % confusion_matrix  "actual" column plus one numeric column per class (counts), both ordered as classes().
      %
      %   r = obj.confusion_matrix ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::LDA::confusion_matrix() const
      r = __datamunge__(2295, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::LDA::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LDA::summary() const
      r = __datamunge__(2296, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::LDA::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::LDA::print_summary() const
      __datamunge__(2297, obj);
    end
    function r = to_md (obj, varargin)
      % to_md  The fit as Markdown tables (priors, group means, coefficients, proportion of trace).
      %
      %   r = obj.to_md ()
      %   r = obj.to_md (digits)
      %
      %   Inputs:
      %     digits  integer, default 4
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LDA::to_md(int digits = 4) const
      r = __datamunge__(2298, obj, varargin{:});
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::LDA::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LDA::predict(const DataFrame & newdata) const
      r = __datamunge__(2299, obj, newdata);
    end
    function r = predict_frame (obj, newdata)
      % predict_frame  "class" column, LD1/LD2/... discriminant scores, and posterior_<class> probability columns.
      %
      %   r = obj.predict_frame (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::LDA::predict_frame(const DataFrame & newdata) const
      r = __datamunge__(2300, obj, newdata);
    end
    function r = plot_discriminants (obj, varargin)
      % plot_discriminants  LD1/LD2 scatter; with `decision_regions`, class regions shaded and boundaries drawn.
      %
      %   r = obj.plot_discriminants ()
      %   r = obj.plot_discriminants (decision_regions)
      %
      %   Inputs:
      %     decision_regions  logical, default true
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::LDA::plot_discriminants(bool decision_regions = true) const
      r = __datamunge__(2301, obj, varargin{:});
    end
    function save_discriminant_plot (obj, path)
      % save_discriminant_plot  C++ datamunge::LDA::save_discriminant_plot.
      %
      %   obj.save_discriminant_plot (path)
      %
      %   Inputs:
      %     path  char row
      %
      %   C++: void datamunge::LDA::save_discriminant_plot(const std::string & path) const
      __datamunge__(2302, obj, path);
    end
    function r = plot_decision_regions (obj)
      % plot_decision_regions  Decision regions in predictor space, for a model with exactly two predictors.
      %
      %   r = obj.plot_decision_regions ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::LDA::plot_decision_regions() const
      r = __datamunge__(2303, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::LDA. Do not edit.
