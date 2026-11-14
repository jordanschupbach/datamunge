classdef SammonMapping < datamunge.NativeObject
  % SammonMapping  SWIG-friendly facade for datamunge::stats::SammonMapping — iteratively minimizes a weighted distance-preservation "stress" (small high-dimensional distances weighted much more heavily than large ones) via Sammon's original pseudo-Newton update.
  %
  %   obj = datamunge.SammonMapping (data, feature_columns)
  %   obj = datamunge.SammonMapping (data, feature_columns, n_components)
  %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate)
  %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate, max_iterations)
  %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate, max_iterations, tolerance)
  %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate, max_iterations, tolerance, metric)
  %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate, max_iterations, tolerance, metric, seed)
  %   obj = datamunge.SammonMapping (data, encoded_feature_columns)
  %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components)
  %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate)
  %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate, max_iterations)
  %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate, max_iterations, tolerance)
  %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate, max_iterations, tolerance, metric)
  %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate, max_iterations, tolerance, metric, seed)
  %
  %   Methods:
  %     feature_names  C++ datamunge::SammonMapping::feature_names.
  %     observations  C++ datamunge::SammonMapping::observations.
  %     n_components  C++ datamunge::SammonMapping::n_components.
  %     kept_row_indices  C++ datamunge::SammonMapping::kept_row_indices.
  %     dimension  C++ datamunge::SammonMapping::dimension.
  %     embedding_frame  C++ datamunge::SammonMapping::embedding_frame.
  %     stress  Final Sammon stress -- lower is better, 0 is a perfect distance-preserving embedding.
  %     iterations_run  C++ datamunge::SammonMapping::iterations_run.
  %     summary  C++ datamunge::SammonMapping::summary.
  %     print_summary  C++ datamunge::SammonMapping::print_summary.
  %     plot_embedding  C++ datamunge::SammonMapping::plot_embedding.
  %     plot_embedding_grouped  C++ datamunge::SammonMapping::plot_embedding_grouped.
  %
  %   C++: class datamunge::SammonMapping
  methods
    function obj = SammonMapping (varargin)
      % SammonMapping  Construct a C++ datamunge::SammonMapping.
      %
      %   obj = datamunge.SammonMapping (data, feature_columns)
      %   obj = datamunge.SammonMapping (data, feature_columns, n_components)
      %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate)
      %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate, max_iterations)
      %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate, max_iterations, tolerance)
      %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate, max_iterations, tolerance, metric)
      %   obj = datamunge.SammonMapping (data, feature_columns, n_components, learning_rate, max_iterations, tolerance, metric, seed)
      %   obj = datamunge.SammonMapping (data, encoded_feature_columns)
      %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components)
      %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate)
      %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate, max_iterations)
      %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate, max_iterations, tolerance)
      %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate, max_iterations, tolerance, metric)
      %   obj = datamunge.SammonMapping (data, encoded_feature_columns, n_components, learning_rate, max_iterations, tolerance, metric, seed)
      %
      %   C++: datamunge::SammonMapping::SammonMapping(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, double learning_rate = 0.29999999999999999, std::size_t max_iterations = 500, double tolerance = 1.0000000000000001E-9, const std::string & metric = "euclidean", std::uint64_t seed = 42)
      %   C++: datamunge::SammonMapping::SammonMapping(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, double learning_rate = 0.29999999999999999, std::size_t max_iterations = 500, double tolerance = 1.0000000000000001E-9, const std::string & metric = "euclidean", std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2599, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::SammonMapping::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::SammonMapping::feature_names() const
      r = __datamunge__(2587, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::SammonMapping::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::SammonMapping::observations() const
      r = __datamunge__(2588, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::SammonMapping::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::SammonMapping::n_components() const
      r = __datamunge__(2589, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  C++ datamunge::SammonMapping::kept_row_indices.
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::SammonMapping::kept_row_indices() const
      r = __datamunge__(2590, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::SammonMapping::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::SammonMapping::dimension(std::size_t index) const
      r = __datamunge__(2591, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  C++ datamunge::SammonMapping::embedding_frame.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::SammonMapping::embedding_frame() const
      r = __datamunge__(2592, obj);
    end
    function r = stress (obj)
      % stress  Final Sammon stress -- lower is better, 0 is a perfect distance-preserving embedding.
      %
      %   r = obj.stress ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::SammonMapping::stress() const
      r = __datamunge__(2593, obj);
    end
    function r = iterations_run (obj)
      % iterations_run  C++ datamunge::SammonMapping::iterations_run.
      %
      %   r = obj.iterations_run ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::SammonMapping::iterations_run() const
      r = __datamunge__(2594, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::SammonMapping::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::SammonMapping::summary() const
      r = __datamunge__(2595, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::SammonMapping::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::SammonMapping::print_summary() const
      __datamunge__(2596, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  C++ datamunge::SammonMapping::plot_embedding.
      %
      %   r = obj.plot_embedding ()
      %   r = obj.plot_embedding (dimension_x)
      %   r = obj.plot_embedding (dimension_x, dimension_y)
      %
      %   Inputs:
      %     dimension_x  integer, default 0
      %     dimension_y  integer, default 1
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::SammonMapping::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2597, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  C++ datamunge::SammonMapping::plot_embedding_grouped.
      %
      %   r = obj.plot_embedding_grouped (group_labels)
      %   r = obj.plot_embedding_grouped (group_labels, dimension_x)
      %   r = obj.plot_embedding_grouped (group_labels, dimension_x, dimension_y)
      %
      %   Inputs:
      %     group_labels  cell array of char
      %     dimension_x  integer, default 0
      %     dimension_y  integer, default 1
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::SammonMapping::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2598, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::SammonMapping. Do not edit.
