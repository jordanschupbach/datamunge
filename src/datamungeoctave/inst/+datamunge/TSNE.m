classdef TSNE < datamunge.NativeObject
  % TSNE  SWIG-friendly facade for datamunge::stats::TSNE — t-distributed Stochastic Neighbor Embedding, a nonlinear method that preserves local neighborhood structure (via a perplexity-calibrated probability distribution) primarily for 2D/3D visualization.
  %
  %   obj = datamunge.TSNE (data, feature_columns)
  %   obj = datamunge.TSNE (data, feature_columns, n_components)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration, metric)
  %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration, metric, seed)
  %   obj = datamunge.TSNE (data, encoded_feature_columns)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration, metric)
  %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration, metric, seed)
  %
  %   Methods:
  %     feature_names  C++ datamunge::TSNE::feature_names.
  %     observations  C++ datamunge::TSNE::observations.
  %     n_components  C++ datamunge::TSNE::n_components.
  %     kept_row_indices  C++ datamunge::TSNE::kept_row_indices.
  %     achieved_perplexity  The achieved perplexity for each fitted point after the per-point binary search -- should be close to the constructor's perplexity argument if calibration converged.
  %     dimension  C++ datamunge::TSNE::dimension.
  %     embedding_frame  C++ datamunge::TSNE::embedding_frame.
  %     summary  C++ datamunge::TSNE::summary.
  %     print_summary  C++ datamunge::TSNE::print_summary.
  %     plot_embedding  C++ datamunge::TSNE::plot_embedding.
  %     plot_embedding_grouped  C++ datamunge::TSNE::plot_embedding_grouped.
  %
  %   C++: class datamunge::TSNE
  methods
    function obj = TSNE (varargin)
      % TSNE  Construct a C++ datamunge::TSNE.
      %
      %   obj = datamunge.TSNE (data, feature_columns)
      %   obj = datamunge.TSNE (data, feature_columns, n_components)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration, metric)
      %   obj = datamunge.TSNE (data, feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration, metric, seed)
      %   obj = datamunge.TSNE (data, encoded_feature_columns)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration, metric)
      %   obj = datamunge.TSNE (data, encoded_feature_columns, n_components, perplexity, max_iterations, learning_rate, early_exaggeration, early_exaggeration_iterations, initial_momentum, final_momentum, momentum_switch_iteration, metric, seed)
      %
      %   C++: datamunge::TSNE::TSNE(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, double perplexity = 30.0, std::size_t max_iterations = 1000, double learning_rate = 200.0, double early_exaggeration = 12.0, std::size_t early_exaggeration_iterations = 250, double initial_momentum = 0.5, double final_momentum = 0.80000000000000004, std::size_t momentum_switch_iteration = 250, const std::string & metric = "euclidean", std::uint64_t seed = 42)
      %   C++: datamunge::TSNE::TSNE(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, double perplexity = 30.0, std::size_t max_iterations = 1000, double learning_rate = 200.0, double early_exaggeration = 12.0, std::size_t early_exaggeration_iterations = 250, double initial_momentum = 0.5, double final_momentum = 0.80000000000000004, std::size_t momentum_switch_iteration = 250, const std::string & metric = "euclidean", std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2550, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::TSNE::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::TSNE::feature_names() const
      r = __datamunge__(2539, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::TSNE::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::TSNE::observations() const
      r = __datamunge__(2540, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::TSNE::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::TSNE::n_components() const
      r = __datamunge__(2541, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  C++ datamunge::TSNE::kept_row_indices.
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::TSNE::kept_row_indices() const
      r = __datamunge__(2542, obj);
    end
    function r = achieved_perplexity (obj)
      % achieved_perplexity  The achieved perplexity for each fitted point after the per-point binary search -- should be close to the constructor's perplexity argument if calibration converged.
      %
      %   r = obj.achieved_perplexity ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::TSNE::achieved_perplexity() const
      r = __datamunge__(2543, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::TSNE::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::TSNE::dimension(std::size_t index) const
      r = __datamunge__(2544, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  C++ datamunge::TSNE::embedding_frame.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::TSNE::embedding_frame() const
      r = __datamunge__(2545, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::TSNE::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::TSNE::summary() const
      r = __datamunge__(2546, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::TSNE::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::TSNE::print_summary() const
      __datamunge__(2547, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  C++ datamunge::TSNE::plot_embedding.
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
      %   C++: datamunge::graphics::RPlot datamunge::TSNE::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2548, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  C++ datamunge::TSNE::plot_embedding_grouped.
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
      %   C++: datamunge::graphics::RPlot datamunge::TSNE::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2549, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::TSNE. Do not edit.
