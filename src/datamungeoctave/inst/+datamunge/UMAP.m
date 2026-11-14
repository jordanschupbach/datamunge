classdef UMAP < datamunge.NativeObject
  % UMAP  SWIG-friendly facade for datamunge::stats::UMAP — builds a fuzzy simplicial set (a smoothly-calibrated, fuzzy-union-symmetrized k-nearest-neighbor graph) and optimizes a low-dimensional embedding via cross-entropy minimization (attractive + negative-sampled repulsive forces). A deliberately simplified but honest implementation -- see the real class's own documentation for the two named simplifications.
  %
  %   obj = datamunge.UMAP (data, feature_columns)
  %   obj = datamunge.UMAP (data, feature_columns, n_components)
  %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors)
  %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist)
  %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations)
  %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate)
  %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate)
  %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate, metric)
  %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate, metric, seed)
  %   obj = datamunge.UMAP (data, encoded_feature_columns)
  %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components)
  %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors)
  %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist)
  %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations)
  %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate)
  %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate)
  %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate, metric)
  %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate, metric, seed)
  %
  %   Methods:
  %     feature_names  C++ datamunge::UMAP::feature_names.
  %     observations  C++ datamunge::UMAP::observations.
  %     n_components  C++ datamunge::UMAP::n_components.
  %     kept_row_indices  C++ datamunge::UMAP::kept_row_indices.
  %     dimension  C++ datamunge::UMAP::dimension.
  %     embedding_frame  C++ datamunge::UMAP::embedding_frame.
  %     sigmas  Per-point sigma_i / rho_i from the smooth k-NN calibration -- mainly for diagnostics.
  %     rhos  C++ datamunge::UMAP::rhos.
  %     summary  C++ datamunge::UMAP::summary.
  %     print_summary  C++ datamunge::UMAP::print_summary.
  %     plot_embedding  C++ datamunge::UMAP::plot_embedding.
  %     plot_embedding_grouped  C++ datamunge::UMAP::plot_embedding_grouped.
  %
  %   C++: class datamunge::UMAP
  methods
    function obj = UMAP (varargin)
      % UMAP  Construct a C++ datamunge::UMAP.
      %
      %   obj = datamunge.UMAP (data, feature_columns)
      %   obj = datamunge.UMAP (data, feature_columns, n_components)
      %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors)
      %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist)
      %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations)
      %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate)
      %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate)
      %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate, metric)
      %   obj = datamunge.UMAP (data, feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate, metric, seed)
      %   obj = datamunge.UMAP (data, encoded_feature_columns)
      %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components)
      %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors)
      %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist)
      %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations)
      %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate)
      %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate)
      %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate, metric)
      %   obj = datamunge.UMAP (data, encoded_feature_columns, n_components, n_neighbors, min_dist, max_iterations, learning_rate, negative_sample_rate, metric, seed)
      %
      %   C++: datamunge::UMAP::UMAP(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, std::size_t n_neighbors = 15, double min_dist = 0.10000000000000001, std::size_t max_iterations = 500, double learning_rate = 1.0, double negative_sample_rate = 5.0, const std::string & metric = "euclidean", std::uint64_t seed = 42)
      %   C++: datamunge::UMAP::UMAP(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, std::size_t n_neighbors = 15, double min_dist = 0.10000000000000001, std::size_t max_iterations = 500, double learning_rate = 1.0, double negative_sample_rate = 5.0, const std::string & metric = "euclidean", std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2612, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::UMAP::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::UMAP::feature_names() const
      r = __datamunge__(2600, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::UMAP::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::UMAP::observations() const
      r = __datamunge__(2601, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::UMAP::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::UMAP::n_components() const
      r = __datamunge__(2602, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  C++ datamunge::UMAP::kept_row_indices.
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::UMAP::kept_row_indices() const
      r = __datamunge__(2603, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::UMAP::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::UMAP::dimension(std::size_t index) const
      r = __datamunge__(2604, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  C++ datamunge::UMAP::embedding_frame.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::UMAP::embedding_frame() const
      r = __datamunge__(2605, obj);
    end
    function r = sigmas (obj)
      % sigmas  Per-point sigma_i / rho_i from the smooth k-NN calibration -- mainly for diagnostics.
      %
      %   r = obj.sigmas ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::UMAP::sigmas() const
      r = __datamunge__(2606, obj);
    end
    function r = rhos (obj)
      % rhos  C++ datamunge::UMAP::rhos.
      %
      %   r = obj.rhos ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::UMAP::rhos() const
      r = __datamunge__(2607, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::UMAP::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::UMAP::summary() const
      r = __datamunge__(2608, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::UMAP::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::UMAP::print_summary() const
      __datamunge__(2609, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  C++ datamunge::UMAP::plot_embedding.
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
      %   C++: datamunge::graphics::RPlot datamunge::UMAP::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2610, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  C++ datamunge::UMAP::plot_embedding_grouped.
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
      %   C++: datamunge::graphics::RPlot datamunge::UMAP::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2611, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::UMAP. Do not edit.
