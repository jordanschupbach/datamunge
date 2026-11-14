classdef LLE < datamunge.NativeObject
  % LLE  SWIG-friendly facade for datamunge::stats::LLE — locally linear embedding: preserves each point's local reconstruction weights from its k nearest neighbors rather than global distances, letting it unfold nonlinear manifolds PCA/MDS cannot.
  %
  %   obj = datamunge.LLE (data, feature_columns)
  %   obj = datamunge.LLE (data, feature_columns, n_components)
  %   obj = datamunge.LLE (data, feature_columns, n_components, n_neighbors)
  %   obj = datamunge.LLE (data, feature_columns, n_components, n_neighbors, regularization)
  %   obj = datamunge.LLE (data, feature_columns, n_components, n_neighbors, regularization, metric)
  %   obj = datamunge.LLE (data, encoded_feature_columns)
  %   obj = datamunge.LLE (data, encoded_feature_columns, n_components)
  %   obj = datamunge.LLE (data, encoded_feature_columns, n_components, n_neighbors)
  %   obj = datamunge.LLE (data, encoded_feature_columns, n_components, n_neighbors, regularization)
  %   obj = datamunge.LLE (data, encoded_feature_columns, n_components, n_neighbors, regularization, metric)
  %
  %   Methods:
  %     feature_names  C++ datamunge::LLE::feature_names.
  %     observations  C++ datamunge::LLE::observations.
  %     n_components  C++ datamunge::LLE::n_components.
  %     kept_row_indices  C++ datamunge::LLE::kept_row_indices.
  %     eigenvalues  C++ datamunge::LLE::eigenvalues.
  %     dimension  C++ datamunge::LLE::dimension.
  %     embedding_frame  C++ datamunge::LLE::embedding_frame.
  %     summary  C++ datamunge::LLE::summary.
  %     print_summary  C++ datamunge::LLE::print_summary.
  %     plot_embedding  C++ datamunge::LLE::plot_embedding.
  %     plot_embedding_grouped  C++ datamunge::LLE::plot_embedding_grouped.
  %
  %   C++: class datamunge::LLE
  methods
    function obj = LLE (varargin)
      % LLE  Construct a C++ datamunge::LLE.
      %
      %   obj = datamunge.LLE (data, feature_columns)
      %   obj = datamunge.LLE (data, feature_columns, n_components)
      %   obj = datamunge.LLE (data, feature_columns, n_components, n_neighbors)
      %   obj = datamunge.LLE (data, feature_columns, n_components, n_neighbors, regularization)
      %   obj = datamunge.LLE (data, feature_columns, n_components, n_neighbors, regularization, metric)
      %   obj = datamunge.LLE (data, encoded_feature_columns)
      %   obj = datamunge.LLE (data, encoded_feature_columns, n_components)
      %   obj = datamunge.LLE (data, encoded_feature_columns, n_components, n_neighbors)
      %   obj = datamunge.LLE (data, encoded_feature_columns, n_components, n_neighbors, regularization)
      %   obj = datamunge.LLE (data, encoded_feature_columns, n_components, n_neighbors, regularization, metric)
      %
      %   C++: datamunge::LLE::LLE(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, std::size_t n_neighbors = 10, double regularization = 0.001, const std::string & metric = "euclidean")
      %   C++: datamunge::LLE::LLE(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, std::size_t n_neighbors = 10, double regularization = 0.001, const std::string & metric = "euclidean")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2538, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::LLE::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LLE::feature_names() const
      r = __datamunge__(2527, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::LLE::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LLE::observations() const
      r = __datamunge__(2528, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::LLE::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LLE::n_components() const
      r = __datamunge__(2529, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  C++ datamunge::LLE::kept_row_indices.
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::LLE::kept_row_indices() const
      r = __datamunge__(2530, obj);
    end
    function r = eigenvalues (obj)
      % eigenvalues  C++ datamunge::LLE::eigenvalues.
      %
      %   r = obj.eigenvalues ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LLE::eigenvalues() const
      r = __datamunge__(2531, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::LLE::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LLE::dimension(std::size_t index) const
      r = __datamunge__(2532, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  C++ datamunge::LLE::embedding_frame.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::LLE::embedding_frame() const
      r = __datamunge__(2533, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::LLE::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LLE::summary() const
      r = __datamunge__(2534, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::LLE::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::LLE::print_summary() const
      __datamunge__(2535, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  C++ datamunge::LLE::plot_embedding.
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
      %   C++: datamunge::graphics::RPlot datamunge::LLE::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2536, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  C++ datamunge::LLE::plot_embedding_grouped.
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
      %   C++: datamunge::graphics::RPlot datamunge::LLE::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2537, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::LLE. Do not edit.
