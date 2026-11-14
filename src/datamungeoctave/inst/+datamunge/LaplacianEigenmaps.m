classdef LaplacianEigenmaps < datamunge.NativeObject
  % LaplacianEigenmaps  SWIG-friendly facade for datamunge::stats::LaplacianEigenmaps — embeds points via the smallest non-trivial eigenvectors of a heat-kernel-weighted, sparse k-nearest-neighbor graph Laplacian. Distinct from DiffusionMaps (also heat-kernel-based, but dense and alpha-normalized).
  %
  %   obj = datamunge.LaplacianEigenmaps (data, feature_columns)
  %   obj = datamunge.LaplacianEigenmaps (data, feature_columns, n_components)
  %   obj = datamunge.LaplacianEigenmaps (data, feature_columns, n_components, n_neighbors)
  %   obj = datamunge.LaplacianEigenmaps (data, feature_columns, n_components, n_neighbors, heat_kernel_t)
  %   obj = datamunge.LaplacianEigenmaps (data, encoded_feature_columns)
  %   obj = datamunge.LaplacianEigenmaps (data, encoded_feature_columns, n_components)
  %   obj = datamunge.LaplacianEigenmaps (data, encoded_feature_columns, n_components, n_neighbors)
  %   obj = datamunge.LaplacianEigenmaps (data, encoded_feature_columns, n_components, n_neighbors, heat_kernel_t)
  %
  %   Methods:
  %     feature_names  C++ datamunge::LaplacianEigenmaps::feature_names.
  %     observations  C++ datamunge::LaplacianEigenmaps::observations.
  %     n_components  C++ datamunge::LaplacianEigenmaps::n_components.
  %     kept_row_indices  C++ datamunge::LaplacianEigenmaps::kept_row_indices.
  %     eigenvalues  C++ datamunge::LaplacianEigenmaps::eigenvalues.
  %     dimension  C++ datamunge::LaplacianEigenmaps::dimension.
  %     embedding_frame  C++ datamunge::LaplacianEigenmaps::embedding_frame.
  %     summary  C++ datamunge::LaplacianEigenmaps::summary.
  %     print_summary  C++ datamunge::LaplacianEigenmaps::print_summary.
  %     plot_embedding  C++ datamunge::LaplacianEigenmaps::plot_embedding.
  %     plot_embedding_grouped  C++ datamunge::LaplacianEigenmaps::plot_embedding_grouped.
  %
  %   C++: class datamunge::LaplacianEigenmaps
  methods
    function obj = LaplacianEigenmaps (varargin)
      % LaplacianEigenmaps  Construct a C++ datamunge::LaplacianEigenmaps.
      %
      %   obj = datamunge.LaplacianEigenmaps (data, feature_columns)
      %   obj = datamunge.LaplacianEigenmaps (data, feature_columns, n_components)
      %   obj = datamunge.LaplacianEigenmaps (data, feature_columns, n_components, n_neighbors)
      %   obj = datamunge.LaplacianEigenmaps (data, feature_columns, n_components, n_neighbors, heat_kernel_t)
      %   obj = datamunge.LaplacianEigenmaps (data, encoded_feature_columns)
      %   obj = datamunge.LaplacianEigenmaps (data, encoded_feature_columns, n_components)
      %   obj = datamunge.LaplacianEigenmaps (data, encoded_feature_columns, n_components, n_neighbors)
      %   obj = datamunge.LaplacianEigenmaps (data, encoded_feature_columns, n_components, n_neighbors, heat_kernel_t)
      %
      %   Same as the vector<string> constructor, but feature_columns is a single "<count> col1..."-encoded string -- see KMeans's encoded constructor for why.
      %
      %   C++: datamunge::LaplacianEigenmaps::LaplacianEigenmaps(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, std::size_t n_neighbors = 10, double heat_kernel_t = 1.0)
      %   C++: datamunge::LaplacianEigenmaps::LaplacianEigenmaps(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, std::size_t n_neighbors = 10, double heat_kernel_t = 1.0)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2562, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::LaplacianEigenmaps::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LaplacianEigenmaps::feature_names() const
      r = __datamunge__(2551, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::LaplacianEigenmaps::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LaplacianEigenmaps::observations() const
      r = __datamunge__(2552, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::LaplacianEigenmaps::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LaplacianEigenmaps::n_components() const
      r = __datamunge__(2553, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  C++ datamunge::LaplacianEigenmaps::kept_row_indices.
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::LaplacianEigenmaps::kept_row_indices() const
      r = __datamunge__(2554, obj);
    end
    function r = eigenvalues (obj)
      % eigenvalues  C++ datamunge::LaplacianEigenmaps::eigenvalues.
      %
      %   r = obj.eigenvalues ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LaplacianEigenmaps::eigenvalues() const
      r = __datamunge__(2555, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::LaplacianEigenmaps::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LaplacianEigenmaps::dimension(std::size_t index) const
      r = __datamunge__(2556, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  C++ datamunge::LaplacianEigenmaps::embedding_frame.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::LaplacianEigenmaps::embedding_frame() const
      r = __datamunge__(2557, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::LaplacianEigenmaps::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LaplacianEigenmaps::summary() const
      r = __datamunge__(2558, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::LaplacianEigenmaps::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::LaplacianEigenmaps::print_summary() const
      __datamunge__(2559, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  C++ datamunge::LaplacianEigenmaps::plot_embedding.
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
      %   C++: datamunge::graphics::RPlot datamunge::LaplacianEigenmaps::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2560, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  C++ datamunge::LaplacianEigenmaps::plot_embedding_grouped.
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
      %   C++: datamunge::graphics::RPlot datamunge::LaplacianEigenmaps::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2561, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::LaplacianEigenmaps. Do not edit.
