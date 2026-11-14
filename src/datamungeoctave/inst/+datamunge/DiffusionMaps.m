classdef DiffusionMaps < datamunge.NativeObject
  % DiffusionMaps  SWIG-friendly facade for datamunge::stats::DiffusionMaps — embeds points via the leading eigenvectors of a dense, alpha-normalized heat-kernel Markov transition matrix, scaled by eigenvalue^diffusion_time. Distinct from LaplacianEigenmaps (sparse k-NN graph Laplacian, no density normalization).
  %
  %   obj = datamunge.DiffusionMaps (data, feature_columns)
  %   obj = datamunge.DiffusionMaps (data, feature_columns, n_components)
  %   obj = datamunge.DiffusionMaps (data, feature_columns, n_components, heat_kernel_epsilon)
  %   obj = datamunge.DiffusionMaps (data, feature_columns, n_components, heat_kernel_epsilon, alpha)
  %   obj = datamunge.DiffusionMaps (data, feature_columns, n_components, heat_kernel_epsilon, alpha, diffusion_time)
  %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns)
  %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns, n_components)
  %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns, n_components, heat_kernel_epsilon)
  %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns, n_components, heat_kernel_epsilon, alpha)
  %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns, n_components, heat_kernel_epsilon, alpha, diffusion_time)
  %
  %   Methods:
  %     feature_names  C++ datamunge::DiffusionMaps::feature_names.
  %     observations  C++ datamunge::DiffusionMaps::observations.
  %     n_components  C++ datamunge::DiffusionMaps::n_components.
  %     kept_row_indices  C++ datamunge::DiffusionMaps::kept_row_indices.
  %     eigenvalues  C++ datamunge::DiffusionMaps::eigenvalues.
  %     dimension  C++ datamunge::DiffusionMaps::dimension.
  %     embedding_frame  C++ datamunge::DiffusionMaps::embedding_frame.
  %     summary  C++ datamunge::DiffusionMaps::summary.
  %     print_summary  C++ datamunge::DiffusionMaps::print_summary.
  %     plot_embedding  C++ datamunge::DiffusionMaps::plot_embedding.
  %     plot_embedding_grouped  C++ datamunge::DiffusionMaps::plot_embedding_grouped.
  %
  %   C++: class datamunge::DiffusionMaps
  methods
    function obj = DiffusionMaps (varargin)
      % DiffusionMaps  Construct a C++ datamunge::DiffusionMaps.
      %
      %   obj = datamunge.DiffusionMaps (data, feature_columns)
      %   obj = datamunge.DiffusionMaps (data, feature_columns, n_components)
      %   obj = datamunge.DiffusionMaps (data, feature_columns, n_components, heat_kernel_epsilon)
      %   obj = datamunge.DiffusionMaps (data, feature_columns, n_components, heat_kernel_epsilon, alpha)
      %   obj = datamunge.DiffusionMaps (data, feature_columns, n_components, heat_kernel_epsilon, alpha, diffusion_time)
      %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns)
      %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns, n_components)
      %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns, n_components, heat_kernel_epsilon)
      %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns, n_components, heat_kernel_epsilon, alpha)
      %   obj = datamunge.DiffusionMaps (data, encoded_feature_columns, n_components, heat_kernel_epsilon, alpha, diffusion_time)
      %
      %   Same as the vector<string> constructor, but feature_columns is a single "<count> col1..."-encoded string -- see KMeans's encoded constructor for why.
      %
      %   C++: datamunge::DiffusionMaps::DiffusionMaps(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, double heat_kernel_epsilon = 1.0, double alpha = 0.5, double diffusion_time = 1.0)
      %   C++: datamunge::DiffusionMaps::DiffusionMaps(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, double heat_kernel_epsilon = 1.0, double alpha = 0.5, double diffusion_time = 1.0)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2574, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::DiffusionMaps::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::DiffusionMaps::feature_names() const
      r = __datamunge__(2563, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::DiffusionMaps::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DiffusionMaps::observations() const
      r = __datamunge__(2564, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::DiffusionMaps::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DiffusionMaps::n_components() const
      r = __datamunge__(2565, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  C++ datamunge::DiffusionMaps::kept_row_indices.
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::DiffusionMaps::kept_row_indices() const
      r = __datamunge__(2566, obj);
    end
    function r = eigenvalues (obj)
      % eigenvalues  C++ datamunge::DiffusionMaps::eigenvalues.
      %
      %   r = obj.eigenvalues ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::DiffusionMaps::eigenvalues() const
      r = __datamunge__(2567, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::DiffusionMaps::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::DiffusionMaps::dimension(std::size_t index) const
      r = __datamunge__(2568, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  C++ datamunge::DiffusionMaps::embedding_frame.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DiffusionMaps::embedding_frame() const
      r = __datamunge__(2569, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::DiffusionMaps::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::DiffusionMaps::summary() const
      r = __datamunge__(2570, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::DiffusionMaps::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::DiffusionMaps::print_summary() const
      __datamunge__(2571, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  C++ datamunge::DiffusionMaps::plot_embedding.
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
      %   C++: datamunge::graphics::RPlot datamunge::DiffusionMaps::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2572, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  C++ datamunge::DiffusionMaps::plot_embedding_grouped.
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
      %   C++: datamunge::graphics::RPlot datamunge::DiffusionMaps::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2573, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::DiffusionMaps. Do not edit.
