classdef Isomap < datamunge.NativeObject
  % Isomap  SWIG-friendly facade for datamunge::stats::Isomap — nonlinear manifold learning via geodesic (shortest-path, over a k-nearest-neighbor graph) distances embedded with classical MDS. Throws if the k-nearest-neighbor graph is disconnected.
  %
  %   obj = datamunge.Isomap (data, feature_columns)
  %   obj = datamunge.Isomap (data, feature_columns, n_components)
  %   obj = datamunge.Isomap (data, feature_columns, n_components, n_neighbors)
  %   obj = datamunge.Isomap (data, feature_columns, n_components, n_neighbors, metric)
  %   obj = datamunge.Isomap (data, encoded_feature_columns)
  %   obj = datamunge.Isomap (data, encoded_feature_columns, n_components)
  %   obj = datamunge.Isomap (data, encoded_feature_columns, n_components, n_neighbors)
  %   obj = datamunge.Isomap (data, encoded_feature_columns, n_components, n_neighbors, metric)
  %
  %   Methods:
  %     feature_names  C++ datamunge::Isomap::feature_names.
  %     observations  C++ datamunge::Isomap::observations.
  %     n_components  C++ datamunge::Isomap::n_components.
  %     kept_row_indices  C++ datamunge::Isomap::kept_row_indices.
  %     eigenvalues  C++ datamunge::Isomap::eigenvalues.
  %     goodness_of_fit  C++ datamunge::Isomap::goodness_of_fit.
  %     dimension  C++ datamunge::Isomap::dimension.
  %     embedding_frame  The fitted embedding as a DataFrame ("Dim1", "Dim2", ... columns), one row per kept observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
  %     summary  C++ datamunge::Isomap::summary.
  %     print_summary  C++ datamunge::Isomap::print_summary.
  %     plot_embedding  C++ datamunge::Isomap::plot_embedding.
  %     plot_embedding_grouped  C++ datamunge::Isomap::plot_embedding_grouped.
  %
  %   C++: class datamunge::Isomap
  methods
    function obj = Isomap (varargin)
      % Isomap  Construct a C++ datamunge::Isomap.
      %
      %   obj = datamunge.Isomap (data, feature_columns)
      %   obj = datamunge.Isomap (data, feature_columns, n_components)
      %   obj = datamunge.Isomap (data, feature_columns, n_components, n_neighbors)
      %   obj = datamunge.Isomap (data, feature_columns, n_components, n_neighbors, metric)
      %   obj = datamunge.Isomap (data, encoded_feature_columns)
      %   obj = datamunge.Isomap (data, encoded_feature_columns, n_components)
      %   obj = datamunge.Isomap (data, encoded_feature_columns, n_components, n_neighbors)
      %   obj = datamunge.Isomap (data, encoded_feature_columns, n_components, n_neighbors, metric)
      %
      %   C++: datamunge::Isomap::Isomap(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, std::size_t n_neighbors = 10, const std::string & metric = "euclidean")
      %   C++: datamunge::Isomap::Isomap(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, std::size_t n_neighbors = 10, const std::string & metric = "euclidean")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2526, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::Isomap::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::Isomap::feature_names() const
      r = __datamunge__(2514, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::Isomap::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Isomap::observations() const
      r = __datamunge__(2515, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::Isomap::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Isomap::n_components() const
      r = __datamunge__(2516, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  C++ datamunge::Isomap::kept_row_indices.
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::Isomap::kept_row_indices() const
      r = __datamunge__(2517, obj);
    end
    function r = eigenvalues (obj)
      % eigenvalues  C++ datamunge::Isomap::eigenvalues.
      %
      %   r = obj.eigenvalues ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Isomap::eigenvalues() const
      r = __datamunge__(2518, obj);
    end
    function r = goodness_of_fit (obj)
      % goodness_of_fit  C++ datamunge::Isomap::goodness_of_fit.
      %
      %   r = obj.goodness_of_fit ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Isomap::goodness_of_fit() const
      r = __datamunge__(2519, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::Isomap::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Isomap::dimension(std::size_t index) const
      r = __datamunge__(2520, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  The fitted embedding as a DataFrame ("Dim1", "Dim2", ... columns), one row per kept observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::Isomap::embedding_frame() const
      r = __datamunge__(2521, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::Isomap::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Isomap::summary() const
      r = __datamunge__(2522, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::Isomap::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::Isomap::print_summary() const
      __datamunge__(2523, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  C++ datamunge::Isomap::plot_embedding.
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
      %   C++: datamunge::graphics::RPlot datamunge::Isomap::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2524, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  C++ datamunge::Isomap::plot_embedding_grouped.
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
      %   C++: datamunge::graphics::RPlot datamunge::Isomap::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2525, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::Isomap. Do not edit.
