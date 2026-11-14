classdef MDS < datamunge.NativeObject
  % MDS  SWIG-friendly facade for datamunge::stats::MDS — classical (metric/Torgerson) multidimensional scaling fit from a DataFrame and a list of numeric feature columns: embeds the rows in a low-dimensional space that best reproduces their original pairwise distances.
  %
  %   obj = datamunge.MDS (data, feature_columns)
  %   obj = datamunge.MDS (data, feature_columns, n_components)
  %   obj = datamunge.MDS (data, feature_columns, n_components, metric)
  %   obj = datamunge.MDS (data, encoded_feature_columns)
  %   obj = datamunge.MDS (data, encoded_feature_columns, n_components)
  %   obj = datamunge.MDS (data, encoded_feature_columns, n_components, metric)
  %
  %   Methods:
  %     feature_names  C++ datamunge::MDS::feature_names.
  %     observations  C++ datamunge::MDS::observations.
  %     n_components  C++ datamunge::MDS::n_components.
  %     kept_row_indices  Row indices (into the DataFrame passed to the constructor) that survived null-dropping, in fitted order -- use to align an external label vector for plot_embedding_grouped().
  %     eigenvalues  C++ datamunge::MDS::eigenvalues.
  %     goodness_of_fit  C++ datamunge::MDS::goodness_of_fit.
  %     dimension  C++ datamunge::MDS::dimension.
  %     embedding_frame  The fitted embedding as a DataFrame ("Dim1", "Dim2", ... columns), one row per kept observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
  %     summary  C++ datamunge::MDS::summary.
  %     print_summary  C++ datamunge::MDS::print_summary.
  %     plot_embedding  Scatter of the embedding in the (dimension_x, dimension_y) plane, single series.
  %     plot_embedding_grouped  Same, colored by an external grouping vector (e.g. a withheld response column); `group_labels` must have one entry per kept_row_indices() entry.
  %
  %   C++: class datamunge::MDS
  methods
    function obj = MDS (varargin)
      % MDS  Construct a C++ datamunge::MDS.
      %
      %   obj = datamunge.MDS (data, feature_columns)
      %   obj = datamunge.MDS (data, feature_columns, n_components)
      %   obj = datamunge.MDS (data, feature_columns, n_components, metric)
      %   obj = datamunge.MDS (data, encoded_feature_columns)
      %   obj = datamunge.MDS (data, encoded_feature_columns, n_components)
      %   obj = datamunge.MDS (data, encoded_feature_columns, n_components, metric)
      %
      %   C++: datamunge::MDS::MDS(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, const std::string & metric = "euclidean")
      %   C++: datamunge::MDS::MDS(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, const std::string & metric = "euclidean")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2513, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::MDS::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::MDS::feature_names() const
      r = __datamunge__(2501, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::MDS::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::MDS::observations() const
      r = __datamunge__(2502, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::MDS::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::MDS::n_components() const
      r = __datamunge__(2503, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  Row indices (into the DataFrame passed to the constructor) that survived null-dropping, in fitted order -- use to align an external label vector for plot_embedding_grouped().
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::MDS::kept_row_indices() const
      r = __datamunge__(2504, obj);
    end
    function r = eigenvalues (obj)
      % eigenvalues  C++ datamunge::MDS::eigenvalues.
      %
      %   r = obj.eigenvalues ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::MDS::eigenvalues() const
      r = __datamunge__(2505, obj);
    end
    function r = goodness_of_fit (obj)
      % goodness_of_fit  C++ datamunge::MDS::goodness_of_fit.
      %
      %   r = obj.goodness_of_fit ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::MDS::goodness_of_fit() const
      r = __datamunge__(2506, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::MDS::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::MDS::dimension(std::size_t index) const
      r = __datamunge__(2507, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  The fitted embedding as a DataFrame ("Dim1", "Dim2", ... columns), one row per kept observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::MDS::embedding_frame() const
      r = __datamunge__(2508, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::MDS::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::MDS::summary() const
      r = __datamunge__(2509, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::MDS::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::MDS::print_summary() const
      __datamunge__(2510, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  Scatter of the embedding in the (dimension_x, dimension_y) plane, single series.
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
      %   C++: datamunge::graphics::RPlot datamunge::MDS::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2511, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  Same, colored by an external grouping vector (e.g. a withheld response column); `group_labels` must have one entry per kept_row_indices() entry.
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
      %   C++: datamunge::graphics::RPlot datamunge::MDS::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2512, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::MDS. Do not edit.
