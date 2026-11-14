classdef PCA < datamunge.NativeObject
  % PCA  SWIG-friendly facade for datamunge::stats::PCA — principal component analysis fit from a DataFrame and a list of numeric feature columns via eigendecomposition of the correlation (default) or covariance matrix.
  %
  %   obj = datamunge.PCA (data, feature_columns)
  %   obj = datamunge.PCA (data, feature_columns, center)
  %   obj = datamunge.PCA (data, feature_columns, center, scale)
  %   obj = datamunge.PCA (data, encoded_feature_columns)
  %   obj = datamunge.PCA (data, encoded_feature_columns, center)
  %   obj = datamunge.PCA (data, encoded_feature_columns, center, scale)
  %
  %   Methods:
  %     feature_names  C++ datamunge::PCA::feature_names.
  %     observations  C++ datamunge::PCA::observations.
  %     num_components  C++ datamunge::PCA::num_components.
  %     kept_row_indices  Row indices (into the DataFrame passed to the constructor) that survived null-dropping, in fitted order -- use to align an external label vector for plot_scores_grouped().
  %     explained_variance  C++ datamunge::PCA::explained_variance.
  %     explained_variance_ratio  C++ datamunge::PCA::explained_variance_ratio.
  %     cumulative_explained_variance_ratio  C++ datamunge::PCA::cumulative_explained_variance_ratio.
  %     component_loadings  C++ datamunge::PCA::component_loadings.
  %     component_scores  C++ datamunge::PCA::component_scores.
  %     scores_frame  The fitted training scores as a DataFrame ("PC1", "PC2", ... columns), one row per kept observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
  %     transform  Projects new data onto the already-fitted components, as a DataFrame with the same "PC1", "PC2", ... columns as scores_frame().
  %     summary  C++ datamunge::PCA::summary.
  %     print_summary  C++ datamunge::PCA::print_summary.
  %     plot_scores  Scatter of scores in the (component_x, component_y) plane, single series.
  %     plot_scores_grouped  Same, colored by an external grouping vector (e.g. a withheld response column); `group_labels` must have one entry per kept_row_indices() entry.
  %     plot_scree  Scree plot: percent of variance explained by each component, as a bar chart.
  %
  %   C++: class datamunge::PCA
  methods
    function obj = PCA (varargin)
      % PCA  Construct a C++ datamunge::PCA.
      %
      %   obj = datamunge.PCA (data, feature_columns)
      %   obj = datamunge.PCA (data, feature_columns, center)
      %   obj = datamunge.PCA (data, feature_columns, center, scale)
      %   obj = datamunge.PCA (data, encoded_feature_columns)
      %   obj = datamunge.PCA (data, encoded_feature_columns, center)
      %   obj = datamunge.PCA (data, encoded_feature_columns, center, scale)
      %
      %   C++: datamunge::PCA::PCA(const DataFrame & data, const std::vector<std::string> & feature_columns, bool center = true, bool scale = true)
      %   C++: datamunge::PCA::PCA(const DataFrame & data, const std::string & encoded_feature_columns, bool center = true, bool scale = true)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2500, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::PCA::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::PCA::feature_names() const
      r = __datamunge__(2484, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::PCA::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::PCA::observations() const
      r = __datamunge__(2485, obj);
    end
    function r = num_components (obj)
      % num_components  C++ datamunge::PCA::num_components.
      %
      %   r = obj.num_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::PCA::num_components() const
      r = __datamunge__(2486, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  Row indices (into the DataFrame passed to the constructor) that survived null-dropping, in fitted order -- use to align an external label vector for plot_scores_grouped().
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::PCA::kept_row_indices() const
      r = __datamunge__(2487, obj);
    end
    function r = explained_variance (obj)
      % explained_variance  C++ datamunge::PCA::explained_variance.
      %
      %   r = obj.explained_variance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::PCA::explained_variance() const
      r = __datamunge__(2488, obj);
    end
    function r = explained_variance_ratio (obj)
      % explained_variance_ratio  C++ datamunge::PCA::explained_variance_ratio.
      %
      %   r = obj.explained_variance_ratio ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::PCA::explained_variance_ratio() const
      r = __datamunge__(2489, obj);
    end
    function r = cumulative_explained_variance_ratio (obj)
      % cumulative_explained_variance_ratio  C++ datamunge::PCA::cumulative_explained_variance_ratio.
      %
      %   r = obj.cumulative_explained_variance_ratio ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::PCA::cumulative_explained_variance_ratio() const
      r = __datamunge__(2490, obj);
    end
    function r = component_loadings (obj, component_index)
      % component_loadings  C++ datamunge::PCA::component_loadings.
      %
      %   r = obj.component_loadings (component_index)
      %
      %   Inputs:
      %     component_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::PCA::component_loadings(std::size_t component_index) const
      r = __datamunge__(2491, obj, component_index);
    end
    function r = component_scores (obj, component_index)
      % component_scores  C++ datamunge::PCA::component_scores.
      %
      %   r = obj.component_scores (component_index)
      %
      %   Inputs:
      %     component_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::PCA::component_scores(std::size_t component_index) const
      r = __datamunge__(2492, obj, component_index);
    end
    function r = scores_frame (obj)
      % scores_frame  The fitted training scores as a DataFrame ("PC1", "PC2", ... columns), one row per kept observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
      %
      %   r = obj.scores_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::PCA::scores_frame() const
      r = __datamunge__(2493, obj);
    end
    function r = transform (obj, newdata)
      % transform  Projects new data onto the already-fitted components, as a DataFrame with the same "PC1", "PC2", ... columns as scores_frame().
      %
      %   r = obj.transform (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::PCA::transform(const DataFrame & newdata) const
      r = __datamunge__(2494, obj, newdata);
    end
    function r = summary (obj)
      % summary  C++ datamunge::PCA::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::PCA::summary() const
      r = __datamunge__(2495, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::PCA::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::PCA::print_summary() const
      __datamunge__(2496, obj);
    end
    function r = plot_scores (obj, varargin)
      % plot_scores  Scatter of scores in the (component_x, component_y) plane, single series.
      %
      %   r = obj.plot_scores ()
      %   r = obj.plot_scores (component_x)
      %   r = obj.plot_scores (component_x, component_y)
      %
      %   Inputs:
      %     component_x  integer, default 0
      %     component_y  integer, default 1
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::PCA::plot_scores(std::size_t component_x = 0, std::size_t component_y = 1) const
      r = __datamunge__(2497, obj, varargin{:});
    end
    function r = plot_scores_grouped (obj, varargin)
      % plot_scores_grouped  Same, colored by an external grouping vector (e.g. a withheld response column); `group_labels` must have one entry per kept_row_indices() entry.
      %
      %   r = obj.plot_scores_grouped (group_labels)
      %   r = obj.plot_scores_grouped (group_labels, component_x)
      %   r = obj.plot_scores_grouped (group_labels, component_x, component_y)
      %
      %   Inputs:
      %     group_labels  cell array of char
      %     component_x  integer, default 0
      %     component_y  integer, default 1
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::PCA::plot_scores_grouped(const std::vector<std::string> & group_labels, std::size_t component_x = 0, std::size_t component_y = 1) const
      r = __datamunge__(2498, obj, varargin{:});
    end
    function r = plot_scree (obj)
      % plot_scree  Scree plot: percent of variance explained by each component, as a bar chart.
      %
      %   r = obj.plot_scree ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::PCA::plot_scree() const
      r = __datamunge__(2499, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::PCA. Do not edit.
