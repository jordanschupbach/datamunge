classdef KernelPCA < datamunge.NativeObject
  % KernelPCA  SWIG-friendly facade for datamunge::stats::KernelPCA — principal component analysis generalized to a nonlinear feature space via the kernel trick. With a linear kernel it reproduces plain unscaled PCA's scores (up to a sign flip per component).
  %
  %   obj = datamunge.KernelPCA (data, feature_columns)
  %   obj = datamunge.KernelPCA (data, feature_columns, n_components)
  %   obj = datamunge.KernelPCA (data, feature_columns, n_components, kernel)
  %   obj = datamunge.KernelPCA (data, feature_columns, n_components, kernel, gamma)
  %   obj = datamunge.KernelPCA (data, feature_columns, n_components, kernel, gamma, degree)
  %   obj = datamunge.KernelPCA (data, feature_columns, n_components, kernel, gamma, degree, coef0)
  %   obj = datamunge.KernelPCA (data, encoded_feature_columns)
  %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components)
  %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components, kernel)
  %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components, kernel, gamma)
  %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components, kernel, gamma, degree)
  %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components, kernel, gamma, degree, coef0)
  %
  %   Methods:
  %     feature_names  C++ datamunge::KernelPCA::feature_names.
  %     observations  C++ datamunge::KernelPCA::observations.
  %     n_components  C++ datamunge::KernelPCA::n_components.
  %     kept_row_indices  C++ datamunge::KernelPCA::kept_row_indices.
  %     eigenvalues  C++ datamunge::KernelPCA::eigenvalues.
  %     dimension  C++ datamunge::KernelPCA::dimension.
  %     embedding_frame  C++ datamunge::KernelPCA::embedding_frame.
  %     summary  C++ datamunge::KernelPCA::summary.
  %     print_summary  C++ datamunge::KernelPCA::print_summary.
  %     plot_embedding  C++ datamunge::KernelPCA::plot_embedding.
  %     plot_embedding_grouped  C++ datamunge::KernelPCA::plot_embedding_grouped.
  %
  %   C++: class datamunge::KernelPCA
  methods
    function obj = KernelPCA (varargin)
      % KernelPCA  Construct a C++ datamunge::KernelPCA.
      %
      %   obj = datamunge.KernelPCA (data, feature_columns)
      %   obj = datamunge.KernelPCA (data, feature_columns, n_components)
      %   obj = datamunge.KernelPCA (data, feature_columns, n_components, kernel)
      %   obj = datamunge.KernelPCA (data, feature_columns, n_components, kernel, gamma)
      %   obj = datamunge.KernelPCA (data, feature_columns, n_components, kernel, gamma, degree)
      %   obj = datamunge.KernelPCA (data, feature_columns, n_components, kernel, gamma, degree, coef0)
      %   obj = datamunge.KernelPCA (data, encoded_feature_columns)
      %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components)
      %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components, kernel)
      %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components, kernel, gamma)
      %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components, kernel, gamma, degree)
      %   obj = datamunge.KernelPCA (data, encoded_feature_columns, n_components, kernel, gamma, degree, coef0)
      %
      %   C++: datamunge::KernelPCA::KernelPCA(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_components = 2, const std::string & kernel = "rbf", double gamma = 1.0, double degree = 3.0, double coef0 = 1.0)
      %   C++: datamunge::KernelPCA::KernelPCA(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_components = 2, const std::string & kernel = "rbf", double gamma = 1.0, double degree = 3.0, double coef0 = 1.0)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2586, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::KernelPCA::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::KernelPCA::feature_names() const
      r = __datamunge__(2575, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::KernelPCA::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KernelPCA::observations() const
      r = __datamunge__(2576, obj);
    end
    function r = n_components (obj)
      % n_components  C++ datamunge::KernelPCA::n_components.
      %
      %   r = obj.n_components ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KernelPCA::n_components() const
      r = __datamunge__(2577, obj);
    end
    function r = kept_row_indices (obj)
      % kept_row_indices  C++ datamunge::KernelPCA::kept_row_indices.
      %
      %   r = obj.kept_row_indices ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::KernelPCA::kept_row_indices() const
      r = __datamunge__(2578, obj);
    end
    function r = eigenvalues (obj)
      % eigenvalues  C++ datamunge::KernelPCA::eigenvalues.
      %
      %   r = obj.eigenvalues ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KernelPCA::eigenvalues() const
      r = __datamunge__(2579, obj);
    end
    function r = dimension (obj, index)
      % dimension  C++ datamunge::KernelPCA::dimension.
      %
      %   r = obj.dimension (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KernelPCA::dimension(std::size_t index) const
      r = __datamunge__(2580, obj, index);
    end
    function r = embedding_frame (obj)
      % embedding_frame  C++ datamunge::KernelPCA::embedding_frame.
      %
      %   r = obj.embedding_frame ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::KernelPCA::embedding_frame() const
      r = __datamunge__(2581, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::KernelPCA::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::KernelPCA::summary() const
      r = __datamunge__(2582, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::KernelPCA::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::KernelPCA::print_summary() const
      __datamunge__(2583, obj);
    end
    function r = plot_embedding (obj, varargin)
      % plot_embedding  C++ datamunge::KernelPCA::plot_embedding.
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
      %   C++: datamunge::graphics::RPlot datamunge::KernelPCA::plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2584, obj, varargin{:});
    end
    function r = plot_embedding_grouped (obj, varargin)
      % plot_embedding_grouped  C++ datamunge::KernelPCA::plot_embedding_grouped.
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
      %   C++: datamunge::graphics::RPlot datamunge::KernelPCA::plot_embedding_grouped(const std::vector<std::string> & group_labels, std::size_t dimension_x = 0, std::size_t dimension_y = 1) const
      r = __datamunge__(2585, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::KernelPCA. Do not edit.
