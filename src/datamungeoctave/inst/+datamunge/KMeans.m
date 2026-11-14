classdef KMeans < datamunge.NativeObject
  % KMeans  SWIG-friendly facade for datamunge::stats::KMeans — k-means clustering (Lloyd's algorithm with k-means++ initialization) fit from a DataFrame and a list of numeric feature columns.
  %
  %   obj = datamunge.KMeans (data, feature_columns)
  %   obj = datamunge.KMeans (data, feature_columns, n_clusters)
  %   obj = datamunge.KMeans (data, feature_columns, n_clusters, max_iterations)
  %   obj = datamunge.KMeans (data, feature_columns, n_clusters, max_iterations, n_init)
  %   obj = datamunge.KMeans (data, feature_columns, n_clusters, max_iterations, n_init, tolerance)
  %   obj = datamunge.KMeans (data, feature_columns, n_clusters, max_iterations, n_init, tolerance, seed)
  %   obj = datamunge.KMeans (data, encoded_feature_columns)
  %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters)
  %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters, max_iterations)
  %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters, max_iterations, n_init)
  %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters, max_iterations, n_init, tolerance)
  %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters, max_iterations, n_init, tolerance, seed)
  %
  %   Methods:
  %     feature_names  C++ datamunge::KMeans::feature_names.
  %     n_clusters  C++ datamunge::KMeans::n_clusters.
  %     observations  C++ datamunge::KMeans::observations.
  %     iterations_used  C++ datamunge::KMeans::iterations_used.
  %     labels  C++ datamunge::KMeans::labels.
  %     inertia  C++ datamunge::KMeans::inertia.
  %     cluster_center  The feature vector of cluster `cluster_index's` center.
  %     predict  C++ datamunge::KMeans::predict.
  %     summary  C++ datamunge::KMeans::summary.
  %     print_summary  C++ datamunge::KMeans::print_summary.
  %
  %   C++: class datamunge::KMeans
  methods
    function obj = KMeans (varargin)
      % KMeans  Construct a C++ datamunge::KMeans.
      %
      %   obj = datamunge.KMeans (data, feature_columns)
      %   obj = datamunge.KMeans (data, feature_columns, n_clusters)
      %   obj = datamunge.KMeans (data, feature_columns, n_clusters, max_iterations)
      %   obj = datamunge.KMeans (data, feature_columns, n_clusters, max_iterations, n_init)
      %   obj = datamunge.KMeans (data, feature_columns, n_clusters, max_iterations, n_init, tolerance)
      %   obj = datamunge.KMeans (data, feature_columns, n_clusters, max_iterations, n_init, tolerance, seed)
      %   obj = datamunge.KMeans (data, encoded_feature_columns)
      %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters)
      %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters, max_iterations)
      %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters, max_iterations, n_init)
      %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters, max_iterations, n_init, tolerance)
      %   obj = datamunge.KMeans (data, encoded_feature_columns, n_clusters, max_iterations, n_init, tolerance, seed)
      %
      %   Same as the vector<string> constructor, but feature_columns is a single "<count> col1..."-encoded string -- a workaround for SWIG-bound languages whose overload resolution can't pass a real string vector to a constructor with other default arguments (see DataFrame::add_string_column_encoded()).
      %
      %   C++: datamunge::KMeans::KMeans(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_clusters = 8, std::size_t max_iterations = 300, std::size_t n_init = 10, double tolerance = 1.0E-4, std::size_t seed = 42)
      %   C++: datamunge::KMeans::KMeans(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_clusters = 8, std::size_t max_iterations = 300, std::size_t n_init = 10, double tolerance = 1.0E-4, std::size_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2483, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::KMeans::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::KMeans::feature_names() const
      r = __datamunge__(2473, obj);
    end
    function r = n_clusters (obj)
      % n_clusters  C++ datamunge::KMeans::n_clusters.
      %
      %   r = obj.n_clusters ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KMeans::n_clusters() const
      r = __datamunge__(2474, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::KMeans::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KMeans::observations() const
      r = __datamunge__(2475, obj);
    end
    function r = iterations_used (obj)
      % iterations_used  C++ datamunge::KMeans::iterations_used.
      %
      %   r = obj.iterations_used ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KMeans::iterations_used() const
      r = __datamunge__(2476, obj);
    end
    function r = labels (obj)
      % labels  C++ datamunge::KMeans::labels.
      %
      %   r = obj.labels ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::KMeans::labels() const
      r = __datamunge__(2477, obj);
    end
    function r = inertia (obj)
      % inertia  C++ datamunge::KMeans::inertia.
      %
      %   r = obj.inertia ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::KMeans::inertia() const
      r = __datamunge__(2478, obj);
    end
    function r = cluster_center (obj, cluster_index)
      % cluster_center  The feature vector of cluster `cluster_index's` center.
      %
      %   r = obj.cluster_center (cluster_index)
      %
      %   Inputs:
      %     cluster_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KMeans::cluster_center(std::size_t cluster_index) const
      r = __datamunge__(2479, obj, cluster_index);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::KMeans::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::KMeans::predict(const DataFrame & newdata) const
      r = __datamunge__(2480, obj, newdata);
    end
    function r = summary (obj)
      % summary  C++ datamunge::KMeans::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::KMeans::summary() const
      r = __datamunge__(2481, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::KMeans::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::KMeans::print_summary() const
      __datamunge__(2482, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::KMeans. Do not edit.
