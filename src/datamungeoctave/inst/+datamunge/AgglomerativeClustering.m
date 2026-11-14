classdef AgglomerativeClustering < datamunge.NativeObject
  % AgglomerativeClustering  SWIG-friendly facade for datamunge::stats::AgglomerativeClustering — bottom-up hierarchical clustering fit from a DataFrame and a list of numeric feature columns.
  %
  %   obj = datamunge.AgglomerativeClustering (data, feature_columns)
  %   obj = datamunge.AgglomerativeClustering (data, feature_columns, n_clusters)
  %   obj = datamunge.AgglomerativeClustering (data, feature_columns, n_clusters, linkage)
  %   obj = datamunge.AgglomerativeClustering (data, feature_columns, n_clusters, linkage, metric)
  %   obj = datamunge.AgglomerativeClustering (data, encoded_feature_columns)
  %   obj = datamunge.AgglomerativeClustering (data, encoded_feature_columns, n_clusters)
  %   obj = datamunge.AgglomerativeClustering (data, encoded_feature_columns, n_clusters, linkage)
  %   obj = datamunge.AgglomerativeClustering (data, encoded_feature_columns, n_clusters, linkage, metric)
  %
  %   Methods:
  %     feature_names  C++ datamunge::AgglomerativeClustering::feature_names.
  %     observations  C++ datamunge::AgglomerativeClustering::observations.
  %     labels  C++ datamunge::AgglomerativeClustering::labels.
  %     cut  Re-cuts the already-built dendrogram to produce `n_clusters` clusters, without refitting.
  %     num_merges  C++ datamunge::AgglomerativeClustering::num_merges.
  %     merge_cluster_a  C++ datamunge::AgglomerativeClustering::merge_cluster_a.
  %     merge_cluster_b  C++ datamunge::AgglomerativeClustering::merge_cluster_b.
  %     merge_distance  C++ datamunge::AgglomerativeClustering::merge_distance.
  %     merge_size  C++ datamunge::AgglomerativeClustering::merge_size.
  %     summary  C++ datamunge::AgglomerativeClustering::summary.
  %     print_summary  C++ datamunge::AgglomerativeClustering::print_summary.
  %
  %   C++: class datamunge::AgglomerativeClustering
  methods
    function obj = AgglomerativeClustering (varargin)
      % AgglomerativeClustering  Construct a C++ datamunge::AgglomerativeClustering.
      %
      %   obj = datamunge.AgglomerativeClustering (data, feature_columns)
      %   obj = datamunge.AgglomerativeClustering (data, feature_columns, n_clusters)
      %   obj = datamunge.AgglomerativeClustering (data, feature_columns, n_clusters, linkage)
      %   obj = datamunge.AgglomerativeClustering (data, feature_columns, n_clusters, linkage, metric)
      %   obj = datamunge.AgglomerativeClustering (data, encoded_feature_columns)
      %   obj = datamunge.AgglomerativeClustering (data, encoded_feature_columns, n_clusters)
      %   obj = datamunge.AgglomerativeClustering (data, encoded_feature_columns, n_clusters, linkage)
      %   obj = datamunge.AgglomerativeClustering (data, encoded_feature_columns, n_clusters, linkage, metric)
      %
      %   C++: datamunge::AgglomerativeClustering::AgglomerativeClustering(const DataFrame & data, const std::vector<std::string> & feature_columns, std::size_t n_clusters = 2, const std::string & linkage = "ward", const std::string & metric = "euclidean")
      %   C++: datamunge::AgglomerativeClustering::AgglomerativeClustering(const DataFrame & data, const std::string & encoded_feature_columns, std::size_t n_clusters = 2, const std::string & linkage = "ward", const std::string & metric = "euclidean")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2624, false, varargin{:});
    end
    function r = feature_names (obj)
      % feature_names  C++ datamunge::AgglomerativeClustering::feature_names.
      %
      %   r = obj.feature_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::AgglomerativeClustering::feature_names() const
      r = __datamunge__(2613, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::AgglomerativeClustering::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::AgglomerativeClustering::observations() const
      r = __datamunge__(2614, obj);
    end
    function r = labels (obj)
      % labels  C++ datamunge::AgglomerativeClustering::labels.
      %
      %   r = obj.labels ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::AgglomerativeClustering::labels() const
      r = __datamunge__(2615, obj);
    end
    function r = cut (obj, n_clusters)
      % cut  Re-cuts the already-built dendrogram to produce `n_clusters` clusters, without refitting.
      %
      %   r = obj.cut (n_clusters)
      %
      %   Inputs:
      %     n_clusters  integer
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::AgglomerativeClustering::cut(std::size_t n_clusters) const
      r = __datamunge__(2616, obj, n_clusters);
    end
    function r = num_merges (obj)
      % num_merges  C++ datamunge::AgglomerativeClustering::num_merges.
      %
      %   r = obj.num_merges ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::AgglomerativeClustering::num_merges() const
      r = __datamunge__(2617, obj);
    end
    function r = merge_cluster_a (obj, merge_index)
      % merge_cluster_a  C++ datamunge::AgglomerativeClustering::merge_cluster_a.
      %
      %   r = obj.merge_cluster_a (merge_index)
      %
      %   Inputs:
      %     merge_index  integer
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::AgglomerativeClustering::merge_cluster_a(std::size_t merge_index) const
      r = __datamunge__(2618, obj, merge_index);
    end
    function r = merge_cluster_b (obj, merge_index)
      % merge_cluster_b  C++ datamunge::AgglomerativeClustering::merge_cluster_b.
      %
      %   r = obj.merge_cluster_b (merge_index)
      %
      %   Inputs:
      %     merge_index  integer
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::AgglomerativeClustering::merge_cluster_b(std::size_t merge_index) const
      r = __datamunge__(2619, obj, merge_index);
    end
    function r = merge_distance (obj, merge_index)
      % merge_distance  C++ datamunge::AgglomerativeClustering::merge_distance.
      %
      %   r = obj.merge_distance (merge_index)
      %
      %   Inputs:
      %     merge_index  integer
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::AgglomerativeClustering::merge_distance(std::size_t merge_index) const
      r = __datamunge__(2620, obj, merge_index);
    end
    function r = merge_size (obj, merge_index)
      % merge_size  C++ datamunge::AgglomerativeClustering::merge_size.
      %
      %   r = obj.merge_size (merge_index)
      %
      %   Inputs:
      %     merge_index  integer
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::AgglomerativeClustering::merge_size(std::size_t merge_index) const
      r = __datamunge__(2621, obj, merge_index);
    end
    function r = summary (obj)
      % summary  C++ datamunge::AgglomerativeClustering::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::AgglomerativeClustering::summary() const
      r = __datamunge__(2622, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::AgglomerativeClustering::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::AgglomerativeClustering::print_summary() const
      __datamunge__(2623, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::AgglomerativeClustering. Do not edit.
