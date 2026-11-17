classdef WeightedGraph < datamunge.NativeObject
  % WeightedGraph  C++ class datamunge::dstruct::WeightedGraph<std::string, double>.
  %
  %   obj = datamunge.WeightedGraph (directed)
  %   obj = datamunge.WeightedGraph (vertices, directed)
  %
  %   Methods:
  %     directed  C++ datamunge::dstruct::WeightedGraph<std::string, double>::directed.
  %     empty  C++ datamunge::dstruct::WeightedGraph<std::string, double>::empty.
  %     vertex_count  C++ datamunge::dstruct::WeightedGraph<std::string, double>::vertex_count.
  %     edge_count  C++ datamunge::dstruct::WeightedGraph<std::string, double>::edge_count.
  %     clear  C++ datamunge::dstruct::WeightedGraph<std::string, double>::clear.
  %     add_vertex  C++ datamunge::dstruct::WeightedGraph<std::string, double>::add_vertex.
  %     remove_vertex  C++ datamunge::dstruct::WeightedGraph<std::string, double>::remove_vertex.
  %     has_vertex  C++ datamunge::dstruct::WeightedGraph<std::string, double>::has_vertex.
  %     add_edge  Replaces the weight if the edge already exists (rather than rejecting the call, unlike DirectedGraph/UndirectedGraph's add_edge -- weights are the whole point of this class, so re-adding with a new weight is the natural way to update one).
  %     remove_edge  C++ datamunge::dstruct::WeightedGraph<std::string, double>::remove_edge.
  %     has_edge  C++ datamunge::dstruct::WeightedGraph<std::string, double>::has_edge.
  %     edge_weight  C++ datamunge::dstruct::WeightedGraph<std::string, double>::edge_weight.
  %     vertices  C++ datamunge::dstruct::WeightedGraph<std::string, double>::vertices.
  %     neighbors  Outgoing neighbors if directed(); all incident neighbors if !directed().
  %     dijkstra_distances  Same computation as dijkstra(), but returned as (vertex, distance) pairs in vertices() order rather than an unordered_map -- a binding-friendly shape (plain vector<pair<>>, using the same std_pair/std_vector machinery already relied on elsewhere) for language bindings where std::unordered_map isn't cleanly exposed.
  %     bellman_ford_distances  Same computation as bellman_ford(), but returned as (vertex, distance) pairs in vertices() order rather than an unordered_map -- see dijkstra_distances() for why. Empty (with has_negative_cycle() separately reporting the reason) when a negative cycle makes the distances themselves meaningless.
  %     bellman_ford_has_negative_cycle  C++ datamunge::dstruct::WeightedGraph<std::string, double>::bellman_ford_has_negative_cycle.
  %     minimum_spanning_tree  Kruskal's minimum spanning tree/forest, as a new undirected WeightedGraph containing every vertex and the selected edges. Throws std::logic_error if directed() (MST is only defined for undirected graphs).
  %     connected_components  Connected components (ignoring weights). Throws std::logic_error if directed() (use strongly_connected_components() instead).
  %     strongly_connected_components  Strongly connected components (Tarjan's algorithm), each as a vector of vertices; components are returned in reverse-topological order (a component with no edges leaving it to another component comes first). Throws std::logic_error if !directed() (use connected_components() instead).
  %     has_cycle  True if the graph contains a cycle: a self-loop (either kind), or -- for a directed graph, a directed cycle (found via 3-color DFS); for an undirected graph, two distinct paths between some pair of vertices.
  %
  %   C++: class datamunge::dstruct::WeightedGraph<std::string, double>
  methods
    function obj = WeightedGraph (varargin)
      % WeightedGraph  Construct a C++ datamunge::dstruct::WeightedGraph<std::string, double>.
      %
      %   obj = datamunge.WeightedGraph (directed)
      %   obj = datamunge.WeightedGraph (vertices, directed)
      %
      %   C++: datamunge::dstruct::WeightedGraph<std::string, double>::WeightedGraph(bool directed)
      %   C++: datamunge::dstruct::WeightedGraph<std::string, double>::WeightedGraph(const std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> & vertices, bool directed)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3244, false, varargin{:});
    end
    function r = directed (obj)
      % directed  C++ datamunge::dstruct::WeightedGraph<std::string, double>::directed.
      %
      %   r = obj.directed ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::directed() const
      r = __datamunge__(3223, obj);
    end
    function r = empty (obj)
      % empty  C++ datamunge::dstruct::WeightedGraph<std::string, double>::empty.
      %
      %   r = obj.empty ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::empty() const
      r = __datamunge__(3224, obj);
    end
    function r = vertex_count (obj)
      % vertex_count  C++ datamunge::dstruct::WeightedGraph<std::string, double>::vertex_count.
      %
      %   r = obj.vertex_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::WeightedGraph<std::string, double>::vertex_count() const
      r = __datamunge__(3225, obj);
    end
    function r = edge_count (obj)
      % edge_count  C++ datamunge::dstruct::WeightedGraph<std::string, double>::edge_count.
      %
      %   r = obj.edge_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::WeightedGraph<std::string, double>::edge_count() const
      r = __datamunge__(3226, obj);
    end
    function clear (obj)
      % clear  C++ datamunge::dstruct::WeightedGraph<std::string, double>::clear.
      %
      %   obj.clear ()
      %
      %   C++: void datamunge::dstruct::WeightedGraph<std::string, double>::clear()
      __datamunge__(3227, obj);
    end
    function r = add_vertex (obj, vertex)
      % add_vertex  C++ datamunge::dstruct::WeightedGraph<std::string, double>::add_vertex.
      %
      %   r = obj.add_vertex (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::add_vertex(const std::basic_string<char> & vertex)
      r = __datamunge__(3228, obj, vertex);
    end
    function r = remove_vertex (obj, vertex)
      % remove_vertex  C++ datamunge::dstruct::WeightedGraph<std::string, double>::remove_vertex.
      %
      %   r = obj.remove_vertex (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::remove_vertex(const std::basic_string<char> & vertex)
      r = __datamunge__(3229, obj, vertex);
    end
    function r = has_vertex (obj, vertex)
      % has_vertex  C++ datamunge::dstruct::WeightedGraph<std::string, double>::has_vertex.
      %
      %   r = obj.has_vertex (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::has_vertex(const std::basic_string<char> & vertex) const
      r = __datamunge__(3230, obj, vertex);
    end
    function r = add_edge (obj, src, dst, weight)
      % add_edge  Replaces the weight if the edge already exists (rather than rejecting the call, unlike DirectedGraph/UndirectedGraph's add_edge -- weights are the whole point of this class, so re-adding with a new weight is the natural way to update one).
      %
      %   r = obj.add_edge (src, dst, weight)
      %
      %   Inputs:
      %     src  char row
      %     dst  char row
      %     weight  double
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::add_edge(const std::basic_string<char> & src, const std::basic_string<char> & dst, const double & weight)
      r = __datamunge__(3231, obj, src, dst, weight);
    end
    function r = remove_edge (obj, src, dst)
      % remove_edge  C++ datamunge::dstruct::WeightedGraph<std::string, double>::remove_edge.
      %
      %   r = obj.remove_edge (src, dst)
      %
      %   Inputs:
      %     src  char row
      %     dst  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::remove_edge(const std::basic_string<char> & src, const std::basic_string<char> & dst)
      r = __datamunge__(3232, obj, src, dst);
    end
    function r = has_edge (obj, src, dst)
      % has_edge  C++ datamunge::dstruct::WeightedGraph<std::string, double>::has_edge.
      %
      %   r = obj.has_edge (src, dst)
      %
      %   Inputs:
      %     src  char row
      %     dst  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::has_edge(const std::basic_string<char> & src, const std::basic_string<char> & dst) const
      r = __datamunge__(3233, obj, src, dst);
    end
    function r = edge_weight (obj, src, dst)
      % edge_weight  C++ datamunge::dstruct::WeightedGraph<std::string, double>::edge_weight.
      %
      %   r = obj.edge_weight (src, dst)
      %
      %   Inputs:
      %     src  char row
      %     dst  char row
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::dstruct::WeightedGraph<std::string, double>::edge_weight(const std::basic_string<char> & src, const std::basic_string<char> & dst) const
      r = __datamunge__(3234, obj, src, dst);
    end
    function r = vertices (obj)
      % vertices  C++ datamunge::dstruct::WeightedGraph<std::string, double>::vertices.
      %
      %   r = obj.vertices ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::WeightedGraph<std::string, double>::vertices() const
      r = __datamunge__(3235, obj);
    end
    function r = neighbors (obj, vertex)
      % neighbors  Outgoing neighbors if directed(); all incident neighbors if !directed().
      %
      %   r = obj.neighbors (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::WeightedGraph<std::string, double>::neighbors(const std::basic_string<char> & vertex) const
      r = __datamunge__(3236, obj, vertex);
    end
    function r = dijkstra_distances (obj, source)
      % dijkstra_distances  Same computation as dijkstra(), but returned as (vertex, distance) pairs in vertices() order rather than an unordered_map -- a binding-friendly shape (plain vector<pair<>>, using the same std_pair/std_vector machinery already relied on elsewhere) for language bindings where std::unordered_map isn't cleanly exposed.
      %
      %   r = obj.dijkstra_distances (source)
      %
      %   Inputs:
      %     source  char row
      %
      %   Output:
      %     r  cell array of cell {char row, double}
      %
      %   C++: std::vector<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char>>, double>> datamunge::dstruct::WeightedGraph<std::string, double>::dijkstra_distances(const std::basic_string<char> & source) const
      r = __datamunge__(3237, obj, source);
    end
    function r = bellman_ford_distances (obj, source)
      % bellman_ford_distances  Same computation as bellman_ford(), but returned as (vertex, distance) pairs in vertices() order rather than an unordered_map -- see dijkstra_distances() for why. Empty (with has_negative_cycle() separately reporting the reason) when a negative cycle makes the distances themselves meaningless.
      %
      %   r = obj.bellman_ford_distances (source)
      %
      %   Inputs:
      %     source  char row
      %
      %   Output:
      %     r  cell array of cell {char row, double}
      %
      %   C++: std::vector<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char>>, double>> datamunge::dstruct::WeightedGraph<std::string, double>::bellman_ford_distances(const std::basic_string<char> & source) const
      r = __datamunge__(3238, obj, source);
    end
    function r = bellman_ford_has_negative_cycle (obj, source)
      % bellman_ford_has_negative_cycle  C++ datamunge::dstruct::WeightedGraph<std::string, double>::bellman_ford_has_negative_cycle.
      %
      %   r = obj.bellman_ford_has_negative_cycle (source)
      %
      %   Inputs:
      %     source  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::bellman_ford_has_negative_cycle(const std::basic_string<char> & source) const
      r = __datamunge__(3239, obj, source);
    end
    function r = minimum_spanning_tree (obj)
      % minimum_spanning_tree  Kruskal's minimum spanning tree/forest, as a new undirected WeightedGraph containing every vertex and the selected edges. Throws std::logic_error if directed() (MST is only defined for undirected graphs).
      %
      %   r = obj.minimum_spanning_tree ()
      %
      %   Output:
      %     r  datamunge.WeightedGraph
      %
      %   C++: WeightedGraph<std::string, double> datamunge::dstruct::WeightedGraph<std::string, double>::minimum_spanning_tree() const
      r = __datamunge__(3240, obj);
    end
    function r = connected_components (obj)
      % connected_components  Connected components (ignoring weights). Throws std::logic_error if directed() (use strongly_connected_components() instead).
      %
      %   r = obj.connected_components ()
      %
      %   Output:
      %     r  cell array of cell array of char
      %
      %   C++: std::vector<std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>> datamunge::dstruct::WeightedGraph<std::string, double>::connected_components() const
      r = __datamunge__(3241, obj);
    end
    function r = strongly_connected_components (obj)
      % strongly_connected_components  Strongly connected components (Tarjan's algorithm), each as a vector of vertices; components are returned in reverse-topological order (a component with no edges leaving it to another component comes first). Throws std::logic_error if !directed() (use connected_components() instead).
      %
      %   r = obj.strongly_connected_components ()
      %
      %   Output:
      %     r  cell array of cell array of char
      %
      %   C++: std::vector<std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>> datamunge::dstruct::WeightedGraph<std::string, double>::strongly_connected_components() const
      r = __datamunge__(3242, obj);
    end
    function r = has_cycle (obj)
      % has_cycle  True if the graph contains a cycle: a self-loop (either kind), or -- for a directed graph, a directed cycle (found via 3-color DFS); for an undirected graph, two distinct paths between some pair of vertices.
      %
      %   r = obj.has_cycle ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::WeightedGraph<std::string, double>::has_cycle() const
      r = __datamunge__(3243, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::dstruct::WeightedGraph<std::string, double>. Do not edit.
