classdef UndirectedGraph < datamunge.NativeObject
  % UndirectedGraph  C++ class datamunge::dstruct::UndirectedGraph<std::string>.
  %
  %   obj = datamunge.UndirectedGraph ()
  %   obj = datamunge.UndirectedGraph (vertices)
  %
  %   Methods:
  %     empty  C++ datamunge::dstruct::UndirectedGraph<std::string>::empty.
  %     vertex_count  C++ datamunge::dstruct::UndirectedGraph<std::string>::vertex_count.
  %     edge_count  C++ datamunge::dstruct::UndirectedGraph<std::string>::edge_count.
  %     clear  C++ datamunge::dstruct::UndirectedGraph<std::string>::clear.
  %     add_vertex  C++ datamunge::dstruct::UndirectedGraph<std::string>::add_vertex.
  %     remove_vertex  C++ datamunge::dstruct::UndirectedGraph<std::string>::remove_vertex.
  %     has_vertex  C++ datamunge::dstruct::UndirectedGraph<std::string>::has_vertex.
  %     add_edge  C++ datamunge::dstruct::UndirectedGraph<std::string>::add_edge.
  %     remove_edge  C++ datamunge::dstruct::UndirectedGraph<std::string>::remove_edge.
  %     has_edge  C++ datamunge::dstruct::UndirectedGraph<std::string>::has_edge.
  %     vertices  C++ datamunge::dstruct::UndirectedGraph<std::string>::vertices.
  %     neighbors  C++ datamunge::dstruct::UndirectedGraph<std::string>::neighbors.
  %     degree  C++ datamunge::dstruct::UndirectedGraph<std::string>::degree.
  %     bfs  C++ datamunge::dstruct::UndirectedGraph<std::string>::bfs.
  %     dfs  C++ datamunge::dstruct::UndirectedGraph<std::string>::dfs.
  %     contains_path  C++ datamunge::dstruct::UndirectedGraph<std::string>::contains_path.
  %     connected_components  The graph's connected components, each as a vector of vertices in BFS-visit order; components themselves appear in vertices() order.
  %     has_cycle  True if the graph contains a cycle (a self-loop, or two distinct paths between some pair of vertices) in any component.
  %
  %   C++: class datamunge::dstruct::UndirectedGraph<std::string>
  methods
    function obj = UndirectedGraph (varargin)
      % UndirectedGraph  Construct a C++ datamunge::dstruct::UndirectedGraph<std::string>.
      %
      %   obj = datamunge.UndirectedGraph ()
      %   obj = datamunge.UndirectedGraph (vertices)
      %
      %   C++: datamunge::dstruct::UndirectedGraph<std::string>::UndirectedGraph()
      %   C++: datamunge::dstruct::UndirectedGraph<std::string>::UndirectedGraph(const std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> & vertices)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3222, false, varargin{:});
    end
    function r = empty (obj)
      % empty  C++ datamunge::dstruct::UndirectedGraph<std::string>::empty.
      %
      %   r = obj.empty ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::empty() const
      r = __datamunge__(3204, obj);
    end
    function r = vertex_count (obj)
      % vertex_count  C++ datamunge::dstruct::UndirectedGraph<std::string>::vertex_count.
      %
      %   r = obj.vertex_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::UndirectedGraph<std::string>::vertex_count() const
      r = __datamunge__(3205, obj);
    end
    function r = edge_count (obj)
      % edge_count  C++ datamunge::dstruct::UndirectedGraph<std::string>::edge_count.
      %
      %   r = obj.edge_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::UndirectedGraph<std::string>::edge_count() const
      r = __datamunge__(3206, obj);
    end
    function clear (obj)
      % clear  C++ datamunge::dstruct::UndirectedGraph<std::string>::clear.
      %
      %   obj.clear ()
      %
      %   C++: void datamunge::dstruct::UndirectedGraph<std::string>::clear()
      __datamunge__(3207, obj);
    end
    function r = add_vertex (obj, varargin)
      % add_vertex  C++ datamunge::dstruct::UndirectedGraph<std::string>::add_vertex.
      %
      %   r = obj.add_vertex (vertex)
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::add_vertex(const std::basic_string<char> & vertex)
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::add_vertex(std::basic_string<char> && vertex)
      r = __datamunge__(3208, obj, varargin{:});
    end
    function r = remove_vertex (obj, vertex)
      % remove_vertex  C++ datamunge::dstruct::UndirectedGraph<std::string>::remove_vertex.
      %
      %   r = obj.remove_vertex (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::remove_vertex(const std::basic_string<char> & vertex)
      r = __datamunge__(3209, obj, vertex);
    end
    function r = has_vertex (obj, vertex)
      % has_vertex  C++ datamunge::dstruct::UndirectedGraph<std::string>::has_vertex.
      %
      %   r = obj.has_vertex (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::has_vertex(const std::basic_string<char> & vertex) const
      r = __datamunge__(3210, obj, vertex);
    end
    function r = add_edge (obj, a, b)
      % add_edge  C++ datamunge::dstruct::UndirectedGraph<std::string>::add_edge.
      %
      %   r = obj.add_edge (a, b)
      %
      %   Inputs:
      %     a  char row
      %     b  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::add_edge(const std::basic_string<char> & a, const std::basic_string<char> & b)
      r = __datamunge__(3211, obj, a, b);
    end
    function r = remove_edge (obj, a, b)
      % remove_edge  C++ datamunge::dstruct::UndirectedGraph<std::string>::remove_edge.
      %
      %   r = obj.remove_edge (a, b)
      %
      %   Inputs:
      %     a  char row
      %     b  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::remove_edge(const std::basic_string<char> & a, const std::basic_string<char> & b)
      r = __datamunge__(3212, obj, a, b);
    end
    function r = has_edge (obj, a, b)
      % has_edge  C++ datamunge::dstruct::UndirectedGraph<std::string>::has_edge.
      %
      %   r = obj.has_edge (a, b)
      %
      %   Inputs:
      %     a  char row
      %     b  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::has_edge(const std::basic_string<char> & a, const std::basic_string<char> & b) const
      r = __datamunge__(3213, obj, a, b);
    end
    function r = vertices (obj)
      % vertices  C++ datamunge::dstruct::UndirectedGraph<std::string>::vertices.
      %
      %   r = obj.vertices ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::UndirectedGraph<std::string>::vertices() const
      r = __datamunge__(3214, obj);
    end
    function r = neighbors (obj, vertex)
      % neighbors  C++ datamunge::dstruct::UndirectedGraph<std::string>::neighbors.
      %
      %   r = obj.neighbors (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::UndirectedGraph<std::string>::neighbors(const std::basic_string<char> & vertex) const
      r = __datamunge__(3215, obj, vertex);
    end
    function r = degree (obj, vertex)
      % degree  C++ datamunge::dstruct::UndirectedGraph<std::string>::degree.
      %
      %   r = obj.degree (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::UndirectedGraph<std::string>::degree(const std::basic_string<char> & vertex) const
      r = __datamunge__(3216, obj, vertex);
    end
    function r = bfs (obj, start)
      % bfs  C++ datamunge::dstruct::UndirectedGraph<std::string>::bfs.
      %
      %   r = obj.bfs (start)
      %
      %   Inputs:
      %     start  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::UndirectedGraph<std::string>::bfs(const std::basic_string<char> & start) const
      r = __datamunge__(3217, obj, start);
    end
    function r = dfs (obj, start)
      % dfs  C++ datamunge::dstruct::UndirectedGraph<std::string>::dfs.
      %
      %   r = obj.dfs (start)
      %
      %   Inputs:
      %     start  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::UndirectedGraph<std::string>::dfs(const std::basic_string<char> & start) const
      r = __datamunge__(3218, obj, start);
    end
    function r = contains_path (obj, src, dst)
      % contains_path  C++ datamunge::dstruct::UndirectedGraph<std::string>::contains_path.
      %
      %   r = obj.contains_path (src, dst)
      %
      %   Inputs:
      %     src  char row
      %     dst  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::contains_path(const std::basic_string<char> & src, const std::basic_string<char> & dst) const
      r = __datamunge__(3219, obj, src, dst);
    end
    function r = connected_components (obj)
      % connected_components  The graph's connected components, each as a vector of vertices in BFS-visit order; components themselves appear in vertices() order.
      %
      %   r = obj.connected_components ()
      %
      %   Output:
      %     r  cell array of cell array of char
      %
      %   C++: std::vector<std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>> datamunge::dstruct::UndirectedGraph<std::string>::connected_components() const
      r = __datamunge__(3220, obj);
    end
    function r = has_cycle (obj)
      % has_cycle  True if the graph contains a cycle (a self-loop, or two distinct paths between some pair of vertices) in any component.
      %
      %   r = obj.has_cycle ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::UndirectedGraph<std::string>::has_cycle() const
      r = __datamunge__(3221, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::dstruct::UndirectedGraph<std::string>. Do not edit.
