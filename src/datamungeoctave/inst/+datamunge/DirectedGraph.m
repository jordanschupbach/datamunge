classdef DirectedGraph < datamunge.NativeObject
  % DirectedGraph  C++ class datamunge::dstruct::DirectedGraph<std::string>.
  %
  %   obj = datamunge.DirectedGraph ()
  %   obj = datamunge.DirectedGraph (vertices)
  %
  %   Methods:
  %     empty  C++ datamunge::dstruct::DirectedGraph<std::string>::empty.
  %     vertex_count  C++ datamunge::dstruct::DirectedGraph<std::string>::vertex_count.
  %     edge_count  C++ datamunge::dstruct::DirectedGraph<std::string>::edge_count.
  %     clear  C++ datamunge::dstruct::DirectedGraph<std::string>::clear.
  %     add_vertex  C++ datamunge::dstruct::DirectedGraph<std::string>::add_vertex.
  %     remove_vertex  C++ datamunge::dstruct::DirectedGraph<std::string>::remove_vertex.
  %     has_vertex  C++ datamunge::dstruct::DirectedGraph<std::string>::has_vertex.
  %     add_edge  C++ datamunge::dstruct::DirectedGraph<std::string>::add_edge.
  %     remove_edge  C++ datamunge::dstruct::DirectedGraph<std::string>::remove_edge.
  %     has_edge  C++ datamunge::dstruct::DirectedGraph<std::string>::has_edge.
  %     vertices  C++ datamunge::dstruct::DirectedGraph<std::string>::vertices.
  %     outgoing_neighbors  C++ datamunge::dstruct::DirectedGraph<std::string>::outgoing_neighbors.
  %     incoming_neighbors  C++ datamunge::dstruct::DirectedGraph<std::string>::incoming_neighbors.
  %     out_degree  C++ datamunge::dstruct::DirectedGraph<std::string>::out_degree.
  %     in_degree  C++ datamunge::dstruct::DirectedGraph<std::string>::in_degree.
  %     bfs  C++ datamunge::dstruct::DirectedGraph<std::string>::bfs.
  %     dfs  C++ datamunge::dstruct::DirectedGraph<std::string>::dfs.
  %     contains_path  C++ datamunge::dstruct::DirectedGraph<std::string>::contains_path.
  %     transpose_  C++ datamunge::dstruct::DirectedGraph<std::string>::transpose.
  %     topological_sort  C++ datamunge::dstruct::DirectedGraph<std::string>::topological_sort.
  %
  %   C++: class datamunge::dstruct::DirectedGraph<std::string>
  methods
    function obj = DirectedGraph (varargin)
      % DirectedGraph  Construct a C++ datamunge::dstruct::DirectedGraph<std::string>.
      %
      %   obj = datamunge.DirectedGraph ()
      %   obj = datamunge.DirectedGraph (vertices)
      %
      %   C++: datamunge::dstruct::DirectedGraph<std::string>::DirectedGraph()
      %   C++: datamunge::dstruct::DirectedGraph<std::string>::DirectedGraph(const std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> & vertices)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3203, false, varargin{:});
    end
    function r = empty (obj)
      % empty  C++ datamunge::dstruct::DirectedGraph<std::string>::empty.
      %
      %   r = obj.empty ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::empty() const
      r = __datamunge__(3183, obj);
    end
    function r = vertex_count (obj)
      % vertex_count  C++ datamunge::dstruct::DirectedGraph<std::string>::vertex_count.
      %
      %   r = obj.vertex_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::DirectedGraph<std::string>::vertex_count() const
      r = __datamunge__(3184, obj);
    end
    function r = edge_count (obj)
      % edge_count  C++ datamunge::dstruct::DirectedGraph<std::string>::edge_count.
      %
      %   r = obj.edge_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::DirectedGraph<std::string>::edge_count() const
      r = __datamunge__(3185, obj);
    end
    function clear (obj)
      % clear  C++ datamunge::dstruct::DirectedGraph<std::string>::clear.
      %
      %   obj.clear ()
      %
      %   C++: void datamunge::dstruct::DirectedGraph<std::string>::clear()
      __datamunge__(3186, obj);
    end
    function r = add_vertex (obj, varargin)
      % add_vertex  C++ datamunge::dstruct::DirectedGraph<std::string>::add_vertex.
      %
      %   r = obj.add_vertex (vertex)
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::add_vertex(const std::basic_string<char> & vertex)
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::add_vertex(std::basic_string<char> && vertex)
      r = __datamunge__(3187, obj, varargin{:});
    end
    function r = remove_vertex (obj, vertex)
      % remove_vertex  C++ datamunge::dstruct::DirectedGraph<std::string>::remove_vertex.
      %
      %   r = obj.remove_vertex (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::remove_vertex(const std::basic_string<char> & vertex)
      r = __datamunge__(3188, obj, vertex);
    end
    function r = has_vertex (obj, vertex)
      % has_vertex  C++ datamunge::dstruct::DirectedGraph<std::string>::has_vertex.
      %
      %   r = obj.has_vertex (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::has_vertex(const std::basic_string<char> & vertex) const
      r = __datamunge__(3189, obj, vertex);
    end
    function r = add_edge (obj, src, dst)
      % add_edge  C++ datamunge::dstruct::DirectedGraph<std::string>::add_edge.
      %
      %   r = obj.add_edge (src, dst)
      %
      %   Inputs:
      %     src  char row
      %     dst  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::add_edge(const std::basic_string<char> & src, const std::basic_string<char> & dst)
      r = __datamunge__(3190, obj, src, dst);
    end
    function r = remove_edge (obj, src, dst)
      % remove_edge  C++ datamunge::dstruct::DirectedGraph<std::string>::remove_edge.
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
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::remove_edge(const std::basic_string<char> & src, const std::basic_string<char> & dst)
      r = __datamunge__(3191, obj, src, dst);
    end
    function r = has_edge (obj, src, dst)
      % has_edge  C++ datamunge::dstruct::DirectedGraph<std::string>::has_edge.
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
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::has_edge(const std::basic_string<char> & src, const std::basic_string<char> & dst) const
      r = __datamunge__(3192, obj, src, dst);
    end
    function r = vertices (obj)
      % vertices  C++ datamunge::dstruct::DirectedGraph<std::string>::vertices.
      %
      %   r = obj.vertices ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::DirectedGraph<std::string>::vertices() const
      r = __datamunge__(3193, obj);
    end
    function r = outgoing_neighbors (obj, vertex)
      % outgoing_neighbors  C++ datamunge::dstruct::DirectedGraph<std::string>::outgoing_neighbors.
      %
      %   r = obj.outgoing_neighbors (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::DirectedGraph<std::string>::outgoing_neighbors(const std::basic_string<char> & vertex) const
      r = __datamunge__(3194, obj, vertex);
    end
    function r = incoming_neighbors (obj, vertex)
      % incoming_neighbors  C++ datamunge::dstruct::DirectedGraph<std::string>::incoming_neighbors.
      %
      %   r = obj.incoming_neighbors (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::DirectedGraph<std::string>::incoming_neighbors(const std::basic_string<char> & vertex) const
      r = __datamunge__(3195, obj, vertex);
    end
    function r = out_degree (obj, vertex)
      % out_degree  C++ datamunge::dstruct::DirectedGraph<std::string>::out_degree.
      %
      %   r = obj.out_degree (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::DirectedGraph<std::string>::out_degree(const std::basic_string<char> & vertex) const
      r = __datamunge__(3196, obj, vertex);
    end
    function r = in_degree (obj, vertex)
      % in_degree  C++ datamunge::dstruct::DirectedGraph<std::string>::in_degree.
      %
      %   r = obj.in_degree (vertex)
      %
      %   Inputs:
      %     vertex  char row
      %
      %   Output:
      %     r  integer
      %
      %   C++: size_type datamunge::dstruct::DirectedGraph<std::string>::in_degree(const std::basic_string<char> & vertex) const
      r = __datamunge__(3197, obj, vertex);
    end
    function r = bfs (obj, start)
      % bfs  C++ datamunge::dstruct::DirectedGraph<std::string>::bfs.
      %
      %   r = obj.bfs (start)
      %
      %   Inputs:
      %     start  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::DirectedGraph<std::string>::bfs(const std::basic_string<char> & start) const
      r = __datamunge__(3198, obj, start);
    end
    function r = dfs (obj, start)
      % dfs  C++ datamunge::dstruct::DirectedGraph<std::string>::dfs.
      %
      %   r = obj.dfs (start)
      %
      %   Inputs:
      %     start  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::DirectedGraph<std::string>::dfs(const std::basic_string<char> & start) const
      r = __datamunge__(3199, obj, start);
    end
    function r = contains_path (obj, src, dst)
      % contains_path  C++ datamunge::dstruct::DirectedGraph<std::string>::contains_path.
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
      %   C++: bool datamunge::dstruct::DirectedGraph<std::string>::contains_path(const std::basic_string<char> & src, const std::basic_string<char> & dst) const
      r = __datamunge__(3200, obj, src, dst);
    end
    function r = transpose_ (obj)
      % transpose_  C++ datamunge::dstruct::DirectedGraph<std::string>::transpose.
      %
      %   r = obj.transpose_ ()
      %
      %   Output:
      %     r  datamunge.DirectedGraph
      %
      %   C++: DirectedGraph<std::string> datamunge::dstruct::DirectedGraph<std::string>::transpose() const
      r = __datamunge__(3201, obj);
    end
    function r = topological_sort (obj)
      % topological_sort  C++ datamunge::dstruct::DirectedGraph<std::string>::topological_sort.
      %
      %   r = obj.topological_sort ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char>>> datamunge::dstruct::DirectedGraph<std::string>::topological_sort() const
      r = __datamunge__(3202, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::dstruct::DirectedGraph<std::string>. Do not edit.
