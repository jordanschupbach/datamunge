classdef PSOOptions < datamunge.NativeObject
  % PSOOptions  Configuration for PSO; fields retain their documented native defaults. Supply finite values in the supported ranges. Controls are copied at construction; validation, stopping and loss scaling belong to optimize().
  %
  %   obj = datamunge.PSOOptions ()
  %
  %   Example:
  %     dm::optim::PSOOptions o;
  %     std::print("{}\n",o.max_iterations); // 1000
  %
  %   Properties:
  %     population_size  integer: Population size; CMA-ES zero selects its dimension-based default. Default 40.
  %     max_iterations  integer: Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 1000.
  %     inertia_weight  double: Initial or constant particle velocity carryover. Default 0.7298.
  %     cognitive_coefficient  double: Particle attraction to its own best position. Default 1.49618.
  %     social_coefficient  double: Particle attraction to its neighbourhood best. Default 1.49618.
  %     topology  char row: global or ring particle neighbourhood. Default "global".
  %     ring_neighbors  integer: Neighbours on each ring side when ring topology is selected. Default 2.
  %     inertia_strategy  char row: constant or linear_decay. Default "constant".
  %     final_inertia_weight  double: Final carryover for linear_decay inertia. Default 0.4.
  %     tolerance  double: Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-10.
  %     seed  integer: Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
  %
  %   C++: class datamunge::optim::PSOOptions
  %
  %   See also: datamunge.PSO
  properties (Dependent)
    population_size  % Population size; CMA-ES zero selects its dimension-based default. Default 40.
    max_iterations  % Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 1000.
    inertia_weight  % Initial or constant particle velocity carryover. Default 0.7298.
    cognitive_coefficient  % Particle attraction to its own best position. Default 1.49618.
    social_coefficient  % Particle attraction to its neighbourhood best. Default 1.49618.
    topology  % global or ring particle neighbourhood. Default "global".
    ring_neighbors  % Neighbours on each ring side when ring topology is selected. Default 2.
    inertia_strategy  % constant or linear_decay. Default "constant".
    final_inertia_weight  % Final carryover for linear_decay inertia. Default 0.4.
    tolerance  % Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-10.
    seed  % Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
  end
  methods
    function obj = PSOOptions (varargin)
      % PSOOptions  Construct a C++ datamunge::optim::PSOOptions.
      %
      %   obj = datamunge.PSOOptions ()
      %
      %   C++: datamunge::optim::PSOOptions::PSOOptions()
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1706, false, varargin{:});
    end
    function v = get.population_size (obj)
      v = __datamunge__(1684, obj);
    end
    function set.population_size (obj, v)
      __datamunge__(1685, obj, v);
    end
    function v = get.max_iterations (obj)
      v = __datamunge__(1686, obj);
    end
    function set.max_iterations (obj, v)
      __datamunge__(1687, obj, v);
    end
    function v = get.inertia_weight (obj)
      v = __datamunge__(1688, obj);
    end
    function set.inertia_weight (obj, v)
      __datamunge__(1689, obj, v);
    end
    function v = get.cognitive_coefficient (obj)
      v = __datamunge__(1690, obj);
    end
    function set.cognitive_coefficient (obj, v)
      __datamunge__(1691, obj, v);
    end
    function v = get.social_coefficient (obj)
      v = __datamunge__(1692, obj);
    end
    function set.social_coefficient (obj, v)
      __datamunge__(1693, obj, v);
    end
    function v = get.topology (obj)
      v = __datamunge__(1694, obj);
    end
    function set.topology (obj, v)
      __datamunge__(1695, obj, v);
    end
    function v = get.ring_neighbors (obj)
      v = __datamunge__(1696, obj);
    end
    function set.ring_neighbors (obj, v)
      __datamunge__(1697, obj, v);
    end
    function v = get.inertia_strategy (obj)
      v = __datamunge__(1698, obj);
    end
    function set.inertia_strategy (obj, v)
      __datamunge__(1699, obj, v);
    end
    function v = get.final_inertia_weight (obj)
      v = __datamunge__(1700, obj);
    end
    function set.final_inertia_weight (obj, v)
      __datamunge__(1701, obj, v);
    end
    function v = get.tolerance (obj)
      v = __datamunge__(1702, obj);
    end
    function set.tolerance (obj, v)
      __datamunge__(1703, obj, v);
    end
    function v = get.seed (obj)
      v = __datamunge__(1704, obj);
    end
    function set.seed (obj, v)
      __datamunge__(1705, obj, v);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::PSOOptions. Do not edit.
