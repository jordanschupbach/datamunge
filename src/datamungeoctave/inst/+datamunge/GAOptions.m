classdef GAOptions < datamunge.NativeObject
  % GAOptions  Configuration for GeneticAlgorithm; fields retain their documented native defaults. Supply finite values in the supported ranges. Controls are copied at construction; validation, stopping and loss scaling belong to optimize().
  %
  %   obj = datamunge.GAOptions ()
  %
  %   Example:
  %     dm::optim::GAOptions o;
  %     std::print("{}\n",o.max_generations); // 500
  %
  %   Properties:
  %     population_size  integer: Population size; CMA-ES zero selects its dimension-based default. Default 100.
  %     max_generations  integer: Maximum population generations. Default 500.
  %     crossover_rate  double: Crossover probability CR/per-gene inclusion as defined by the method. Default 0.8.
  %     mutation_rate  double: Per-gene Gaussian mutation probability. Default 0.1.
  %     mutation_std_dev  double: Mutation SD as a fraction of the gene bounds width. Default 0.1.
  %     selection_strategy  char row: tournament, roulette or rank. Default "tournament".
  %     tournament_size  integer: Tournament candidate count. Default 3.
  %     crossover_strategy  char row: DE binomial/exponential or GA single_point/uniform/blend. Default "blend".
  %     blend_alpha  double: BLX-alpha expansion for blend crossover. Default 0.5.
  %     elitism  logical: Preserve elite population members. Default true.
  %     elite_count  integer: Number of members retained under elitism. Default 2.
  %     tolerance  double: Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-10.
  %     seed  integer: Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
  %
  %   C++: class datamunge::optim::GAOptions
  %
  %   See also: datamunge.GeneticAlgorithm
  properties (Dependent)
    population_size  % Population size; CMA-ES zero selects its dimension-based default. Default 100.
    max_generations  % Maximum population generations. Default 500.
    crossover_rate  % Crossover probability CR/per-gene inclusion as defined by the method. Default 0.8.
    mutation_rate  % Per-gene Gaussian mutation probability. Default 0.1.
    mutation_std_dev  % Mutation SD as a fraction of the gene bounds width. Default 0.1.
    selection_strategy  % tournament, roulette or rank. Default "tournament".
    tournament_size  % Tournament candidate count. Default 3.
    crossover_strategy  % DE binomial/exponential or GA single_point/uniform/blend. Default "blend".
    blend_alpha  % BLX-alpha expansion for blend crossover. Default 0.5.
    elitism  % Preserve elite population members. Default true.
    elite_count  % Number of members retained under elitism. Default 2.
    tolerance  % Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-10.
    seed  % Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
  end
  methods
    function obj = GAOptions (varargin)
      % GAOptions  Construct a C++ datamunge::optim::GAOptions.
      %
      %   obj = datamunge.GAOptions ()
      %
      %   C++: datamunge::optim::GAOptions::GAOptions()
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1524, false, varargin{:});
    end
    function v = get.population_size (obj)
      v = __datamunge__(1498, obj);
    end
    function set.population_size (obj, v)
      __datamunge__(1499, obj, v);
    end
    function v = get.max_generations (obj)
      v = __datamunge__(1500, obj);
    end
    function set.max_generations (obj, v)
      __datamunge__(1501, obj, v);
    end
    function v = get.crossover_rate (obj)
      v = __datamunge__(1502, obj);
    end
    function set.crossover_rate (obj, v)
      __datamunge__(1503, obj, v);
    end
    function v = get.mutation_rate (obj)
      v = __datamunge__(1504, obj);
    end
    function set.mutation_rate (obj, v)
      __datamunge__(1505, obj, v);
    end
    function v = get.mutation_std_dev (obj)
      v = __datamunge__(1506, obj);
    end
    function set.mutation_std_dev (obj, v)
      __datamunge__(1507, obj, v);
    end
    function v = get.selection_strategy (obj)
      v = __datamunge__(1508, obj);
    end
    function set.selection_strategy (obj, v)
      __datamunge__(1509, obj, v);
    end
    function v = get.tournament_size (obj)
      v = __datamunge__(1510, obj);
    end
    function set.tournament_size (obj, v)
      __datamunge__(1511, obj, v);
    end
    function v = get.crossover_strategy (obj)
      v = __datamunge__(1512, obj);
    end
    function set.crossover_strategy (obj, v)
      __datamunge__(1513, obj, v);
    end
    function v = get.blend_alpha (obj)
      v = __datamunge__(1514, obj);
    end
    function set.blend_alpha (obj, v)
      __datamunge__(1515, obj, v);
    end
    function v = get.elitism (obj)
      v = __datamunge__(1516, obj);
    end
    function set.elitism (obj, v)
      __datamunge__(1517, obj, v);
    end
    function v = get.elite_count (obj)
      v = __datamunge__(1518, obj);
    end
    function set.elite_count (obj, v)
      __datamunge__(1519, obj, v);
    end
    function v = get.tolerance (obj)
      v = __datamunge__(1520, obj);
    end
    function set.tolerance (obj, v)
      __datamunge__(1521, obj, v);
    end
    function v = get.seed (obj)
      v = __datamunge__(1522, obj);
    end
    function set.seed (obj, v)
      __datamunge__(1523, obj, v);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::GAOptions. Do not edit.
