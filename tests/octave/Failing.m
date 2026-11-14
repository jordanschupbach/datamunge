% An override raising an Octave error, which C++ propagates back to the caller; used by
% tests/octave/test_datamunge.m.
classdef Failing < datamunge.Callback
  methods
    function r = call (obj, x)
      error ('Failing:call', 'from Octave');
    end
  end
end
