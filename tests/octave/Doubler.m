% An Octave subclass overriding the C++ Callback's virtual call() (a director); used by
% tests/octave/test_datamunge.m.
classdef Doubler < datamunge.Callback
  methods
    function r = call (obj, x)
      r = 2 * x;
    end
  end
end
