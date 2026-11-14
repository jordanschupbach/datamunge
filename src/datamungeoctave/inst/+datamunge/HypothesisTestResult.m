classdef HypothesisTestResult < datamunge.NativeObject
  % HypothesisTestResult  Owned hypothesis-test result, analogous to R's htest list. Each test documents which fields it populates and what its interval estimates. Unused numeric fields retain zero; this default is storage, not a claim of zero effect. Read bounds only when has_conf_int is true; one-sided intervals may be unbounded. p_value is a null-model tail probability, not a posterior probability of the null. Objects own their method string and contain no borrowed references or shared state.
  %
  %   obj = datamunge.HypothesisTestResult ()
  %
  %   Example:
  %     const auto t=dm::stats::t_test_one_sample({1,2,3,4});
  %     std::print("{}\n",t.has_conf_int); // true
  %
  %   References:
  %   - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html
  %
  %   Properties:
  %     statistic  double: Test-specific statistic (t, U, D, F, H, count or chi-square).
  %     parameter1  double: Primary df or count parameter; consult the test contract.
  %     parameter2  double: Secondary df/count parameter where used; otherwise zero.
  %     p_value  double: Null-reference p-value, not adjusted for multiple testing.
  %     estimate1  double: First test-specific estimate, such as a mean, ratio or correlation.
  %     estimate2  double: Second estimate when applicable, such as a second group mean.
  %     conf_int_lower  double: Lower confidence bound, meaningful only when has_conf_int.
  %     conf_int_upper  double: Upper confidence bound, meaningful only when has_conf_int.
  %     has_conf_int  logical: Whether confidence bounds were computed for this result.
  %     alternative  datamunge.Alternative value: Tail requested by the caller.
  %     method  char row: Owned human-readable method, including relevant approximation labels.
  %
  %   C++: class datamunge::stats::HypothesisTestResult
  %
  %   See also: datamunge.Alternative, datamunge.hypothesis_test_table()
  properties (Dependent)
    statistic  % Test-specific statistic (t, U, D, F, H, count or chi-square).
    parameter1  % Primary df or count parameter; consult the test contract.
    parameter2  % Secondary df/count parameter where used; otherwise zero.
    p_value  % Null-reference p-value, not adjusted for multiple testing.
    estimate1  % First test-specific estimate, such as a mean, ratio or correlation.
    estimate2  % Second estimate when applicable, such as a second group mean.
    conf_int_lower  % Lower confidence bound, meaningful only when has_conf_int.
    conf_int_upper  % Upper confidence bound, meaningful only when has_conf_int.
    has_conf_int  % Whether confidence bounds were computed for this result.
    alternative  % Tail requested by the caller.
    method  % Owned human-readable method, including relevant approximation labels.
  end
  methods
    function obj = HypothesisTestResult (varargin)
      % HypothesisTestResult  Construct a C++ datamunge::stats::HypothesisTestResult.
      %
      %   obj = datamunge.HypothesisTestResult ()
      %
      %   C++: datamunge::stats::HypothesisTestResult::HypothesisTestResult()
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1917, false, varargin{:});
    end
    function v = get.statistic (obj)
      v = __datamunge__(1895, obj);
    end
    function set.statistic (obj, v)
      __datamunge__(1896, obj, v);
    end
    function v = get.parameter1 (obj)
      v = __datamunge__(1897, obj);
    end
    function set.parameter1 (obj, v)
      __datamunge__(1898, obj, v);
    end
    function v = get.parameter2 (obj)
      v = __datamunge__(1899, obj);
    end
    function set.parameter2 (obj, v)
      __datamunge__(1900, obj, v);
    end
    function v = get.p_value (obj)
      v = __datamunge__(1901, obj);
    end
    function set.p_value (obj, v)
      __datamunge__(1902, obj, v);
    end
    function v = get.estimate1 (obj)
      v = __datamunge__(1903, obj);
    end
    function set.estimate1 (obj, v)
      __datamunge__(1904, obj, v);
    end
    function v = get.estimate2 (obj)
      v = __datamunge__(1905, obj);
    end
    function set.estimate2 (obj, v)
      __datamunge__(1906, obj, v);
    end
    function v = get.conf_int_lower (obj)
      v = __datamunge__(1907, obj);
    end
    function set.conf_int_lower (obj, v)
      __datamunge__(1908, obj, v);
    end
    function v = get.conf_int_upper (obj)
      v = __datamunge__(1909, obj);
    end
    function set.conf_int_upper (obj, v)
      __datamunge__(1910, obj, v);
    end
    function v = get.has_conf_int (obj)
      v = __datamunge__(1911, obj);
    end
    function set.has_conf_int (obj, v)
      __datamunge__(1912, obj, v);
    end
    function v = get.alternative (obj)
      v = __datamunge__(1913, obj);
    end
    function set.alternative (obj, v)
      __datamunge__(1914, obj, v);
    end
    function v = get.method (obj)
      v = __datamunge__(1915, obj);
    end
    function set.method (obj, v)
      __datamunge__(1916, obj, v);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::stats::HypothesisTestResult. Do not edit.
