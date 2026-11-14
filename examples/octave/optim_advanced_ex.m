% CMD: just TARGET=optim_advanced_ex run-octave
% Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
% AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Octave.
%
% Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
% EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
% ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
% implement the objective. From Octave, derive a classdef from the interface class and override
% its methods; the optimizer then calls the Octave overrides. Coordinates updated in place (the
% C++ parameter is a non-const std::vector<double>&) come back as an extra output.
%
% Octave classdefs must live in their own files, so this script writes the three objective
% classes into a temporary directory and adds it to the path.

1;

function write_class (dir, name, text)
  fid = fopen (fullfile (dir, [name '.m']), 'w');
  fputs (fid, text);
  fclose (fid);
end

classdir = tempname ();
mkdir (classdir);
% f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
write_class (classdir, 'Quadratic', strjoin ({
  'classdef Quadratic < datamunge.HessianFunction'
  '  methods'
  '    function r = evaluate (obj, c)'
  '      r = (c(1) - 1)^2 + 10 * (c(2) + 2)^2;'
  '    end'
  '    function r = gradient (obj, c)'
  '      r = [2 * (c(1) - 1), 20 * (c(2) + 2)];'
  '    end'
  '    function r = hessian (obj, c)'
  '      r = [2 0; 0 20];'
  '    end'
  '  end'
  'end'}, "\n"));
% Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5.
write_class (classdir, 'Exponential', strjoin ({
  'classdef Exponential < datamunge.ResidualFunction'
  '  properties'
  '    ts'
  '  end'
  '  methods'
  '    function obj = Exponential (ts)'
  '      obj.ts = ts;'
  '    end'
  '    function r = residuals (obj, p)'
  '      r = p(1) * exp (p(2) * obj.ts) - 2 * exp (0.5 * obj.ts);'
  '    end'
  '    function r = jacobian (obj, p)'
  '      t = obj.ts(:);'
  '      r = [exp(p(2) * t), p(1) * t .* exp(p(2) * t)];'
  '    end'
  '  end'
  'end'}, "\n"));
% Leaves gradient() and hessian() out.
write_class (classdir, 'Incomplete', strjoin ({
  'classdef Incomplete < datamunge.HessianFunction'
  '  methods'
  '    function r = evaluate (obj, c)'
  '      r = 0;'
  '    end'
  '  end'
  'end'}, "\n"));
addpath (classdir);

printf ('=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================\n');
% Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
% then check the acquisition function favors an unexplored region.
xs = [0.0 0.5 1.0 1.5 2.0 3.0];
ys = (xs - 1) .^ 2;
surrogate = datamunge.RBFGaussianProcessSurrogate (1.0, 1e-6);
surrogate.fit (xs(:), ys);
incumbent = min (0.0, min (ys));  % best observed value so far
printf ('incumbent (best observed y): %g\n', incumbent);
for x = [0.25 1.0 2.5]
  printf ('acquisition(x=%.2f) = %.6f\n', x, surrogate.acquisition (x, incumbent));
end

printf ('\n=================== Options structs + optimizer construction (all 10 new classes) ===================\n');
% Every new Options struct and optimizer class constructs cleanly, proving
% the binding for the whole batch is wired correctly.
pg_options = datamunge.ProximalGradientOptions ();
pg_options.step_size = 0.05;
pg = datamunge.ProximalGradient (pg_options);
printf ('ProximalGradient: %s()\n', class (pg)(11:end));
fista = datamunge.FISTA (datamunge.FISTAOptions ());
printf ('FISTA: %s()\n', class (fista)(11:end));
lm_options = datamunge.LevenbergMarquardtOptions ();
lm_options.max_iterations = 200;
lm = datamunge.LevenbergMarquardt (lm_options);
printf ('LevenbergMarquardt: %s()\n', class (lm)(11:end));
newton = datamunge.Newton (datamunge.NewtonOptions ());
printf ('Newton: %s()\n', class (newton)(11:end));
trn = datamunge.TrustRegionNewton (datamunge.TrustRegionNewtonOptions ());
printf ('TrustRegionNewton: %s()\n', class (trn)(11:end));
al = datamunge.AugmentedLagrangian (datamunge.AugmentedLagrangianOptions ());
printf ('AugmentedLagrangian: %s()\n', class (al)(11:end));
sqp = datamunge.SQP (datamunge.SQPOptions ());
printf ('SQP: %s()\n', class (sqp)(11:end));
ip = datamunge.InteriorPoint (datamunge.InteriorPointOptions ());
printf ('InteriorPoint: %s()\n', class (ip)(11:end));
bo_options = datamunge.BayesianOptimizationOptions ();
bo_options.initial_samples = 8;
bo = datamunge.BayesianOptimization (bo_options);
printf ('BayesianOptimization: %s()\n', class (bo)(11:end));

printf ('\n=================== Octave objectives (subclassing the C++ interfaces) ===================\n');
% Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
[value, start] = newton.optimize (Quadratic (), [5.0 5.0]);
printf ('Newton: f = %.6g at (%.6f, %.6f) (exact minimum 0 at (1, -2))\n', value, start(1), start(2));

% Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
[sse, params] = lm.optimize (Exponential ([0.0 0.5 1.0 1.5 2.0]), [1.0 0.1]);
printf ('LevenbergMarquardt: residual = %.3g, a = %.6f, b = %.6f (exact: 2, 0.5)\n', sse, params(1), params(2));

% An interface method a subclass leaves out throws when the optimizer calls it.
err = [];
try
  newton.optimize (Incomplete (), [0.0 0.0]);
catch e
  err = e;
end
bools = {'false', 'true'};
printf ('missing gradient(): ok=%s, error mentions it: %s\n', bools{isempty (err) + 1}, ...
        bools{(~isempty (err) && ~isempty (strfind (err.message, 'gradient'))) + 1});

rmpath (classdir);
confirm_recursive_rmdir (false, 'local');
rmdir (classdir, 's');
