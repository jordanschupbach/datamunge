1;

datamunge;

% NOTE: this exercises the 10 new datamunge::optim classes (ProximalGradient, FISTA,
% LevenbergMarquardt, Newton, TrustRegionNewton, AugmentedLagrangian, SQP, InteriorPoint,
% BayesianOptimization, RBFGaussianProcessSurrogate) added alongside ProximalFunction,
% HessianFunction, EqualityConstrainedFunction, InequalityConstrainedFunction, and the
% standalone ResidualFunction / BayesianSurrogate interfaces.
%
% As documented in datamunge_octave_bindings.md, Octave's SWIG director glue crashes the
% whole interpreter (uncaught Swig::DirectorPureVirtualException / SIGABRT) on director
% methods that RETURN std::vector<double>, at minimum DifferentiableFunction::gradient().
% This was re-tested empirically for this task against ResidualFunction::residuals() (also a
% vector<double>-returning director method, but on a class that does NOT derive from
% DifferentiableFunction) -- see the report for the exact outcome. Every one of the 9 new
% optimizers besides BayesianOptimization requires the user to override at least one
% vector-returning method on ProximalFunction / HessianFunction /
% EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction
% (gradient/proximal/hessian/constraints/constraint_jacobian/inequalities/
% inequality_jacobian/residuals/jacobian). BayesianOptimization is portable regardless,
% because its objective only needs ArbitraryFunction::evaluate() (returns a plain double,
% known-good since optim_ex.m) and the ready-made RBFGaussianProcessSurrogate (a concrete
% C++ class -- no BayesianSurrogate subclassing required, so its void fit()/double
% acquisition() methods are never exercised as directors at all).

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function y = shifted_bowl_evaluate(self, x)
  y = (x{1} - 1.7)^2 + 0.1 * sin(10.0 * x{1});
endfunction

printf("=================== BayesianOptimization with RBFGaussianProcessSurrogate ===================\n");
f = ArbitraryFunction();
f.evaluate = @shifted_bowl_evaluate;
x = dv([0.0]);
lower = dv([-3.0]);
upper = dv([5.0]);
surrogate = RBFGaussianProcessSurrogate(1.0, 1e-6);
bo_options = BayesianOptimizationOptions();
BayesianOptimizationOptions_initial_samples_set(bo_options, 10);
BayesianOptimizationOptions_max_iterations_set(bo_options, 60);
BayesianOptimizationOptions_seed_set(bo_options, 42);
value = BayesianOptimization_optimize(BayesianOptimization(bo_options), f, x, lower, upper, surrogate);
printf("BayesianOptimization: f=%g x=[%g] (target minimum near x=1.7)\n", value, DVector___paren__(x, 0));

printf("\n=================== ResidualFunction/LevenbergMarquardt: confirming the director crash ===================\n");
printf("(NOT run automatically -- see comment block at top of this file and\n");
printf(" datamunge_octave_bindings.md for the empirical isolated-call confirmation that any\n");
printf(" director method returning std::vector<double> -- including ResidualFunction::residuals,\n");
printf(" which ProximalGradient/FISTA/LevenbergMarquardt/Newton/TrustRegionNewton/\n");
printf(" AugmentedLagrangian/SQP/InteriorPoint all ultimately require -- crashes the Octave\n");
printf(" process (SIGABRT), so no example using them can be included here.)\n");
