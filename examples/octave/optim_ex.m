1;

datamunge;

% NOTE: Octave's SWIG director glue has a runtime bug where any director method returning
% std::vector<double> (e.g. DifferentiableFunction::gradient()) crashes the interpreter
% (uncaught Swig::DirectorPureVirtualException / SIGABRT), even in an isolated direct call
% with a real handler assigned. ArbitraryFunction::evaluate() (returns a plain double) works
% fine. So only the derivative-free (ArbitraryFunction) portion of this example is portable
% to Octave -- the DifferentiableFunction-based portion (GradientDescent/Adam/LBFGS/SGD on
% QuadraticBowl/RosenbrockFn/LinearRegressionLoss) is omitted; see
% datamunge_octave_bindings.md for the full diagnosis.

function y = bumpy_evaluate(self, x)
  bowl = (x{1} - 3.0)^2 + (x{2} + 1.0)^2;
  ripples = 5.0 * sin(x{1}) * cos(x{2});
  y = bowl + ripples;
endfunction

function y = rastrigin_evaluate(self, x)
  total = 10.0 * numel(x);
  for i = 1:numel(x)
    xi = x{i};
    total = total + xi * xi - 10.0 * cos(2.0 * pi * xi);
  end
  y = total;
endfunction

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

printf("=================== ArbitraryFunction: derivative-free SimulatedAnnealing ===================\n");
f = ArbitraryFunction();
f.evaluate = @bumpy_evaluate;
x = dv([0.0, 0.0]);
sa_options = SimulatedAnnealingOptions();
SimulatedAnnealingOptions_initial_temperature_set(sa_options, 10.0);
SimulatedAnnealingOptions_cooling_rate_set(sa_options, 0.999);
SimulatedAnnealingOptions_max_iterations_set(sa_options, 20000);
SimulatedAnnealingOptions_step_std_dev_set(sa_options, 0.5);
value = SimulatedAnnealing_optimize(SimulatedAnnealing(sa_options), f, x);
printf("f=%g x=[%g,%g] (found without ever computing a gradient)\n", value, DVector___paren__(x, 0), DVector___paren__(x, 1));

printf("\n=================== Population-based methods on the Rastrigin function ===================\n");
lower = dv([-5.12, -5.12]);
upper = dv([5.12, 5.12]);

f = ArbitraryFunction();
f.evaluate = @rastrigin_evaluate;
x = dv([3.0, -4.0]);
pso_options = PSOOptions();
PSOOptions_topology_set(pso_options, "global");
PSOOptions_inertia_strategy_set(pso_options, "constant");
value = PSO_optimize(PSO(pso_options), f, x, lower, upper);
printf("PSO (global topology, constant inertia):    f=%g x=[%g,%g]\n", value, DVector___paren__(x, 0), DVector___paren__(x, 1));

f = ArbitraryFunction();
f.evaluate = @rastrigin_evaluate;
x = dv([3.0, -4.0]);
pso_options = PSOOptions();
PSOOptions_topology_set(pso_options, "ring");
PSOOptions_inertia_strategy_set(pso_options, "linear_decay");
value = PSO_optimize(PSO(pso_options), f, x, lower, upper);
printf("PSO (ring topology, linear-decay inertia):  f=%g x=[%g,%g]\n", value, DVector___paren__(x, 0), DVector___paren__(x, 1));

f = ArbitraryFunction();
f.evaluate = @rastrigin_evaluate;
x = dv([3.0, -4.0]);
de_options = DEOptions();
DEOptions_mutation_strategy_set(de_options, "rand1");
DEOptions_crossover_strategy_set(de_options, "binomial");
value = DifferentialEvolution_optimize(DifferentialEvolution(de_options), f, x, lower, upper);
printf("DE (rand1/binomial):                        f=%g x=[%g,%g]\n", value, DVector___paren__(x, 0), DVector___paren__(x, 1));

f = ArbitraryFunction();
f.evaluate = @rastrigin_evaluate;
x = dv([3.0, -4.0]);
de_options = DEOptions();
DEOptions_mutation_strategy_set(de_options, "best1");
DEOptions_crossover_strategy_set(de_options, "exponential");
value = DifferentialEvolution_optimize(DifferentialEvolution(de_options), f, x, lower, upper);
printf("DE (best1/exponential):                     f=%g x=[%g,%g]\n", value, DVector___paren__(x, 0), DVector___paren__(x, 1));

f = ArbitraryFunction();
f.evaluate = @rastrigin_evaluate;
x = dv([3.0, -4.0]);
ga_options = GAOptions();
GAOptions_selection_strategy_set(ga_options, "tournament");
GAOptions_crossover_strategy_set(ga_options, "blend");
value = GeneticAlgorithm_optimize(GeneticAlgorithm(ga_options), f, x, lower, upper);
printf("GA (tournament/blend, elitism on):           f=%g x=[%g,%g]\n", value, DVector___paren__(x, 0), DVector___paren__(x, 1));

f = ArbitraryFunction();
f.evaluate = @rastrigin_evaluate;
x = dv([3.0, -4.0]);
ga_options = GAOptions();
GAOptions_selection_strategy_set(ga_options, "rank");
GAOptions_crossover_strategy_set(ga_options, "uniform");
GAOptions_elitism_set(ga_options, false);
value = GeneticAlgorithm_optimize(GeneticAlgorithm(ga_options), f, x, lower, upper);
printf("GA (rank/uniform, elitism off):              f=%g x=[%g,%g] (true minimum: f=0 at [0, 0])\n", value, DVector___paren__(x, 0), DVector___paren__(x, 1));
