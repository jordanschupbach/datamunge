// CMD: just TARGET=optim_advanced_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
// AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from C#.
//
// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From C#, derive from the interface class and override its virtual
// methods; the optimizer then calls the C# overrides. Coordinates are updated in place through
// a ref parameter (the C++ parameter is a non-const std::vector<double>&).
using Datamunge;

Console.WriteLine("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================");
// Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
// then check the acquisition function favors an unexplored region.
{
    double[] xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    var ys = xs.Select(x => (x - 1) * (x - 1)).ToArray();
    var surrogate = new RBFGaussianProcessSurrogate(1.0, 1e-6);
    surrogate.Fit(xs.Select(x => new[] { x }), ys);
    var incumbent = Math.Min(0.0, ys.Min());  // best observed value so far
    Console.WriteLine($"incumbent (best observed y): {incumbent}");
    foreach (var x in new[] { 0.25, 1.0, 2.5 })
        Console.WriteLine($"acquisition(x={x:F2}) = {surrogate.Acquisition([x], incumbent):F6}");
}

Console.WriteLine("\n=================== Options structs + optimizer construction (all 10 new classes) ===================");
// Every new Options struct and optimizer class constructs cleanly, proving
// the binding for the whole batch is wired correctly.
var pg = new ProximalGradient(new ProximalGradientOptions { StepSize = 0.05 });
Console.WriteLine($"ProximalGradient: {pg}");
var fista = new FISTA(new FISTAOptions());
Console.WriteLine($"FISTA: {fista}");
var lm = new LevenbergMarquardt(new LevenbergMarquardtOptions { MaxIterations = 200 });
Console.WriteLine($"LevenbergMarquardt: {lm}");
var newton = new Newton(new NewtonOptions());
Console.WriteLine($"Newton: {newton}");
var trn = new TrustRegionNewton(new TrustRegionNewtonOptions());
Console.WriteLine($"TrustRegionNewton: {trn}");
var al = new AugmentedLagrangian(new AugmentedLagrangianOptions());
Console.WriteLine($"AugmentedLagrangian: {al}");
var sqp = new SQP(new SQPOptions());
Console.WriteLine($"SQP: {sqp}");
var ip = new InteriorPoint(new InteriorPointOptions());
Console.WriteLine($"InteriorPoint: {ip}");
var bo = new BayesianOptimization(new BayesianOptimizationOptions { InitialSamples = 8 });
Console.WriteLine($"BayesianOptimization: {bo}");

Console.WriteLine("\n=================== C# objectives (subclassing the C++ interfaces) ===================");
// Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
{
    List<double> start = [5.0, 5.0];
    var value = newton.Optimize(new Quadratic(), ref start);
    Console.WriteLine($"Newton: f = {value:G6} at ({start[0]:F6}, {start[1]:F6}) (exact minimum 0 at (1, -2))");
}

// Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
{
    List<double> parameters = [1.0, 0.1];
    var sse = lm.Optimize(new Exponential([0.0, 0.5, 1.0, 1.5, 2.0]), ref parameters);
    Console.WriteLine($"LevenbergMarquardt: residual = {sse:G3}, a = {parameters[0]:F6}, b = {parameters[1]:F6} (exact: 2, 0.5)");
}

// An interface method a subclass leaves out throws when the optimizer calls it.
{
    Exception? err = null;
    try
    {
        List<double> origin = [0.0, 0.0];
        newton.Optimize(new Incomplete(), ref origin);
    }
    catch (Exception e)
    {
        err = e;
    }
    Console.WriteLine($"missing gradient(): ok={err is null}, error mentions it: {err is not null && err.Message.Contains("gradient")}");
}

class Quadratic : HessianFunction
{
    public override double Evaluate(IEnumerable<double> coordinates)
    {
        var c = coordinates.ToList();
        return Math.Pow(c[0] - 1, 2) + 10 * Math.Pow(c[1] + 2, 2);
    }

    public override List<double> Gradient(IEnumerable<double> coordinates)
    {
        var c = coordinates.ToList();
        return [2 * (c[0] - 1), 20 * (c[1] + 2)];
    }

    public override List<List<double>> Hessian(IEnumerable<double> coordinates) => [[2.0, 0.0], [0.0, 20.0]];
}

class Exponential(double[] ts) : ResidualFunction
{
    public override List<double> Residuals(IEnumerable<double> coordinates)
    {
        var p = coordinates.ToList();
        return [.. ts.Select(t => p[0] * Math.Exp(p[1] * t) - 2 * Math.Exp(0.5 * t))];
    }

    public override List<List<double>> Jacobian(IEnumerable<double> coordinates)
    {
        var p = coordinates.ToList();
        return [.. ts.Select(t => new List<double> { Math.Exp(p[1] * t), p[0] * t * Math.Exp(p[1] * t) })];
    }
}

class Incomplete : HessianFunction
{
    public override double Evaluate(IEnumerable<double> coordinates) => 0.0;
}
