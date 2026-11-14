// CMD: just TARGET=ode_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
using Datamunge;

// Approximate C's %g: six significant digits, no padding.
static string FmtG(double x) => x.ToString("G6");

static List<double> LastState(ODESolution sol) => sol.StateAt(sol.Size() - 1);

// The time vector followed by the requested state component (0-based), one value per step.
static (List<double> Times, List<double>[] Components) SolutionSeries(ODESolution sol, params int[] components)
{
    var steps = Enumerable.Range(0, (int)sol.Size()).Select(i => (ulong)i).ToList();
    var states = steps.Select(sol.StateAt).ToList();
    return (steps.Select(sol.TimeAt).ToList(), components.Select(c => states.Select(s => s[c]).ToList()).ToArray());
}

static ODESolver Rk4Solver(double step) => new(new ODEOptions { Method = StepMethod.RK4, StepSize = step });

// NOTE: like the Lua version this example uses the built-in named systems via SolveBuiltin.

Console.WriteLine("=================== Every named built-in system (no director needed) ===================");
{
    var solver = new ODESolver();
    (string Name, double[] Params, double[] Y0)[] systems =
    [
        ("exponential_decay", [1.0], [1.0]),
        ("logistic_growth", [1.0, 1.0], [0.5]),
        ("harmonic_oscillator", [1.0], [1.0, 0.0]),
        ("van_der_pol", [1.0], [2.0, 0.0]),
        ("lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0]),
    ];
    foreach (var (name, parameters, y0) in systems)
    {
        var sol = solver.SolveBuiltin(name, parameters, y0, 0.0, 1.0);
        Console.WriteLine($"{name,-22} steps={sol.StepsTaken,-6} final state=[{string.Join(", ", LastState(sol).Select(FmtG))}]");
    }
}

Console.WriteLine("\n=================== Harmonic oscillator (energy conservation) ===================");
{
    var sol = Rk4Solver(0.01).SolveBuiltin("harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0);
    var last = LastState(sol);
    var (x, v) = (last[0], last[1]);
    Console.WriteLine($"x(20) = {x:F6} (cos(20) = {Math.Cos(20.0):F6})");
    Console.WriteLine($"energy x^2+v^2 = {x * x + v * v:F6} (should stay near 1.0)");
    var (ts, series) = SolutionSeries(sol, 0, 1);
    var p = RPlot.Plot(ts, series[0], "l", "x(t)");
    p.Lines(ts, series[1], "v(t)");
    p.Title("Harmonic Oscillator").XLabel("t").YLabel("state");
    p.SaveSvg("ode_harmonic_oscillator_csharp.svg");
    Console.WriteLine("wrote ode_harmonic_oscillator_csharp.svg");
}

Console.WriteLine("\n=================== Lorenz attractor (phase plane) ===================");
{
    var sol = Rk4Solver(0.005).SolveBuiltin("lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0], 0.0, 25.0);
    Console.WriteLine($"steps_taken = {sol.StepsTaken}");
    var (_, series) = SolutionSeries(sol, 0, 2);
    var p = RPlot.Plot(series[0], series[1], "l", "trajectory");
    p.Title("Lorenz Attractor (x-z phase plane)").XLabel("x").YLabel("z");
    p.SaveSvg("ode_lorenz_phase_plane_csharp.svg");
    Console.WriteLine("wrote ode_lorenz_phase_plane_csharp.svg");
}
