// CMD: just TARGET=autodiff_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
using Datamunge;

// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// Var.Index() is the Var's position in the adjoint vector Tape.Backward returns.

static Dual RosenbrockDual(Dual x0, Dual x1)
{
    var a = new Dual(1.0, 0.0).Subtract(x0);
    var b = x1.Subtract(x0.Multiply(x0));
    return a.Multiply(a).Add(b.Multiply(b).MultiplyScalar(100.0));
}

static double RosenbrockValue(double x0, double x1) => Math.Pow(1.0 - x0, 2) + 100.0 * Math.Pow(x1 - x0 * x0, 2);

// Build the Rosenbrock graph on tape; returns the output Var.
static Var RosenbrockVar(Tape tape, Var x0, Var x1)
{
    var a = new Var(tape, 1.0).Subtract(x0);
    var b = x1.Subtract(x0.Multiply(x0));
    return a.Multiply(a).Add(b.Multiply(b).MultiplyScalar(100.0));
}

static HyperDual RosenbrockHyperDual(HyperDual x0, HyperDual x1)
{
    var a = new HyperDual(1.0, 0.0, 0.0, 0.0).Subtract(x0);
    var b = x1.Subtract(x0.Multiply(x0));
    return a.Multiply(a).Add(b.Multiply(b).MultiplyScalar(100.0));
}

static double Adjoint(List<double> adjoints, Var v) => adjoints[(int)v.Index()];

Console.WriteLine("=================== Forward mode: scalar derivative ===================");
{
    var x0 = new Dual(2.0, 1.0);  // seed derivative = 1 to read df/dx directly
    var f = x0.Multiply(x0).Multiply(x0).Subtract(x0.MultiplyScalar(2.0));  // x^3 - 2x
    Console.WriteLine($"f(x) = x^3 - 2x, f'(2) = {f.Derivative()} (exact: 10)");
}

Console.WriteLine("\n=================== Reverse mode: build a graph by hand ===================");
{
    var tape = new Tape();
    var a = new Var(tape, 2.0);
    var b = new Var(tape, 3.0);
    var y = a.Multiply(b).Add(a.Sin());
    Console.WriteLine($"y = a*b + sin(a) at a=2, b=3 -> y = {y.Value()}");
    var adjoint = tape.Backward(y);
    Console.WriteLine($"dy/da = {Adjoint(adjoint, a)} (exact: b + cos(a))");
    Console.WriteLine($"dy/db = {Adjoint(adjoint, b)} (exact: a)");
}

Console.WriteLine("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================");
{
    double px = 0.0, py = 0.0;
    // Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
    var gx = RosenbrockDual(new Dual(px, 1.0), new Dual(py, 0.0)).Derivative();
    var gy = RosenbrockDual(new Dual(px, 0.0), new Dual(py, 1.0)).Derivative();
    // Reverse mode: one Tape/Var pass computes every partial at once.
    var tape = new Tape();
    var vx = new Var(tape, px);
    var vy = new Var(tape, py);
    var grad = tape.Backward(RosenbrockVar(tape, vx, vy));
    Console.WriteLine($"f(0,0) = {RosenbrockValue(px, py)}");
    Console.WriteLine($"gradient (forward mode): [{gx}, {gy}]");
    Console.WriteLine($"gradient (reverse mode): [{Adjoint(grad, vx)}, {Adjoint(grad, vy)}]");
}

Console.WriteLine("\n=================== Jacobian of a vector-valued function ===================");
// f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
{
    static Dual[] VectorFn(Dual x, Dual y) => [x.Multiply(x), x.Multiply(y), y.Multiply(y).Multiply(y)];
    var jac = new double[3, 2];
    foreach (var (col, sx, sy) in new[] { (0, 1.0, 0.0), (1, 0.0, 1.0) })
    {
        var outputs = VectorFn(new Dual(2.0, sx), new Dual(3.0, sy));
        for (var row = 0; row < outputs.Length; row++)
            jac[row, col] = outputs[row].Derivative();
    }
    Console.WriteLine("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
    for (var row = 0; row < 3; row++)
        Console.WriteLine($"  [{jac[row, 0]}, {jac[row, 1]}]");
}

Console.WriteLine("\n=================== Hessian via second-order forward mode (HyperDual) ===================");
// One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
{
    double mx = 1.0, my = 1.0;  // the Rosenbrock function's minimum
    (double X, double Y)[] seeds = [(1.0, 0.0), (0.0, 1.0)];
    var h = new double[2, 2];
    for (var i = 0; i < 2; i++)
        for (var j = 0; j < 2; j++)
        {
            var (e1, e2) = (seeds[i], seeds[j]);
            h[i, j] = RosenbrockHyperDual(new HyperDual(mx, e1.X, e2.X, 0.0), new HyperDual(my, e1.Y, e2.Y, 0.0)).Eps1eps2();
        }
    Console.WriteLine("Hessian of the Rosenbrock function at its minimum (1,1):");
    for (var i = 0; i < 2; i++)
        Console.WriteLine($"  [{h[i, 0]}, {h[i, 1]}]");
}

Console.WriteLine("\n=================== Gradient descent driven by reverse-mode gradients ===================");
{
    double[] point = [-1.2, 1.0];  // the classic Rosenbrock start
    const double learningRate = 0.001;
    const int nSteps = 2000;
    for (var step = 0; step < nSteps; step++)
    {
        var tape = new Tape();
        var vx = new Var(tape, point[0]);
        var vy = new Var(tape, point[1]);
        var vf = RosenbrockVar(tape, vx, vy);
        var grad = tape.Backward(vf);
        var loss = vf.Value();
        point[0] -= learningRate * Adjoint(grad, vx);
        point[1] -= learningRate * Adjoint(grad, vy);
        if (step == 0 || step == nSteps - 1)
            Console.WriteLine($"step {step}: loss = {loss}, x = [{point[0]}, {point[1]}]");
    }
    Console.WriteLine("(true minimum is at [1, 1] with loss 0)");
}
