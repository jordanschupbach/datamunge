using System;

class Program {
  static Dual RosenbrockDual(Dual x0, Dual x1) {
    var a = new Dual(1.0, 0.0).subtract(x0);
    var b = x1.subtract(x0.multiply(x0));
    return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0));
  }

  static double RosenbrockValue(double x0, double x1) {
    double a = 1.0 - x0;
    double b = x1 - x0 * x0;
    return a * a + 100.0 * b * b;
  }

  static HyperDual RosenbrockHyperdual(HyperDual x0, HyperDual x1) {
    var a = new HyperDual(1.0, 0.0, 0.0, 0.0).subtract(x0);
    var b = x1.subtract(x0.multiply(x0));
    return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0));
  }

  static void Main() {
    Console.WriteLine("=================== Forward mode: scalar derivative ===================");
    var x0 = new Dual(2.0, 1.0);
    var f = x0.multiply(x0).multiply(x0).subtract(x0.multiply_scalar(2.0));
    Console.WriteLine($"f(x) = x^3 - 2x, f'(2) = {f.derivative()} (exact: 10)");

    Console.WriteLine("\n=================== Reverse mode: build a graph by hand ===================");
    var tape = new Tape();
    var a = new Var(tape, 2.0);
    var b = new Var(tape, 3.0);
    var y = a.multiply(b).add(a.sin());
    Console.WriteLine($"y = a*b + sin(a) at a=2, b=3 -> y = {y.value()}");
    var adjoint = tape.backward(y);
    Console.WriteLine($"dy/da = {adjoint[(int)a.index()]} (exact: b + cos(a))");
    Console.WriteLine($"dy/db = {adjoint[(int)b.index()]} (exact: a)");

    Console.WriteLine("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================");
    double px = 0.0;
    double py = 0.0;

    double gx = RosenbrockDual(new Dual(px, 1.0), new Dual(py, 0.0)).derivative();
    double gy = RosenbrockDual(new Dual(px, 0.0), new Dual(py, 1.0)).derivative();

    var tape2 = new Tape();
    var vx = new Var(tape2, px);
    var vy = new Var(tape2, py);
    var va = new Var(tape2, 1.0).subtract(vx);
    var vb = vy.subtract(vx.multiply(vx));
    var vf = va.multiply(va).add(vb.multiply(vb).multiply_scalar(100.0));
    var gradRev = tape2.backward(vf);

    Console.WriteLine($"f(0,0) = {RosenbrockValue(px, py)}");
    Console.WriteLine($"gradient (forward mode): [{gx}, {gy}]");
    Console.WriteLine($"gradient (reverse mode): [{gradRev[(int)vx.index()]}, {gradRev[(int)vy.index()]}]");

    Console.WriteLine("\n=================== Jacobian of a vector-valued function ===================");
    double vxVal = 2.0;
    double vyVal = 3.0;

    var jac = new double[3, 2];
    var cols = new (int col, double seedX, double seedY)[] { (0, 1.0, 0.0), (1, 0.0, 1.0) };
    foreach (var c in cols) {
      var ox = new Dual(vxVal, c.seedX).multiply(new Dual(vxVal, c.seedX));
      var oy = new Dual(vxVal, c.seedX).multiply(new Dual(vyVal, c.seedY));
      var oz = new Dual(vyVal, c.seedY).multiply(new Dual(vyVal, c.seedY)).multiply(new Dual(vyVal, c.seedY));
      jac[0, c.col] = ox.derivative();
      jac[1, c.col] = oy.derivative();
      jac[2, c.col] = oz.derivative();
    }
    Console.WriteLine("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
    for (int row = 0; row < 3; row++) {
      Console.WriteLine($"  [{jac[row, 0]}, {jac[row, 1]}]");
    }

    Console.WriteLine("\n=================== Hessian via second-order forward mode (HyperDual) ===================");
    double mx = 1.0;
    double my = 1.0;

    var seeds = new (double, double)[] { (1.0, 0.0), (0.0, 1.0) };
    var h = new double[2, 2];
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        var e1 = seeds[i];
        var e2 = seeds[j];
        var hx = new HyperDual(mx, e1.Item1, e2.Item1, 0.0);
        var hy = new HyperDual(my, e1.Item2, e2.Item2, 0.0);
        h[i, j] = RosenbrockHyperdual(hx, hy).eps1eps2();
      }
    }
    Console.WriteLine("Hessian of the Rosenbrock function at its minimum (1,1):");
    for (int row = 0; row < 2; row++) {
      Console.WriteLine($"  [{h[row, 0]}, {h[row, 1]}]");
    }

    Console.WriteLine("\n=================== Gradient descent driven by reverse-mode gradients ===================");
    double pointX = -1.2;
    double pointY = 1.0;
    double learningRate = 0.001;
    int nSteps = 2000;
    for (int step = 0; step < nSteps; step++) {
      var t = new Tape();
      var vx2 = new Var(t, pointX);
      var vy2 = new Var(t, pointY);
      var va2 = new Var(t, 1.0).subtract(vx2);
      var vb2 = vy2.subtract(vx2.multiply(vx2));
      var vf2 = va2.multiply(va2).add(vb2.multiply(vb2).multiply_scalar(100.0));
      var grad = t.backward(vf2);
      double loss = vf2.value();
      pointX -= learningRate * grad[(int)vx2.index()];
      pointY -= learningRate * grad[(int)vy2.index()];
      if (step == 0 || step == nSteps - 1) {
        Console.WriteLine($"step {step}: loss = {loss}, x = [{pointX}, {pointY}]");
      }
    }
    Console.WriteLine("(true minimum is at [1, 1] with loss 0)");
  }
}
