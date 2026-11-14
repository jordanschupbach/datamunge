// CMD: just TARGET=autodiff_ex run-java
// Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
import datamunge.Dual;
import datamunge.HyperDual;
import datamunge.Tape;
import datamunge.Var;
import java.util.ArrayList;
import java.util.List;

// Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
// (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
// bound. This port demonstrates the same ideas with the bound named-method API.
// var.index() is the Var's position in the adjoint array tape.backward(output) returns.
public class AutodiffEx {
  static Dual rosenbrockDual(Dual x0, Dual x1) {
    var a = new Dual(1.0, 0.0).subtract(x0);
    var b = x1.subtract(x0.multiply(x0));
    return a.multiply(a).add(b.multiply(b).multiplyScalar(100.0));
  }

  static double rosenbrockValue(double x0, double x1) {
    return Math.pow(1.0 - x0, 2) + 100.0 * Math.pow(x1 - x0 * x0, 2);
  }

  // Build the Rosenbrock graph on tape; returns the output Var.
  static Var rosenbrockVar(Tape tape, Var x0, Var x1) {
    var a = new Var(tape, 1.0).subtract(x0);
    var b = x1.subtract(x0.multiply(x0));
    return a.multiply(a).add(b.multiply(b).multiplyScalar(100.0));
  }

  static HyperDual rosenbrockHyperDual(HyperDual x0, HyperDual x1) {
    var a = new HyperDual(1.0, 0.0, 0.0, 0.0).subtract(x0);
    var b = x1.subtract(x0.multiply(x0));
    return a.multiply(a).add(b.multiply(b).multiplyScalar(100.0));
  }

  static double adjoint(double[] adjoints, Var v) {
    return adjoints[(int) v.index()];
  }

  static List<Dual> vectorFn(Dual x, Dual y) {
    return List.of(x.multiply(x), x.multiply(y), y.multiply(y).multiply(y));
  }

  static String join(double[] values) {
    var parts = new ArrayList<String>();
    for (double v : values) {
      parts.add(Double.toString(v));
    }
    return "[" + String.join(", ", parts) + "]";
  }

  public static void main(String[] args) {
    System.out.println(
        "=================== Forward mode: scalar derivative ===================");
    var x0 = new Dual(2.0, 1.0); // seed derivative = 1 to read df/dx directly
    var f = x0.multiply(x0).multiply(x0).subtract(x0.multiplyScalar(2.0)); // x^3 - 2x
    System.out.println("f(x) = x^3 - 2x, f'(2) = " + f.derivative() + " (exact: 10)");

    System.out.println(
        "\n=================== Reverse mode: build a graph by hand ===================");
    var tape = new Tape();
    var a = new Var(tape, 2.0);
    var b = new Var(tape, 3.0);
    var y = a.multiply(b).add(a.sin());
    System.out.println("y = a*b + sin(a) at a=2, b=3 -> y = " + y.value());
    double[] adj = tape.backward(y);
    System.out.println("dy/da = " + adjoint(adj, a) + " (exact: b + cos(a))");
    System.out.println("dy/db = " + adjoint(adj, b) + " (exact: a)");

    System.out.println(
        "\n=================== Forward vs reverse mode agree on the Rosenbrock function"
            + " ===================");
    double px = 0.0;
    double py = 0.0;
    // Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
    double gx = rosenbrockDual(new Dual(px, 1.0), new Dual(py, 0.0)).derivative();
    double gy = rosenbrockDual(new Dual(px, 0.0), new Dual(py, 1.0)).derivative();
    // Reverse mode: one Tape/Var pass computes every partial at once.
    tape = new Tape();
    var vx = new Var(tape, px);
    var vy = new Var(tape, py);
    double[] grad = tape.backward(rosenbrockVar(tape, vx, vy));
    System.out.println("f(0,0) = " + rosenbrockValue(px, py));
    System.out.println("gradient (forward mode): " + join(new double[] {gx, gy}));
    System.out.println(
        "gradient (reverse mode): " + join(new double[] {adjoint(grad, vx), adjoint(grad, vy)}));

    System.out.println(
        "\n=================== Jacobian of a vector-valued function ===================");
    // f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
    double[][] jac = new double[3][2];
    double[][] seeds = {{1.0, 0.0}, {0.0, 1.0}};
    for (int col = 0; col < 2; col++) {
      var outs = vectorFn(new Dual(2.0, seeds[col][0]), new Dual(3.0, seeds[col][1]));
      for (int row = 0; row < outs.size(); row++) {
        jac[row][col] = outs.get(row).derivative();
      }
    }
    System.out.println("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
    for (double[] r : jac) {
      System.out.println("  " + join(r));
    }

    System.out.println(
        "\n=================== Hessian via second-order forward mode (HyperDual)"
            + " ===================");
    // One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
    double mx = 1.0;
    double my = 1.0; // the Rosenbrock function's minimum
    double[][] h = new double[2][2];
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        double[] e1 = seeds[i];
        double[] e2 = seeds[j];
        h[i][j] =
            rosenbrockHyperDual(
                    new HyperDual(mx, e1[0], e2[0], 0.0), new HyperDual(my, e1[1], e2[1], 0.0))
                .eps1eps2();
      }
    }
    System.out.println("Hessian of the Rosenbrock function at its minimum (1,1):");
    for (double[] r : h) {
      System.out.println("  " + join(r));
    }

    System.out.println(
        "\n=================== Gradient descent driven by reverse-mode gradients"
            + " ===================");
    double[] point = {-1.2, 1.0}; // the classic Rosenbrock start
    double learningRate = 0.001;
    int nSteps = 2000;
    for (int step = 0; step < nSteps; step++) {
      try (var stepTape = new Tape()) {
        var sx = new Var(stepTape, point[0]);
        var sy = new Var(stepTape, point[1]);
        var vf = rosenbrockVar(stepTape, sx, sy);
        double[] g = stepTape.backward(vf);
        double loss = vf.value();
        point[0] = point[0] - learningRate * adjoint(g, sx);
        point[1] = point[1] - learningRate * adjoint(g, sy);
        if (step == 0 || step == nSteps - 1) {
          System.out.println("step " + step + ": loss = " + loss + ", x = " + join(point));
        }
      }
    }
    System.out.println("(true minimum is at [1, 1] with loss 0)");
  }
}
