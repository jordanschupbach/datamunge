package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.Dual;
import js.datamunge.jdatamunge.HyperDual;
import js.datamunge.jdatamunge.Tape;
import js.datamunge.jdatamunge.Var;

public class AutodiffEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  static Dual rosenbrockDual(Dual x0, Dual x1) {
    Dual a = new Dual(1.0, 0.0).subtract(x0);
    Dual b = x1.subtract(x0.multiply(x0));
    return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0));
  }

  static double rosenbrockValue(double x0, double x1) {
    double a = 1.0 - x0;
    double b = x1 - x0 * x0;
    return a * a + 100.0 * b * b;
  }

  static HyperDual rosenbrockHyperdual(HyperDual x0, HyperDual x1) {
    HyperDual a = new HyperDual(1.0, 0.0, 0.0, 0.0).subtract(x0);
    HyperDual b = x1.subtract(x0.multiply(x0));
    return a.multiply(a).add(b.multiply(b).multiply_scalar(100.0));
  }

  public static void run() {
    System.out.println("=================== Forward mode: scalar derivative ===================");
    var x0 = new Dual(2.0, 1.0);
    var f = x0.multiply(x0).multiply(x0).subtract(x0.multiply_scalar(2.0));
    System.out.println("f(x) = x^3 - 2x, f'(2) = " + f.derivative() + " (exact: 10)");

    System.out.println("\n=================== Reverse mode: build a graph by hand ===================");
    var tape = new Tape();
    var a = new Var(tape, 2.0);
    var b = new Var(tape, 3.0);
    var y = a.multiply(b).add(a.sin());
    System.out.println("y = a*b + sin(a) at a=2, b=3 -> y = " + y.value());
    var adjoint = tape.backward(y);
    System.out.println("dy/da = " + adjoint.get((int) a.index()) + " (exact: b + cos(a))");
    System.out.println("dy/db = " + adjoint.get((int) b.index()) + " (exact: a)");

    System.out.println("\n=================== Forward vs reverse mode agree on the Rosenbrock function ===================");
    double px = 0.0;
    double py = 0.0;

    double gx = rosenbrockDual(new Dual(px, 1.0), new Dual(py, 0.0)).derivative();
    double gy = rosenbrockDual(new Dual(px, 0.0), new Dual(py, 1.0)).derivative();

    var tape2 = new Tape();
    var vx = new Var(tape2, px);
    var vy = new Var(tape2, py);
    var va = new Var(tape2, 1.0).subtract(vx);
    var vb = vy.subtract(vx.multiply(vx));
    var vf = va.multiply(va).add(vb.multiply(vb).multiply_scalar(100.0));
    var gradRev = tape2.backward(vf);

    System.out.println("f(0,0) = " + rosenbrockValue(px, py));
    System.out.println("gradient (forward mode): [" + gx + ", " + gy + "]");
    System.out.println("gradient (reverse mode): [" + gradRev.get((int) vx.index()) + ", " + gradRev.get((int) vy.index()) + "]");

    System.out.println("\n=================== Jacobian of a vector-valued function ===================");
    double vxVal = 2.0;
    double vyVal = 3.0;

    double[][] jac = new double[3][2];
    double[][] cols = {{0, 1.0, 0.0}, {1, 0.0, 1.0}};
    for (double[] c : cols) {
      int col = (int) c[0];
      double seedX = c[1];
      double seedY = c[2];
      var ox = new Dual(vxVal, seedX).multiply(new Dual(vxVal, seedX));
      var oy = new Dual(vxVal, seedX).multiply(new Dual(vyVal, seedY));
      var oz = new Dual(vyVal, seedY).multiply(new Dual(vyVal, seedY)).multiply(new Dual(vyVal, seedY));
      jac[0][col] = ox.derivative();
      jac[1][col] = oy.derivative();
      jac[2][col] = oz.derivative();
    }
    System.out.println("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:");
    for (int row = 0; row < 3; row++) {
      System.out.println("  [" + jac[row][0] + ", " + jac[row][1] + "]");
    }

    System.out.println("\n=================== Hessian via second-order forward mode (HyperDual) ===================");
    double mx = 1.0;
    double my = 1.0;

    double[][] seeds = {{1.0, 0.0}, {0.0, 1.0}};
    double[][] h = new double[2][2];
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        double[] e1 = seeds[i];
        double[] e2 = seeds[j];
        var hx = new HyperDual(mx, e1[0], e2[0], 0.0);
        var hy = new HyperDual(my, e1[1], e2[1], 0.0);
        h[i][j] = rosenbrockHyperdual(hx, hy).eps1eps2();
      }
    }
    System.out.println("Hessian of the Rosenbrock function at its minimum (1,1):");
    for (int row = 0; row < 2; row++) {
      System.out.println("  [" + h[row][0] + ", " + h[row][1] + "]");
    }

    System.out.println("\n=================== Gradient descent driven by reverse-mode gradients ===================");
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
      pointX -= learningRate * grad.get((int) vx2.index());
      pointY -= learningRate * grad.get((int) vy2.index());
      if (step == 0 || step == nSteps - 1) {
        System.out.println("step " + step + ": loss = " + loss + ", x = [" + pointX + ", " + pointY + "]");
      }
    }
    System.out.println("(true minimum is at [1, 1] with loss 0)");
  }
}
