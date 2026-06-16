package js.octra.joctra.examples;

import js.octra.joctra.DPair;
import js.octra.joctra.DVector;
import js.octra.joctra.IPair;
import js.octra.joctra.IVector;

public class StlEx {
  static {
    System.loadLibrary("octra_jni");
  }

  public static void run() {
    System.out.println("Stl Example:");
    var v = new DVector(10, 0.0);
    for (int i = 0; i < v.size(); i++) {
      v.set(i, i * 1.1);
    }
    for (int i = 0; i < v.size(); i++) {
      System.out.println(v.get(i));
    }
    var v2 = new IVector(10, 0);
    for (int i = 0; i < v2.size(); i++) {
      v2.set(i, i * 2);
    }
    for (int i = 0; i < v2.size(); i++) {
      System.out.println(v2.get(i));
    }
    var p = new DPair(3.14, 2.71);
    System.out.println("DPair: " + p.getFirst() + ", " + p.getSecond());
    var p2 = new IPair(42, 7);
    System.out.println("IPair: " + p2.getFirst() + ", " + p2.getSecond());
  }
}
