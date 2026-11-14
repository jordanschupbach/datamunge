// Smoke tests of the datamunge Java binding (just test-java), run as a single-file source launch with
// datamunge.jar on the class path and DATAMUNGE_JAVA_LIBRARY naming libdatamunge_java. Argument: the
// generated source root (default src/datamungejava/java), for the documentation checks.
import datamunge.Callback;
import datamunge.DataFrame;
import datamunge.Datamunge;
import datamunge.NativeError;
import datamunge.Pair;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TestDatamunge {
  static int checks = 0;

  static void check(boolean ok, String message) {
    ++checks;
    if (!ok) {
      System.err.println("check failed: " + message);
      System.exit(1);
    }
  }

  // A Java subclass overriding the C++ Callback's virtual call() (a director).
  static final class Doubler extends Callback {
    @Override
    public double call(double x) {
      return 2 * x;
    }
  }

  // An override that throws a Java exception of its own.
  static final class Failing extends Callback {
    @Override
    public double call(double x) {
      throw new IllegalStateException("from Java");
    }
  }

  public static void main(String[] args) throws Exception {
    // Smoke test: call into the native library.
    Datamunge.hello();

    // std::vector<double> and std::pair<double, double> round trips.
    double[] v = Datamunge.makeDvector(1.0, 2.0, 3.0);
    check(java.util.Arrays.equals(v, new double[] {1.0, 2.0, 3.0}), "std::vector<double> is a double[]");
    check(Datamunge.sumDvector(v) == 6.0, "sumDvector");
    check(Datamunge.sumDvector(new double[] {0.5, 1.5}) == 2.0, "arrays convert to std::vector");
    Pair<Double, Double> p = Datamunge.makeDpair(1.25, 2.75);
    check(p.first() == 1.25 && p.second() == 2.75, "std::pair is the Pair record");
    check(Datamunge.sumDpair(p) == 4.0, "sumDpair");

    // The C++ Callback class, and a Java subclass overriding its virtual call().
    try (Callback cb = new Callback()) {
      check(Datamunge.callWithCallback(3.0, cb) == 3.0, "native Callback");
      check(Datamunge.sumDvector(Datamunge.mapDvectorWithCallback(new double[] {1, 2, 3}, cb)) == 6.0, "native map");
    }
    try (Doubler doubler = new Doubler()) {
      check(Datamunge.callWithCallback(3.0, doubler) == 6.0, "the Java override runs");
      check(Datamunge.sumDvector(Datamunge.mapDvectorWithCallback(new double[] {1, 2, 3}, doubler)) == 12.0,
          "override from C++ loops");
    }

    // A Java exception inside an override reaches the caller unchanged.
    try (Failing failing = new Failing()) {
      boolean propagated = false;
      try {
        Datamunge.callWithCallback(1.0, failing);
      } catch (IllegalStateException error) {
        propagated = "from Java".equals(error.getMessage());
      }
      check(propagated, "the Java exception propagates");
    }

    // C++ exceptions raise typed Java exceptions implementing NativeError.
    boolean typed = false;
    try {
      Datamunge.runif(1.0, 0.0, 3);
    } catch (NativeError.InvalidArgument error) {
      typed = error.kind().equals("invalid_argument");
    }
    check(typed, "std::invalid_argument is NativeError.InvalidArgument");

    // Classes and methods; close() releases the C++ object.
    DataFrame frame = new DataFrame();
    frame.addNumericColumn("x", new double[] {1.0, 2.0, 3.0});
    check(frame.nrows() == 3, "DataFrame");
    check(frame.columns().equals(List.of("x")), "std::vector<std::string> is a List<String>");
    frame.close();
    boolean closed = false;
    try {
      frame.nrows();
    } catch (datamunge.NativeException error) {
      closed = error.getMessage().contains("closed");
    }
    check(closed, "a closed object refuses calls");

    // The committed sources carry the C++ documentation as Javadoc.
    Path root = Path.of(args.length > 0 ? args[0] : "src/datamungejava/java");
    String functions = Files.readString(root.resolve("datamunge/Datamunge.java"));
    Matcher seql = Pattern.compile("/\\*\\*((?:(?!\\*/).)*)\\*/\\s*public static double\\[\\] seql\\(", Pattern.DOTALL)
        .matcher(functions);
    check(seql.find() && seql.group(1).contains("evenly spaced"), "function Javadoc");
    String dataFrame = Files.readString(root.resolve("datamunge/DataFrame.java"));
    Matcher sortBy = Pattern.compile("/\\*\\*((?:(?!\\*/).)*)\\*/\\s*public DataFrame sortBy\\(String columnName\\)", Pattern.DOTALL)
        .matcher(dataFrame);
    check(sortBy.find() && sortBy.group(1).contains("@param columnName"), "method Javadoc");

    System.out.println("TestDatamunge.java: all " + checks + " checks passed");
  }
}
