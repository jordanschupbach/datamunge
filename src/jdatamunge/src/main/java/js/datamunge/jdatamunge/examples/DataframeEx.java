package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.IVector;
import js.datamunge.jdatamunge.SVector;
import js.datamunge.jdatamunge.DataFrame;

public class DataframeEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var sales = new DataFrame();
    sales.add_string_column("region", new SVector(new String[] {"west", "west", "east", "south", "south", "south"}));
    sales.add_string_column("product", new SVector(new String[] {"widget", "widget", "widget", "gizmo", "gizmo", "gizmo"}));
    sales.add_numeric_column("sales", new DVector(new double[] {10.0, 10.0, 14.0, 8.0, 0.0, 11.0}), new IVector(new int[] {1, 1, 1, 1, 0, 1}));
    sales.add_string_column("quarter", new SVector(new String[] {"Q1", "Q1", "Q1", "Q2", "Q2", ""}), new IVector(new int[] {1, 1, 1, 1, 1, 0}));

    System.out.println("raw data");
    System.out.println(sales.to_string());
    System.out.println();

    var cleaned = sales.drop_duplicates(new SVector(new String[] {"region", "product", "sales", "quarter"}));
    cleaned.fill_null_string("quarter", "unknown");
    cleaned.fill_null_numeric("sales", 0.0);
    System.out.println("after drop_duplicates + fill_null");
    System.out.println(cleaned.to_string());
    System.out.println();

    var selected = cleaned.select(new SVector(new String[] {"region", "sales", "quarter"})).sort_by("sales", false);
    System.out.println("selected + sorted");
    System.out.println(selected.to_string());
    System.out.println();

    var grouped = cleaned.group_by_sum(new SVector(new String[] {"region"}), new SVector(new String[] {"sales"})).sort_by("sales", false);
    System.out.println("group_by_sum(region)");
    System.out.println(grouped.to_string());
    System.out.println();

    var targets = new DataFrame();
    targets.add_string_column("region", new SVector(new String[] {"west", "east", "south"}));
    targets.add_numeric_column("target", new DVector(new double[] {18.0, 12.0, 25.0}));
    var joined = grouped.join(targets, "region", "region", true);
    System.out.println("joined with targets");
    System.out.println(joined.to_string());
    System.out.println();

    var shape = cleaned.shape();
    System.out.println("shape = (" + shape.get(0) + ", " + shape.get(1) + ")");
    System.out.println("sales count = " + cleaned.numeric_count("sales"));
    System.out.println("sales nulls = " + cleaned.numeric_null_count("sales"));
    System.out.println("sales sum = " + cleaned.numeric_sum("sales"));
    System.out.println("sales mean = " + cleaned.numeric_mean("sales"));
  }
}
