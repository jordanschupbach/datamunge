// CMD: just TARGET=dataframe_ex run-java
// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
import datamunge.DataFrame;
import java.util.List;

public class DataframeEx {
  public static void main(String[] args) {
    var sales = new DataFrame();
    sales.addStringColumn("region", List.of("west", "west", "east", "south", "south", "south"));
    sales.addStringColumn("product", List.of("widget", "widget", "widget", "gizmo", "gizmo", "gizmo"));
    sales.addNumericColumn("sales", new double[] {10.0, 10.0, 14.0, 8.0, 0.0, 11.0}, new int[] {1, 1, 1, 1, 0, 1});
    sales.addStringColumn("quarter", List.of("Q1", "Q1", "Q1", "Q2", "Q2", ""), new int[] {1, 1, 1, 1, 1, 0});

    System.out.println("raw data\n" + sales + "\n");

    var cleaned = sales.dropDuplicates(List.of("region", "product", "sales", "quarter"));
    cleaned.fillNullString("quarter", "unknown");
    cleaned.fillNullNumeric("sales", 0.0);
    System.out.println("after drop_duplicates + fill_null\n" + cleaned + "\n");

    var selected = cleaned.select(List.of("region", "sales", "quarter")).sortBy("sales", false);
    System.out.println("selected + sorted\n" + selected + "\n");

    var grouped = cleaned.groupBySum(List.of("region"), List.of("sales")).sortBy("sales", false);
    System.out.println("group_by_sum(region)\n" + grouped + "\n");

    var targets = new DataFrame();
    targets.addStringColumn("region", List.of("west", "east", "south"));
    targets.addNumericColumn("target", new double[] {18.0, 12.0, 25.0});
    System.out.println("joined with targets\n" + grouped.join(targets, "region", "region", "left") + "\n");

    long[] shape = cleaned.shape();
    System.out.println("shape = (" + shape[0] + ", " + shape[1] + ")");
    System.out.println("sales count = " + cleaned.numericCount("sales"));
    System.out.println("sales nulls = " + cleaned.numericNullCount("sales"));
    System.out.println("sales sum = " + cleaned.numericSum("sales"));
    System.out.println("sales mean = " + cleaned.numericMean("sales"));
  }
}
