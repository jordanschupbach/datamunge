// CMD: just TARGET=gis_ex run-java
// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
import datamunge.ShapeLayer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout ShapeLayer.read expects, then reads them back through the public API.
public class GisEx {
  // A shape is a list of rings; a ring is an array of {x, y} points.
  record Shape(double[][][] rings) {}

  // Bounding box {minx, miny, maxx, maxy} of every point in the given shapes.
  static double[] bounds(List<Shape> shapes) {
    double minX = Double.POSITIVE_INFINITY;
    double minY = Double.POSITIVE_INFINITY;
    double maxX = Double.NEGATIVE_INFINITY;
    double maxY = Double.NEGATIVE_INFINITY;
    for (Shape shape : shapes) {
      for (double[][] ring : shape.rings()) {
        for (double[] p : ring) {
          minX = Math.min(minX, p[0]);
          minY = Math.min(minY, p[1]);
          maxX = Math.max(maxX, p[0]);
          maxY = Math.max(maxY, p[1]);
        }
      }
    }
    return new double[] {minX, minY, maxX, maxY};
  }

  static ByteBuffer little(int size) {
    return ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
  }

  // Bytes of one Polygon record's contents (without the record header).
  static byte[] polygonRecord(Shape shape) {
    double[][][] rings = shape.rings();
    int total = Arrays.stream(rings).mapToInt(r -> r.length).sum();
    ByteBuffer buf = little(4 + 32 + 8 + 4 * rings.length + 16 * total);
    // Shape type 5 = Polygon, then bounds, part/point counts, part starts and the points.
    buf.putInt(5);
    for (double b : bounds(List.of(shape))) {
      buf.putDouble(b);
    }
    buf.putInt(rings.length).putInt(total);
    int start = 0;
    for (double[][] ring : rings) {
      buf.putInt(start);
      start += ring.length;
    }
    for (double[][] ring : rings) {
      for (double[] p : ring) {
        buf.putDouble(p[0]).putDouble(p[1]);
      }
    }
    return buf.array();
  }

  static void writeCountiesShp(String path, List<Shape> shapes) throws IOException {
    List<byte[]> contents = shapes.stream().map(GisEx::polygonRecord).toList();
    int totalWords = contents.stream().mapToInt(c -> 4 + c.length / 2).sum();
    var out = new ByteArrayOutputStream();
    ByteBuffer header = ByteBuffer.allocate(100); // big-endian file code and length
    header.putInt(9994);
    for (int i = 0; i < 5; i++) {
      header.putInt(0);
    }
    header.putInt(50 + totalWords);
    header.order(ByteOrder.LITTLE_ENDIAN).putInt(1000).putInt(5); // version, Polygon
    for (double b : bounds(shapes)) {
      header.putDouble(b);
    }
    for (int i = 0; i < 4; i++) {
      header.putDouble(0.0);
    }
    out.write(header.array());
    int i = 0;
    for (byte[] c : contents) {
      out.write(ByteBuffer.allocate(8).putInt(++i).putInt(c.length / 2).array());
      out.write(c);
    }
    Files.write(Path.of(path), out.toByteArray());
  }

  // One 32-byte dBASE field descriptor.
  static byte[] dbfField(String name, char type, int length) {
    ByteBuffer buf = little(32);
    buf.put(Arrays.copyOf(name.getBytes(StandardCharsets.US_ASCII), 11));
    buf.put((byte) type).putInt(0).put((byte) length);
    return buf.array();
  }

  static void writeCountiesDbf(String path, List<String> names, List<String> populations)
      throws IOException {
    int headerSize = 32 + 2 * 32 + 1;
    int recordSize = 1 + 12 + 8;
    var out = new ByteArrayOutputStream();
    ByteBuffer header = little(32);
    header.put(new byte[] {3, 0, 0, 0}).putInt(names.size());
    header.putShort((short) headerSize).putShort((short) recordSize);
    out.write(header.array());
    out.write(dbfField("NAME", 'C', 12));
    out.write(dbfField("POP", 'N', 8));
    out.write('\r');
    for (int i = 0; i < names.size(); i++) {
      String rec = String.format(Locale.ROOT, " %-12.12s%-8.8s", names.get(i), populations.get(i));
      out.write(rec.getBytes(StandardCharsets.US_ASCII));
    }
    Files.write(Path.of(path), out.toByteArray());
  }

  // Roughly C's %g: integral doubles print without a fractional part.
  static String fmtG(double x) {
    return x == Math.floor(x)
        ? String.valueOf((long) x)
        : String.format(Locale.ROOT, "%.6f", x);
  }

  public static void main(String[] args) throws IOException {
    String basePath = "datamunge_gis_ex_counties_java";

    // Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
    var plain = new Shape(new double[][][] {{{10, 0}, {10, 10}, {20, 10}, {20, 0}, {10, 0}}});
    var withLake =
        new Shape(
            new double[][][] {
              {{0, 0}, {0, 10}, {10, 10}, {10, 0}, {0, 0}},
              {{3, 3}, {4, 3}, {4, 4}, {3, 4}, {3, 3}}
            });

    writeCountiesShp(basePath + ".shp", List.of(withLake, plain));
    writeCountiesDbf(
        basePath + ".dbf", List.of("Lakeside", "Plainview"), List.of("48231", "19876"));

    var counties = ShapeLayer.read(basePath);

    System.out.println(
        "shapes = " + counties.size() + ", shape_type = " + counties.shapeType());
    String bounds =
        Arrays.stream(counties.bounds()).mapToObj(GisEx::fmtG).collect(Collectors.joining(", "));
    System.out.println("bounds = [" + bounds + "]\n");

    var attrs = counties.attributes();
    System.out.println("attributes\n" + attrs + "\n");

    for (long i = 0; i < counties.size(); i++) {
      System.out.println(attrs.stringAt("NAME", i) + ": " + counties.numParts(i) + " ring(s)");
    }

    String svgPath = "datamunge_gis_ex_map_java.svg";
    counties.plot().saveSvg(svgPath);
    System.out.println("\nmap saved to " + svgPath);
  }
}
