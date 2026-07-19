package js.datamunge.jdatamunge.examples;

import java.lang.reflect.Method;

public final class ExampleRunner {
  private ExampleRunner() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 1) {
      System.err.println("usage: ExampleRunner <snake_case_example_name>");
      System.exit(1);
    }
    String className = toPascalCase(args[0]);
    Class<?> clazz = Class.forName("js.datamunge.jdatamunge.examples." + className);
    Method run = clazz.getMethod("run");
    run.invoke(null);
  }

  private static String toPascalCase(String snakeCase) {
    StringBuilder sb = new StringBuilder();
    for (String part : snakeCase.split("_")) {
      if (!part.isEmpty()) {
        sb.append(Character.toUpperCase(part.charAt(0))).append(part.substring(1));
      }
    }
    return sb.toString();
  }
}
