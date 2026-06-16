package js.datamunge.jdatamunge;

public class App {

  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void main(String[] args) {

    System.out.println("Hello, World!");
  }
}
