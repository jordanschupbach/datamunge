package js.octra.joctra;

public class App {

  static {
    System.loadLibrary("octra_jni");
  }

  public static void main(String[] args) {

    System.out.println("Hello, World!");
  }
}
