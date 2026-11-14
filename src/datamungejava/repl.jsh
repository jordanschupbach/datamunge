// jshell start-up for `just repl-java`: the datamunge package (build/java/prefix/share/java/datamunge.jar)
// is on the class path and DATAMUNGE_JAVA_LIBRARY names libdatamunge_java. Try
//   var df = new DataFrame(); df.addNumericColumn("x", Datamunge.seql(0, 1, 5)); System.out.println(df);
import datamunge.*;
import datamunge.data.Data;
System.out.println("datamunge " + "loaded: import datamunge.*; (try Data.iris().nrows(), Datamunge.seql(0, 1, 5))");
