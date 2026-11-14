#:project ../../src/datamungecsharp/Datamunge.csproj
// Smoke tests of the Datamunge C# binding (just test-csharp).
using System.Xml.Linq;
using Datamunge;

int checks = 0;
void Check(bool ok, string message)
{
    checks++;
    if (!ok) throw new Exception("check failed: " + message);
}
Exception? Caught(Action action)
{
    try { action(); return null; }
    catch (Exception error) { return error; }
}

// Smoke test: call into the native library.
Functions.Hello();

// std::vector<double> and std::pair<double,double> round trips.
var v = Functions.MakeDvector(1.0, 2.0, 3.0);
Check(v is List<double>, "std::vector<double> is a List<double>");
Check(v.SequenceEqual(new[] { 1.0, 2.0, 3.0 }), "vector contents");
Check(Functions.SumDvector(v) == 6.0, "SumDvector");
Check(Functions.SumDvector(new[] { 0.5, 1.5 }) == 2.0, "arrays convert to std::vector");
var p = Functions.MakeDpair(1.25, 2.75);
Check(p == (1.25, 2.75), "std::pair is a tuple");
Check(Functions.SumDpair(p) == 4.0, "SumDpair");

// The C++ Callback class, and a C# subclass overriding its virtual Call() (a director).
using (var cb = new Callback())
{
    Check(Functions.CallWithCallback(3.0, cb) == 3.0, "native Callback");
    Check(Functions.SumDvector(Functions.MapDvectorWithCallback(Functions.MakeDvector(1.0, 2.0, 3.0), cb)) == 6.0, "native map");
}
var doubler = new Doubler();
Check(Functions.CallWithCallback(3.0, doubler) == 6.0, "the C# override runs");
Check(Functions.SumDvector(Functions.MapDvectorWithCallback(new[] { 1.0, 2.0, 3.0 }, doubler)) == 12.0, "override from C++ loops");

// A C# exception inside an override reaches the caller as the original exception.
Check(Caught(() => Functions.CallWithCallback(1.0, new Failing())) is FromCSharp, "the C# exception propagates");

// C++ exceptions become NativeException.
Check(Caught(() => Functions.Runif(1.0, 0.0, 3)) is NativeException, "C++ exceptions throw NativeException");

// Classes, methods and documentation carrying the C++ documentation.
using (var frame = new DataFrame())
{
    frame.AddNumericColumn("x", new[] { 1.0, 2.0, 3.0 });
    Check(frame.Nrows() == 3, "DataFrame");
}
var xml = Path.ChangeExtension(typeof(DataFrame).Assembly.Location, ".xml");
Check(File.Exists(xml), "the XML documentation is built");
var seql = XDocument.Load(xml).Descendants("member").First(m => ((string?)m.Attribute("name") ?? "").StartsWith("M:Datamunge.Functions.Seql("));
Check(seql.Element("summary")!.Value.Contains("evenly spaced"), "XML documentation");
Check(seql.Elements("param").Any(e => (string?)e.Attribute("name") == "n"), "parameter documentation");

Console.WriteLine($"test_datamunge.cs: all {checks} checks passed");

class Doubler : Callback
{
    public override double Call(double x) => 2 * x;
}
class FromCSharp : Exception { }
class Failing : Callback
{
    public override double Call(double x) => throw new FromCSharp();
}
