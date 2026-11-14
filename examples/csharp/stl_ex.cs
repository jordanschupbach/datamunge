// CMD: just TARGET=stl_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// C++ std::vector and std::pair values in C#: vectors are List<T>, pairs are tuples.
using Datamunge;

const int n = 100;
var v = Enumerable.Range(0, n).Select(i => i * 1.5).ToList();
v.ForEach(Console.WriteLine);
var v2 = Enumerable.Range(0, n).Select(i => (int)Math.Floor(i * 1.5)).ToList();
v2.ForEach(Console.WriteLine);

var p = (3, 4);
Console.WriteLine($"p: ({p.Item1}, {p.Item2})");

var p2 = (10, 20);
Console.WriteLine($"p2: ({p2.Item1}, {p2.Item2})");

// A C++ std::pair<double,double> comes back as a tuple.
var p3 = Functions.MakeDpair(1.25, 2.75);
Console.WriteLine($"make_dpair(1.25, 2.75): {p3}, sum_dpair = {Functions.SumDpair(p3)}");
