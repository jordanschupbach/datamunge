// CMD: just TARGET=datamunge_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Smoke test of the basic bindings: vectors, pairs and a native Callback object.
using Datamunge;

Functions.Hello();

var v = Functions.MakeDvector(1.0, 2.0, 3.0);
Console.WriteLine($"sum_dvector:\t{Functions.SumDvector(v)}");

var p = Functions.MakeDpair(1.25, 2.75);
Console.WriteLine($"sum_dpair:\t{Functions.SumDpair(p)}");

using var cb = new Callback();
Console.WriteLine($"call_with_callback(3.0):\t{Functions.CallWithCallback(3.0, cb)}");
var v2 = Functions.MapDvectorWithCallback(Functions.MakeDvector(1.0, 2.0, 3.0), cb);
Console.WriteLine($"sum_dvector(map_dvector_with_callback(1,2,3)):\t{Functions.SumDvector(v2)}");
