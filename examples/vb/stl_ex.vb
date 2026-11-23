' CMD: just TARGET=stl_ex run-vb
' C++ std::vector and std::pair values in Visual Basic: vectors are List(Of T), pairs are tuples.
Option Strict On
Imports System
Imports System.Linq
Imports Datamunge

Module Example
    Sub Main()
        Const n As Integer = 100
        Dim v = Enumerable.Range(0, n).Select(Function(i) i * 1.5).ToList()
        v.ForEach(AddressOf Console.WriteLine)
        Dim v2 = Enumerable.Range(0, n).Select(Function(i) CInt(Math.Floor(i * 1.5))).ToList()
        v2.ForEach(AddressOf Console.WriteLine)

        Dim p = (3, 4)
        Console.WriteLine($"p: ({p.Item1}, {p.Item2})")

        Dim p2 = (10, 20)
        Console.WriteLine($"p2: ({p2.Item1}, {p2.Item2})")

        ' A C++ std::pair<double,double> comes back as a tuple.
        Dim p3 = Functions.MakeDpair(1.25, 2.75)
        Console.WriteLine($"make_dpair(1.25, 2.75): {p3}, sum_dpair = {Functions.SumDpair(p3)}")
    End Sub
End Module
