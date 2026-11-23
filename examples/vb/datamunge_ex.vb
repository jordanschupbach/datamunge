' CMD: just TARGET=datamunge_ex run-vb
' Smoke test of the basic bindings: vectors, pairs and a native Callback object.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Sub Main()
        Functions.Hello()

        Dim v = Functions.MakeDvector(1.0, 2.0, 3.0)
        Console.WriteLine($"sum_dvector:{vbTab}{Functions.SumDvector(v)}")

        Dim p = Functions.MakeDpair(1.25, 2.75)
        Console.WriteLine($"sum_dpair:{vbTab}{Functions.SumDpair(p)}")

        Using cb = New Callback()
            Console.WriteLine($"call_with_callback(3.0):{vbTab}{Functions.CallWithCallback(3.0, cb)}")
            Dim v2 = Functions.MapDvectorWithCallback(Functions.MakeDvector(1.0, 2.0, 3.0), cb)
            Console.WriteLine($"sum_dvector(map_dvector_with_callback(1,2,3)):{vbTab}{Functions.SumDvector(v2)}")
        End Using
    End Sub
End Module
