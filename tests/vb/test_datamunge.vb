' Smoke tests of the Datamunge Visual Basic binding (just test-vb).
Option Strict On
Imports System
Imports System.Collections.Generic
Imports System.IO
Imports System.Linq
Imports System.Xml.Linq
Imports Datamunge

Module TestDatamunge
    Private checks As Integer = 0

    Private Sub Check(ok As Boolean, message As String)
        checks += 1
        If Not ok Then Throw New Exception("check failed: " & message)
    End Sub

    Private Function Caught(action As Action) As Exception
        Try
            action()
            Return Nothing
        Catch e As Exception
            Return e
        End Try
    End Function

    Class Doubler
        Inherits Callback
        Public Overrides Function [Call](x As Double) As Double
            Return 2 * x
        End Function
    End Class

    Class FromVisualBasic
        Inherits Exception
    End Class

    Class Failing
        Inherits Callback
        Public Overrides Function [Call](x As Double) As Double
            Throw New FromVisualBasic()
        End Function
    End Class

    Sub Main()
        ' Smoke test: call into the native library.
        Functions.Hello()

        ' std::vector<double> and std::pair<double,double> round trips.
        Dim v = Functions.MakeDvector(1.0, 2.0, 3.0)
        Check(TypeOf v Is List(Of Double), "std::vector<double> is a List(Of Double)")
        Check(v.SequenceEqual({1.0, 2.0, 3.0}), "vector contents")
        Check(Functions.SumDvector(v) = 6.0, "SumDvector")
        Check(Functions.SumDvector({0.5, 1.5}) = 2.0, "arrays convert to std::vector")
        Dim p = Functions.MakeDpair(1.25, 2.75)
        Check(p.Item1 = 1.25 AndAlso p.Item2 = 2.75, "std::pair is a tuple")
        Check(Functions.SumDpair(p) = 4.0, "SumDpair")

        ' The C++ Callback class, and a Visual Basic subclass overriding its virtual Call() (a director).
        Using cb As New Callback()
            Check(Functions.CallWithCallback(3.0, cb) = 3.0, "native Callback")
            Check(Functions.SumDvector(Functions.MapDvectorWithCallback(Functions.MakeDvector(1.0, 2.0, 3.0), cb)) = 6.0, "native map")
        End Using
        Dim twice As New Doubler()
        Check(Functions.CallWithCallback(3.0, twice) = 6.0, "the Visual Basic override runs")
        Check(Functions.SumDvector(Functions.MapDvectorWithCallback({1.0, 2.0, 3.0}, twice)) = 12.0, "override from C++ loops")

        ' A Visual Basic exception inside an override reaches the caller as the original exception.
        Check(TypeOf Caught(Sub() Functions.CallWithCallback(1.0, New Failing())) Is FromVisualBasic, "the Visual Basic exception propagates")

        ' C++ exceptions become NativeException.
        Check(TypeOf Caught(Sub() Functions.Runif(1.0, 0.0, 3UL)) Is NativeException, "C++ exceptions throw NativeException")

        ' Classes, methods and the XML documentation carrying the C++ documentation.
        Using frame As New DataFrame()
            frame.AddNumericColumn("x", {1.0, 2.0, 3.0})
            Check(frame.Nrows() = 3UL, "DataFrame")
        End Using
        Dim xml = Path.ChangeExtension(GetType(DataFrame).Assembly.Location, ".xml")
        Check(File.Exists(xml), "the XML documentation is built")
        Dim seql = XDocument.Load(xml).Descendants("member").First(Function(m) If(CStr(m.Attribute("name")), "").StartsWith("M:Datamunge.Functions.Seql("))
        Check(seql.Element("summary").Value.Contains("evenly spaced"), "XML documentation")
        Check(seql.Elements("param").Any(Function(e) CStr(e.Attribute("name")) = "n"), "parameter documentation")

        Console.WriteLine($"test_datamunge.vb: all {checks} checks passed")
    End Sub
End Module
