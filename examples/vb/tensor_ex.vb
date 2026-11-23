' CMD: just TARGET=tensor_ex run-vb
' Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
Option Strict On
Imports System
Imports System.Collections.Generic
Imports Datamunge

Module Example
    Private Function ShapeString(tensor As Tensor) As String
        Return $"[{String.Join(", ", tensor.Shape())}]"
    End Function

    Sub Main()
        Console.WriteLine("=================== Construction ===================")
        Console.WriteLine($"zeros([2,3]): {Tensor.Zeros({2UL, 3UL})}")
        Console.WriteLine($"eye(3): {Tensor.Eye(3UL)}")

        Dim r = Tensor.Arange(0.0, 12.0, 1.0).Reshape({3UL, 4UL})
        Console.WriteLine($"arange(0,12).reshape([3,4]): {r}")

        Console.WriteLine($"{vbLf}=================== Shape ops ===================")
        Console.WriteLine($"transpose -> shape {ShapeString(r.Transpose())}")
        Console.WriteLine($"slice(axis=1, start=1, stop=3): {r.Slice(1UL, 1UL, 3UL)}")

        Console.WriteLine($"{vbLf}=================== Broadcasting arithmetic ===================")
        Dim col = Tensor.FromValues({3UL, 1UL}, {1.0, 2.0, 3.0})
        Dim row = Tensor.FromValues({1UL, 4UL}, {10.0, 20.0, 30.0, 40.0})
        Console.WriteLine($"(3,1) + (1,4) -> {col.Add(row)}")

        Console.WriteLine($"{vbLf}=================== Reductions ===================")
        Console.WriteLine($"r.sum() = {r.Sum()}, r.mean() = {r.Mean()}")
        Console.WriteLine($"column means (axis=0): {r.MeanAxis(0UL)}")

        Console.WriteLine($"{vbLf}=================== Linear algebra ===================")
        Dim a = Tensor.FromValues({2UL, 3UL}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
        Dim b = Tensor.FromValues({3UL, 2UL}, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0})
        Console.WriteLine($"matmul(2x3, 3x2) -> {a.Matmul(b)}")
        Dim v1 = Tensor.FromValues({3UL}, {1.0, 2.0, 3.0})
        Dim v2 = Tensor.FromValues({3UL}, {4.0, 5.0, 6.0})
        Console.WriteLine($"dot([1,2,3], [4,5,6]) = {v1.Dot(v2)}")
        Console.WriteLine($"outer(v1, v2) -> {v1.Outer(v2)}")

        Console.WriteLine($"{vbLf}=================== Comparisons & masks ===================")
        Dim mask = r.GreaterEqual(Tensor.Full({3UL, 4UL}, 6.0))
        Console.WriteLine($"r >= 6 -> {mask}")
        Console.WriteLine($"count(r >= 6) = {mask.Sum()}")

        Console.WriteLine($"{vbLf}=================== A real dataset as a Tensor ===================")
        Dim iris = Data.Iris()
        Dim n = iris.Nrows()
        Dim columns = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}
        Dim flat As New List(Of Double)
        For i As ULong = 0UL To n - 1UL
            For Each column In columns
                flat.Add(iris.NumericAt(column, i))
            Next
        Next
        Dim x = Tensor.FromValues({n, 4UL}, flat)
        Console.WriteLine($"iris feature tensor shape: {ShapeString(x)}")

        Dim featureMeans = x.MeanAxis(0UL)
        Dim centered = x.Subtract(featureMeans.Reshape({1UL, 4UL}))
        Dim scatter = centered.Transpose().Matmul(centered)
        Console.WriteLine($"feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): {featureMeans}")
        Console.WriteLine($"(X-mean)^T (X-mean) [4x4 scatter matrix]: {scatter.ToString(16UL)}")

        Dim species As New List(Of String)
        For i As ULong = 0UL To n - 1UL
            species.Add(iris.StringAt("Species", i))
        Next
        Dim speciesTensor = Tensor.FromStringValues({n}, species)
        Dim setosaMask = speciesTensor.Equal(Tensor.FromStringValues({1UL}, {"setosa"}))
        Console.WriteLine($"setosa count = {setosaMask.Sum()} (of {n} rows)")
    End Sub
End Module
