' CMD: just TARGET=lmm_ex run-vb
' Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
Option Strict On
Imports System
Imports System.Collections.Generic
Imports System.Linq
Imports Datamunge

Module Example
    Private ReadOnly rng As New Random(2024)

    Private Function Uniform() As Double
        Return 1 - rng.NextDouble()  ' (0, 1]: avoids log(0) in the Box-Muller transform.
    End Function

    Private Function Gauss() As Double
        Return Math.Sqrt(-2 * Math.Log(Uniform())) * Math.Cos(2 * Math.PI * Uniform())
    End Function

    Sub Main()
        Console.WriteLine("=================== Random intercept on a real dataset (penguins) ===================")
        Call New LMM(Data.Penguins(), "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)").PrintSummary()

        Console.WriteLine($"{vbLf}=================== Random intercept + slope on a simulated multi-school dataset ===================")
        ' 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
        ' and the return on an extra hour of study vary by school -- the textbook case for a
        ' random-intercept-and-slope model.
        Const nSchools As Integer = 30
        Dim schoolIntercept = Enumerable.Range(0, nSchools).Select(Function(unused) Gauss() * 6).ToArray()
        Dim schoolSlope = Enumerable.Range(0, nSchools).Select(Function(unused) Gauss() * 1.2).ToArray()
        Const trueIntercept As Double = 60.0
        Const trueSlope As Double = 3.0

        Dim school As New List(Of Double), studyHours As New List(Of Double), score As New List(Of Double)
        For s = 0 To nSchools - 1
            Dim nStudents = 15 + rng.Next(0, 21)
            For student = 0 To nStudents - 1
                Dim hours = rng.NextDouble() * 10
                school.Add(s)
                studyHours.Add(hours)
                score.Add(trueIntercept + schoolIntercept(s) + (trueSlope + schoolSlope(s)) * hours + Gauss() * 4)
            Next
        Next
        Dim df = New DataFrame()
        df.AddNumericColumn("school", school)
        df.AddNumericColumn("study_hours", studyHours)
        df.AddNumericColumn("score", score)

        Dim model = New LMM(df, "score ~ study_hours + (1 + study_hours | school)")
        model.PrintSummary()

        Console.WriteLine($"{vbLf}True generating values: intercept={trueIntercept}, slope={trueSlope}, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0")

        Console.WriteLine($"{vbLf}--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---")
        Dim labels = model.GroupLabels()
        For idx = 0 To 2
            Dim re = model.RandomEffectsForGroup(CULng(idx))
            Console.WriteLine($"school {labels(idx)}: intercept shift={re(0)}, slope shift={re(1)}")
        Next

        Console.WriteLine($"{vbLf}--- Prediction: population-level vs. school-adjusted ---")
        Dim population = New DataFrame()
        Dim school0 = New DataFrame()
        population.AddNumericColumn("study_hours", {5.0})
        school0.AddNumericColumn("study_hours", {5.0})
        school0.AddNumericColumn("school", {0.0})
        Console.WriteLine($"5 study hours, unseen school:      {model.Predict(population)(0)} (fixed effects only)")
        Console.WriteLine($"5 study hours, school 0 (known):    {model.Predict(school0)(0)} (fixed effects + school 0's BLUP)")
    End Sub
End Module
