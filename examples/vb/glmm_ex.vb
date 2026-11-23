' CMD: just TARGET=glmm_ex run-vb
' Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
Option Strict On
Imports System
Imports System.Collections.Generic
Imports System.Linq
Imports Datamunge

Module Example
    Private ReadOnly rng As New Random(4242)

    Private Function Uniform() As Double
        Return rng.NextDouble()
    End Function

    Private Function Gauss() As Double
        Dim u1 = 1 - Uniform(), u2 = Uniform()  ' 1 - u keeps log() away from 0
        Return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)
    End Function

    Sub Main()
        Console.WriteLine("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================")
        ' Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
        ' confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
        ' is not tied to location, so this is a well-behaved fit.
        Dim penguins = Data.Penguins()
        Dim isMale As New List(Of Double), bodyMass As New List(Of Double)
        Dim island As New List(Of String)
        For i As ULong = 0UL To penguins.Nrows() - 1UL
            Dim row = i
            If {"sex", "body_mass_g", "island"}.Any(Function(c) penguins.IsNull(c, row)) Then Continue For
            isMale.Add(If(penguins.StringAt("sex", i) = "male", 1.0, 0.0))
            bodyMass.Add(penguins.NumericAt("body_mass_g", i))
            island.Add(penguins.StringAt("island", i))
        Next
        Dim sexDf = New DataFrame()
        sexDf.AddNumericColumn("is_male", isMale)
        sexDf.AddNumericColumn("body_mass_g", bodyMass)
        sexDf.AddStringColumn("island", island)
        Call New GLMM(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial").PrintSummary()

        Console.WriteLine($"{vbLf}=================== Poisson mixed model on simulated multi-site count data ===================")
        ' 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
        ' the baseline traffic and the promo's effectiveness vary by store.
        Const nStores As Integer = 25
        Dim storeEffect = Enumerable.Range(0, nStores).Select(Function(unused) Gauss() * 0.4).ToArray()
        Const trueIntercept As Double = 2.0
        Const trueSlope As Double = 0.3

        Dim store As New List(Of Double), promo As New List(Of Double), visits As New List(Of Double)
        For s = 0 To nStores - 1
            Dim nDays = 15 + rng.Next(0, 11)
            For dayIndex = 0 To nDays - 1
                Dim promoIntensity = Uniform() * 3
                Dim lam = Math.Exp(trueIntercept + storeEffect(s) + trueSlope * promoIntensity)
                ' Knuth's Poisson sampler.
                Dim threshold = Math.Exp(-lam), k = 0, p = 1.0
                Do
                    k += 1
                    p *= Uniform()
                Loop While p > threshold
                store.Add(s)
                promo.Add(promoIntensity)
                visits.Add(k - 1)
            Next
        Next
        Dim df = New DataFrame()
        df.AddNumericColumn("store", store)
        df.AddNumericColumn("promo", promo)
        df.AddNumericColumn("visits", visits)
        Dim storeModel = New GLMM(df, "visits ~ promo + (1 | store)", "poisson")

        storeModel.PrintSummary()

        Console.WriteLine($"{vbLf}True generating values: intercept={trueIntercept}, slope={trueSlope}, random-intercept SD (log scale)=0.4")

        Console.WriteLine($"{vbLf}--- BLUPs for a few stores ---")
        Dim labels = storeModel.GroupLabels()
        For idx = 0 To 2
            Console.WriteLine($"store {labels(idx)}: intercept shift={storeModel.RandomEffectsForGroup(CULng(idx))(0)}")
        Next

        Console.WriteLine($"{vbLf}--- Prediction: population-level vs. store-adjusted ---")
        Dim population = New DataFrame()
        Dim store0 = New DataFrame()
        population.AddNumericColumn("promo", {1.5})
        store0.AddNumericColumn("promo", {1.5})
        store0.AddNumericColumn("store", {0.0})
        Console.WriteLine($"promo=1.5, unseen store:   {storeModel.Predict(population)(0)} expected visits (fixed effects only)")
        Console.WriteLine($"promo=1.5, store 0 (known): {storeModel.Predict(store0)(0)} expected visits (fixed effects + store 0's BLUP)")
    End Sub
End Module
