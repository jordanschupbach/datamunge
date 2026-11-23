' CMD: just TARGET=naive_bayes_penguins_ex run-vb
' Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
Option Strict On
Imports System
Imports System.Linq
Imports Datamunge

Module Example
    Sub Main()
        ' Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
        ' when fitting, matching every other formula-based model in this library.
        Dim penguins = Data.Penguins()
        Console.WriteLine($"penguins: {penguins.Nrows()} rows x {penguins.Ncols()} cols{vbLf}")

        ' A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
        ' predictors in one formula -- each modeled independently given the class, per the naive
        ' Bayes assumption.
        Dim model = New NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")
        model.PrintSummary()

        Console.WriteLine($"{vbLf}Confusion matrix (rows = actual, cols = predicted):{vbLf}{model.ConfusionMatrix()}")

        Console.WriteLine($"{vbLf}Misclassified rows:")
        Dim predictions = model.Predict(penguins)
        Dim misclassified = 0
        Dim columns = {"species", "bill_length_mm", "bill_depth_mm", "island", "sex"}
        For i As ULong = 0UL To penguins.Nrows() - 1UL
            Dim row = i
            If columns.Any(Function(c) penguins.IsNull(c, row)) Then Continue For
            Dim actual = penguins.StringAt("species", i), predicted = predictions(CInt(i))
            If predicted = actual Then Continue For
            misclassified += 1
            Console.WriteLine($"  row {i}: bill_length={penguins.NumericAt("bill_length_mm", i)} bill_depth={penguins.NumericAt("bill_depth_mm", i)} " &
                              $"island={penguins.StringAt("island", i)} sex={penguins.StringAt("sex", i)}  actual={actual}  predicted={predicted}")
        Next
        Console.WriteLine($"{misclassified} misclassified (of {penguins.Nrows()} rows, some incomplete)")

        ' PlotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
        ' two-predictor model (bill measurements alone) just for visualization.
        Dim billOnly = New NaiveBayesClassifier(penguins, "species ~ bill_length_mm + bill_depth_mm")
        Console.WriteLine($"{vbLf}bill-measurements-only model training accuracy: {100 * billOnly.TrainingAccuracy()}%")
        billOnly.PlotClassification(penguins, "bill_length_mm", "bill_depth_mm").Save("naive_bayes_penguins_classification.svg")
        billOnly.PlotDecisionRegions("bill_length_mm", "bill_depth_mm").Save("naive_bayes_penguins_decision_regions.svg")
        Console.WriteLine("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg")
    End Sub
End Module
