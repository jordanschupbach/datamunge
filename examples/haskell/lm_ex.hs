-- CMD: just TARGET=lm_ex run-haskell
-- Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
module Main (main) where

import Datamunge

main :: IO ()
main = do
  cars <- newDataFrame
  dataFrameAddNumericColumn cars "hp" [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0]
  dataFrameAddNumericColumn cars "wt" [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44]
  dataFrameAddStringColumn cars "transmission" ["manual", "manual", "manual", "automatic", "automatic",
                                               "automatic", "automatic", "automatic", "automatic", "automatic"]
  dataFrameAddNumericColumn cars "mpg" [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2]

  putStrLn "Fitting: mpg ~ hp + wt + transmission\n"
  model <- newLm cars "mpg ~ hp + wt + transmission"
  lmPrintSummary model

  anova <- lmAnova model >>= dataFrameToString
  putStrLn ("\nSequential ANOVA:\n" ++ anova)

  newcars <- newDataFrame
  dataFrameAddNumericColumn newcars "hp" [150.0, 90.0]
  dataFrameAddNumericColumn newcars "wt" [3.0, 2.5]
  dataFrameAddStringColumn newcars "transmission" ["manual", "automatic"]
  predictions <- lmPredictFrameWithIntervalKind model newcars "confidence" >>= dataFrameToString
  putStrLn ("\nPredictions with 95% confidence intervals:\n" ++ predictions)

  lmSaveDiagnosticPlots model "lm_ex_diagnostics"
  putStrLn "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg"
