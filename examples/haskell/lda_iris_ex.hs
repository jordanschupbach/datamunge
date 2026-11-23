-- CMD: just TARGET=lda_iris_ex run-haskell
-- Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
module Main (main) where

import Datamunge

main :: IO ()
main = do
  iris <- dataIris
  rows <- dataFrameNrows iris
  cols <- dataFrameNcols iris
  putStrLn ("iris: " ++ show rows ++ " rows x " ++ show cols ++ " cols\n")

  model <- newLda iris "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"
  ldaPrintSummary model

  confusion <- ldaConfusionMatrix model >>= dataFrameToString
  putStrLn ("\nConfusion matrix (rows = actual, cols = predicted):\n" ++ confusion)

  newdata <- newDataFrame
  dataFrameAddNumericColumn newdata "Sepal.Length" [5.1, 6.0, 6.5, 6.2]
  dataFrameAddNumericColumn newdata "Sepal.Width" [3.5, 2.7, 3.0, 2.8]
  dataFrameAddNumericColumn newdata "Petal.Length" [1.4, 4.5, 5.5, 4.8]
  dataFrameAddNumericColumn newdata "Petal.Width" [0.2, 1.5, 2.0, 1.8]
  predictions <- ldaPredictFrame model newdata >>= dataFrameToString
  putStrLn ("\nPredictions for new flowers:\n" ++ predictions)

  ldaSaveDiscriminantPlot model "lda_iris_discriminants.svg"
  putStrLn "\nSaved discriminant plot as lda_iris_discriminants.svg"
