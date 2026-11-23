-- CMD: just TARGET=gaussian_process_regression_iris_ex run-haskell
-- Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
module Main (main) where

import Datamunge

formula :: String
formula = "Petal.Length ~ Petal.Width"

main :: IO ()
main = do
  iris <- dataIris
  rows <- dataFrameNrows iris
  cols <- dataFrameNcols iris
  putStrLn ("iris: " ++ show rows ++ " rows x " ++ show cols ++ " cols")
  putStrLn ("formula: " ++ formula ++ "\n")

  -- Both the length scale and the noise ratio are auto-selected by maximizing the exact log
  -- marginal likelihood.
  model <- newGaussianProcessRegression iris formula
  gaussianProcessRegressionPrintSummary model

  gaussianProcessRegressionPlotFit model iris >>= (`plotSave` "gpr_iris_fit.svg")
  gaussianProcessRegressionPlotLengthScaleProfile model >>= (`plotSave` "gpr_iris_length_scale_profile.svg")
  putStrLn "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg"

  -- Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
  -- interval at every point -- including far outside the training data, where it should widen
  -- substantially as the model's uncertainty grows.
  query <- newDataFrame
  dataFrameAddNumericColumn query "Petal.Width" [0.2, 1.3, 2.5, 10.0]
  detail <- gaussianProcessRegressionPredictFrameWithIntervalKind model query "confidence"
  putStrLn "\nPredictions with 95% confidence intervals:"
  dataFrameToString detail >>= putStrLn
  putStrLn "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)"
