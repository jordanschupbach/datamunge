-- CMD: just TARGET=elastic_net_iris_ex run-haskell
-- Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
module Main (main) where

import Control.Monad (forM_)
import Datamunge

formula :: String
formula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width"

main :: IO ()
main = do
  iris <- dataIris
  rows <- dataFrameNrows iris
  cols <- dataFrameNcols iris
  putStrLn ("iris: " ++ show rows ++ " rows x " ++ show cols ++ " cols")
  putStrLn ("formula: " ++ formula ++ "\n")

  putStrLn "=================== Ridge ==================="
  ridge <- newRidge iris formula
  ridgePrintSummary ridge

  putStrLn "\n=================== Lasso ==================="
  lasso <- newLasso iris formula
  lassoPrintSummary lasso

  putStrLn "\n================= Elastic Net ================="
  -- newElasticNetWith... data formula alpha=0.5 lambda nLambda cvFolds standardize seed
  elastic <- newElasticNetWithAlpha iris formula 0.5
  elasticNetPrintSummary elastic

  putStrLn ("\nSaved figures showing how each model's coefficients respond to the regularization strength, "
              ++ "and the cross-validation curve used to pick it:")

  figures <- sequence
    [ (,,,) <$> ridgePlotCoefficientPath ridge <*> ridgePlotCvCurve ridge <*> pure "ridge" <*> pure "ridge:      "
    , (,,,) <$> lassoPlotCoefficientPath lasso <*> lassoPlotCvCurve lasso <*> pure "lasso" <*> pure "lasso:      "
    , (,,,) <$> elasticNetPlotCoefficientPath elastic <*> elasticNetPlotCvCurve elastic <*> pure "elasticnet" <*> pure "elastic net:"
    ]
  forM_ figures $ \(path, cv, name, label) -> do
    plotSave path ("elastic_net_" ++ name ++ "_path.svg")
    plotSave cv ("elastic_net_" ++ name ++ "_cv.svg")
    putStrLn ("  " ++ label ++ " elastic_net_" ++ name ++ "_path.svg, elastic_net_" ++ name ++ "_cv.svg")
