-- CMD: just TARGET=xgboost_iris_ex run-haskell
-- XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
module Main (main) where

import Control.Monad (forM)
import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  iris <- dataIris
  rows <- dataFrameNrows iris
  cols <- dataFrameNcols iris
  putStrLn ("iris: " ++ show rows ++ " rows x " ++ show cols ++ " cols\n")

  model <- newXgBoostClassifier iris "Species ~ Petal.Length + Petal.Width"
  xgBoostClassifierPrintSummary model

  confusion <- xgBoostClassifierConfusionMatrix model >>= dataFrameToString
  putStrLn ("\nConfusion matrix (rows = actual, cols = predicted):\n" ++ confusion)

  putStrLn "\nMisclassified rows:"
  xgBoostClassifierPredict model iris >>= reportMisclassified iris

  xgBoostClassifierPlotTrainingDeviance model >>= (`plotSave` "xgboost_iris_training_deviance.svg")
  xgBoostClassifierPlotDecisionRegions model "Petal.Length" "Petal.Width" >>= (`plotSave` "xgboost_iris_decision_regions.svg")
  putStrLn "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg"

  -- XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
  -- grows the same deep trees but keeps every leaf weight small, producing a much smoother
  -- decision boundary than the lightly-regularized default.
  -- newXgBoostClassifierWith... data formula nTrees learningRate maxDepth lambda alpha gamma
  --                             minChildWeight minSamplesLeaf subsample colsampleBytree seed
  heavy <- newXgBoostClassifierWithLambda iris "Species ~ Petal.Length + Petal.Width" 100 0.3 6 50.0
  accuracy <- xgBoostClassifierTrainingAccuracy model
  deviance <- last <$> xgBoostClassifierTrainingDeviance model
  putStrLn ("\nlambda=1 (default):   training accuracy=" ++ fmt (100 * accuracy) ++ "%  deviance=" ++ fmt deviance)
  heavyAccuracy <- xgBoostClassifierTrainingAccuracy heavy
  heavyDeviance <- last <$> xgBoostClassifierTrainingDeviance heavy
  putStrLn ("lambda=50 (heavy L2): training accuracy=" ++ fmt (100 * heavyAccuracy) ++ "%  deviance=" ++ fmt heavyDeviance)
  xgBoostClassifierPlotDecisionRegions heavy "Petal.Length" "Petal.Width" >>= (`plotSave` "xgboost_iris_decision_regions_heavy_lambda.svg")
  putStrLn "Saved xgboost_iris_decision_regions_heavy_lambda.svg"

-- | Prints the rows whose prediction differs from the actual species, and the error rate.
reportMisclassified :: DataFrame -> [String] -> IO ()
reportMisclassified iris predictions = do
  n <- dataFrameNrows iris
  misclassified <- fmap sum . forM (zip [0 .. n - 1] predictions) $ \(i, predicted) -> do
    actual <- dataFrameStringAt iris "Species" i
    if predicted == actual
      then pure (0 :: Int)
      else do
        len <- dataFrameNumericAt iris "Petal.Length" i
        width <- dataFrameNumericAt iris "Petal.Width" i
        putStrLn ("  row " ++ show i ++ ": Petal.Length=" ++ fmt len ++ " Petal.Width=" ++ fmt width
                    ++ "  actual=" ++ actual ++ "  predicted=" ++ predicted)
        pure 1
  putStrLn (show misclassified ++ " of " ++ show n ++ " misclassified ("
              ++ fmt (100.0 * fromIntegral misclassified / fromIntegral n) ++ "%)")

-- | A Double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
-- when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
fmt :: Double -> String
fmt x
  | isNaN x = "NaN"
  | isInfinite x = if x < 0 then "-∞" else "∞"
  | x < 0 || isNegativeZero x = '-' : fmt (negate x)
  | x == 0 = "0"
  | exp10 < -4 || exp10 >= 15 =
      let (d, rest) = splitAt 1 digits
          magnitude = show (abs exp10)
       in d ++ (if null rest then "" else '.' : rest) ++ "E" ++ (if exp10 < 0 then "-" else "+")
            ++ replicate (2 - length magnitude) '0' ++ magnitude
  | e <= 0 = "0." ++ replicate (negate e) '0' ++ digits
  | e >= length digits = digits ++ replicate (e - length digits) '0'
  | otherwise = let (whole, frac) = splitAt e digits in whole ++ "." ++ frac
  where
    (ds, e) = floatToDigits 10 x
    digits = concatMap show ds
    exp10 = e - 1
