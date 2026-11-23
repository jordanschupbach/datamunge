-- CMD: just TARGET=gbm_iris_ex run-haskell
-- Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
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

  model <- newGbmClassifier iris "Species ~ Petal.Length + Petal.Width"
  gbmClassifierPrintSummary model

  confusion <- gbmClassifierConfusionMatrix model >>= dataFrameToString
  putStrLn ("\nConfusion matrix (rows = actual, cols = predicted):\n" ++ confusion)

  putStrLn "\nMisclassified rows:"
  gbmClassifierPredict model iris >>= reportMisclassified iris

  gbmClassifierPlotTrainingDeviance model >>= (`plotSave` "gbm_iris_training_deviance.svg")
  gbmClassifierPlotDecisionRegions model "Petal.Length" "Petal.Width" >>= (`plotSave` "gbm_iris_decision_regions.svg")
  putStrLn "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg"

  -- A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
  -- at the training loss, gradually sharpening the decision boundary.
  -- newGbmClassifierWith... data formula nTrees learningRate maxDepth minSamplesSplit
  --                         minSamplesLeaf subsample seed
  few <- newGbmClassifierWithNTrees iris "Species ~ Petal.Length + Petal.Width" 5
  fewAccuracy <- gbmClassifierTrainingAccuracy few
  fewDeviance <- last <$> gbmClassifierTrainingDeviance few
  putStrLn ("\n5-round ensemble:   training accuracy=" ++ fmt (100 * fewAccuracy) ++ "%  deviance=" ++ fmt fewDeviance)
  accuracy <- gbmClassifierTrainingAccuracy model
  deviance <- last <$> gbmClassifierTrainingDeviance model
  putStrLn ("100-round ensemble: training accuracy=" ++ fmt (100 * accuracy) ++ "%  deviance=" ++ fmt deviance)
  gbmClassifierPlotDecisionRegions few "Petal.Length" "Petal.Width" >>= (`plotSave` "gbm_iris_decision_regions_5rounds.svg")
  putStrLn "Saved gbm_iris_decision_regions_5rounds.svg"

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
