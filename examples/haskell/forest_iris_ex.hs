-- CMD: just TARGET=forest_iris_ex run-haskell
-- Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
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

  model <- newRandomForestClassifier iris "Species ~ Petal.Length + Petal.Width"
  randomForestClassifierPrintSummary model

  putStrLn "\nConfusion matrix (rows = actual, cols = predicted):"
  randomForestClassifierConfusionMatrix model >>= dataFrameToString >>= putStrLn

  putStrLn "\nMisclassified rows:"
  randomForestClassifierPredict model iris >>= reportMisclassified iris

  randomForestClassifierPlotClassification model iris "Petal.Length" "Petal.Width" >>= (`plotSave` "forest_iris_classification.svg")
  randomForestClassifierPlotDecisionRegions model "Petal.Length" "Petal.Width" >>= (`plotSave` "forest_iris_decision_regions.svg")
  putStrLn "\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg"

  -- A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
  -- are added -- the defining random forest effect a single decision tree cannot demonstrate.
  -- newRandomForestClassifierWith... data formula nTrees maxDepth minSamplesSplit minSamplesLeaf
  --                                  maxFeatures criterion bootstrap sampleFraction seed
  smallForest <- newRandomForestClassifierWithNTrees iris "Species ~ Petal.Length + Petal.Width" 5
  smallTraining <- randomForestClassifierTrainingAccuracy smallForest
  smallOob <- randomForestClassifierOobAccuracy smallForest
  putStrLn ("\n5-tree forest:   training accuracy=" ++ fmt (smallTraining * 100) ++ "%  OOB accuracy=" ++ fmt (smallOob * 100) ++ "%")
  training <- randomForestClassifierTrainingAccuracy model
  oob <- randomForestClassifierOobAccuracy model
  putStrLn ("100-tree forest: training accuracy=" ++ fmt (training * 100) ++ "%  OOB accuracy=" ++ fmt (oob * 100) ++ "%")
  randomForestClassifierPlotDecisionRegions smallForest "Petal.Length" "Petal.Width" >>= (`plotSave` "forest_iris_decision_regions_5trees.svg")
  putStrLn "Saved forest_iris_decision_regions_5trees.svg"

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
