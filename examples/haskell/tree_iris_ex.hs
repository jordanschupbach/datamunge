-- CMD: just TARGET=tree_iris_ex run-haskell
-- Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
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

  model <- newDecisionTreeClassifier iris "Species ~ Petal.Length + Petal.Width"
  decisionTreeClassifierPrintSummary model

  putStrLn "\nConfusion matrix (rows = actual, cols = predicted):"
  decisionTreeClassifierConfusionMatrix model >>= dataFrameToString >>= putStrLn

  putStrLn "\nMisclassified rows:"
  decisionTreeClassifierPredict model iris >>= reportMisclassified iris

  decisionTreeClassifierPlotClassification model iris "Petal.Length" "Petal.Width" >>= (`plotSave` "tree_iris_classification.svg")
  decisionTreeClassifierPlotDecisionRegions model "Petal.Length" "Petal.Width" >>= (`plotSave` "tree_iris_decision_regions.svg")
  putStrLn "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg"

  -- A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
  shallow <- newDecisionTreeClassifierWithMaxDepth iris "Species ~ Petal.Length + Petal.Width" 2
  accuracy <- decisionTreeClassifierTrainingAccuracy shallow
  leaves <- decisionTreeClassifierLeafCount shallow
  putStrLn ("\nDepth-2 tree training accuracy: " ++ fmt (accuracy * 100) ++ "% (" ++ show leaves ++ " leaves)")
  decisionTreeClassifierPlotDecisionRegions shallow "Petal.Length" "Petal.Width" >>= (`plotSave` "tree_iris_decision_regions_depth2.svg")
  putStrLn "Saved tree_iris_decision_regions_depth2.svg"

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
