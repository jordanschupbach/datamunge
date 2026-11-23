-- CMD: just TARGET=knn_iris_ex run-haskell
-- k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
module Main (main) where

import Control.Monad (forM, forM_)
import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  iris <- dataIris
  rows <- dataFrameNrows iris
  cols <- dataFrameNcols iris
  putStrLn ("iris: " ++ show rows ++ " rows x " ++ show cols ++ " cols\n")

  model <- newKnnClassifier iris "Species ~ Petal.Length + Petal.Width"
  knnClassifierPrintSummary model

  putStrLn "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):"
  knnClassifierConfusionMatrix model >>= dataFrameToString >>= putStrLn

  putStrLn "\nLeave-one-out misclassified rows:"
  knnClassifierPredict model iris >>= reportMisclassified iris

  knnClassifierPlotDecisionRegions model "Petal.Length" "Petal.Width" >>= (`plotSave` "knn_iris_decision_regions_k5.svg")
  putStrLn "\nSaved knn_iris_decision_regions_k5.svg"

  -- k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
  -- every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
  -- underfit boundary). newKnnClassifierWith... data formula k metric weighted standardize
  forM_ [1, 25] $ \k -> do
    knn <- newKnnClassifierWithK iris "Species ~ Petal.Length + Petal.Width" k
    let file = "knn_iris_decision_regions_k" ++ show k ++ ".svg"
    accuracy <- knnClassifierTrainingAccuracy knn
    putStrLn ("\nk=" ++ padRight 2 (show k) ++ " leave-one-out accuracy: " ++ fmt (accuracy * 100) ++ "%")
    knnClassifierPlotDecisionRegions knn "Petal.Length" "Petal.Width" >>= (`plotSave` file)
    putStrLn ("Saved " ++ file)

-- | C#'s {value,-n}: left-aligned in a field of n characters.
padRight :: Int -> String -> String
padRight n s = s ++ replicate (n - length s) ' '

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
