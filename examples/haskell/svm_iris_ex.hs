-- CMD: just TARGET=svm_iris_ex run-haskell
-- Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
module Main (main) where

import Datamunge
import Numeric (floatToDigits)

formula :: String
formula = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"

main :: IO ()
main = do
  iris <- dataIris
  rows <- dataFrameNrows iris
  cols <- dataFrameNcols iris
  putStrLn ("iris: " ++ show rows ++ " rows x " ++ show cols ++ " cols\n")

  putStrLn "=== RBF kernel (default) ==="
  rbfModel <- newSvm iris formula
  svmPrintSummary rbfModel
  confusion <- svmConfusionMatrix rbfModel >>= dataFrameToString
  putStrLn ("\nConfusion matrix (rows = actual, cols = predicted):\n" ++ confusion)

  putStrLn "\n=== Linear kernel, for comparison ==="
  linearModel <- newSvmWithKernel iris formula "linear"
  accuracy <- svmTrainingAccuracy linearModel
  putStrLn ("Training accuracy: " ++ fmt (accuracy * 100) ++ "%")
  supportVectors <- svmNumSupportVectors linearModel
  putStrLn ("Support vectors: " ++ show supportVectors)

  newdata <- newDataFrame
  dataFrameAddNumericColumn newdata "Sepal.Length" [5.1, 6.0, 6.5, 6.2]
  dataFrameAddNumericColumn newdata "Sepal.Width" [3.5, 2.7, 3.0, 2.8]
  dataFrameAddNumericColumn newdata "Petal.Length" [1.4, 4.5, 5.5, 4.8]
  dataFrameAddNumericColumn newdata "Petal.Width" [0.2, 1.5, 2.0, 1.8]
  predictions <- svmPredictFrame rbfModel newdata >>= dataFrameToString
  putStrLn ("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n" ++ predictions)

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
