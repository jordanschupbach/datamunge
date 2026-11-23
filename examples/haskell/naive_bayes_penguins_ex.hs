-- CMD: just TARGET=naive_bayes_penguins_ex run-haskell
-- Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
module Main (main) where

import Control.Monad (forM)
import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  -- Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
  -- when fitting, matching every other formula-based model in this library.
  penguins <- dataPenguins
  rows <- dataFrameNrows penguins
  cols <- dataFrameNcols penguins
  putStrLn ("penguins: " ++ show rows ++ " rows x " ++ show cols ++ " cols\n")

  -- A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
  -- predictors in one formula -- each modeled independently given the class, per the naive
  -- Bayes assumption.
  model <- newNaiveBayesClassifier penguins "species ~ bill_length_mm + bill_depth_mm + island + sex"
  naiveBayesClassifierPrintSummary model

  confusion <- naiveBayesClassifierConfusionMatrix model >>= dataFrameToString
  putStrLn ("\nConfusion matrix (rows = actual, cols = predicted):\n" ++ confusion)

  putStrLn "\nMisclassified rows:"
  predictions <- naiveBayesClassifierPredict model penguins
  let columns = ["species", "bill_length_mm", "bill_depth_mm", "island", "sex"]
  misclassified <- fmap sum . forM (zip [0 .. rows - 1] predictions) $ \(i, predicted) -> do
    incomplete <- or <$> mapM (\c -> dataFrameIsNull penguins c i) columns
    actual <- dataFrameStringAt penguins "species" i
    if incomplete || predicted == actual
      then pure (0 :: Int)
      else do
        billLength <- dataFrameNumericAt penguins "bill_length_mm" i
        billDepth <- dataFrameNumericAt penguins "bill_depth_mm" i
        island <- dataFrameStringAt penguins "island" i
        sex <- dataFrameStringAt penguins "sex" i
        putStrLn ("  row " ++ show i ++ ": bill_length=" ++ fmt billLength ++ " bill_depth=" ++ fmt billDepth ++ " "
                    ++ "island=" ++ island ++ " sex=" ++ sex ++ "  actual=" ++ actual ++ "  predicted=" ++ predicted)
        pure 1
  putStrLn (show misclassified ++ " misclassified (of " ++ show rows ++ " rows, some incomplete)")

  -- PlotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
  -- two-predictor model (bill measurements alone) just for visualization.
  billOnly <- newNaiveBayesClassifier penguins "species ~ bill_length_mm + bill_depth_mm"
  accuracy <- naiveBayesClassifierTrainingAccuracy billOnly
  putStrLn ("\nbill-measurements-only model training accuracy: " ++ fmt (100 * accuracy) ++ "%")
  naiveBayesClassifierPlotClassification billOnly penguins "bill_length_mm" "bill_depth_mm" >>= (`plotSave` "naive_bayes_penguins_classification.svg")
  naiveBayesClassifierPlotDecisionRegions billOnly "bill_length_mm" "bill_depth_mm" >>= (`plotSave` "naive_bayes_penguins_decision_regions.svg")
  putStrLn "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg"

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
