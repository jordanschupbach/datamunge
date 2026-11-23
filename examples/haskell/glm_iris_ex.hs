-- CMD: just TARGET=glm_iris_ex run-haskell
-- Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
module Main (main) where

import Control.Monad (filterM)
import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  iris <- dataIris
  n0 <- dataFrameNrows iris
  let rows = [0 .. n0 - 1]

  -- Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
  -- the other two on these predictors, which sends logistic regression's coefficients toward
  -- +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
  -- binary split for a demo.
  kept <- filterM (\i -> (`elem` ["versicolor", "virginica"]) <$> dataFrameStringAt iris "Species" i) rows
  isVirginica <- mapM (\i -> (\s -> if s == "virginica" then 1.0 else 0.0) <$> dataFrameStringAt iris "Species" i) kept
  petalLength <- mapM (dataFrameNumericAt iris "Petal.Length") kept
  petalWidth <- mapM (dataFrameNumericAt iris "Petal.Width") kept

  sub <- newDataFrame
  dataFrameAddNumericColumn sub "Petal.Length" petalLength
  dataFrameAddNumericColumn sub "Petal.Width" petalWidth
  dataFrameAddNumericColumn sub "is_virginica" isVirginica

  putStrLn "=================== Logistic regression (binomial, logit link) ==================="
  logit <- newGlmWithFamily sub "is_virginica ~ Petal.Length + Petal.Width" "binomial"
  glmPrintSummary logit

  fitted <- glmFittedValues logit
  let n = length isVirginica
      correct = length (filter id (zipWith (\f y -> (f >= 0.5) == (y >= 0.5)) fitted isVirginica))
  putStrLn ("\nResubstitution accuracy at 0.5 threshold: " ++ fmt (100.0 * fromIntegral correct / fromIntegral n) ++ "%")

  glmSaveDiagnosticPlots logit "glm_logistic_iris"
  putStrLn "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg"

  -- Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
  -- classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
  do
    let gridN = 100 :: Int
        widthMean = sum petalWidth / fromIntegral (length petalWidth)
        plMin = minimum petalLength - 0.3
        plMax = maximum petalLength + 0.3
    grid <- newDataFrame
    dataFrameAddNumericColumn grid "Petal.Length" [plMin + (plMax - plMin) * fromIntegral i / fromIntegral (gridN - 1) | i <- [0 .. gridN - 1]]
    dataFrameAddNumericColumn grid "Petal.Width" (replicate gridN widthMean)
    curveFrame <- glmPredictFrameWithIntervalKind logit grid "confidence"
    putStrLn "\nPredicted-probability curve (first 5 rows):"
    dataFrameToStringWithMaxRows curveFrame 5 >>= putStrLn

  -- Poisson regression, for contrast: same IRLS engine, different family/link.
  putStrLn "\n=================== Poisson regression (log link) ==================="
  countData <- newDataFrame
  mapM (dataFrameNumericAt iris "Sepal.Width") rows >>= dataFrameAddNumericColumn countData "Sepal.Width"
  mapM (dataFrameNumericAt iris "Petal.Length") rows >>= dataFrameAddNumericColumn countData "Petal.Length"
  mapM (fmap (\x -> fromIntegral (floor (x + 0.5) :: Int)) . dataFrameNumericAt iris "Sepal.Length") rows
    >>= dataFrameAddNumericColumn countData "count"
  newGlmWithFamily countData "count ~ Sepal.Width + Petal.Length" "poisson" >>= glmPrintSummary

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
