-- CMD: just TARGET=kernel_regression_iris_ex run-haskell
-- Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
module Main (main) where

import Datamunge
import Numeric (floatToDigits)

formula :: String
formula = "Petal.Length ~ Petal.Width"

main :: IO ()
main = do
  iris <- dataIris
  rows <- dataFrameNrows iris
  cols <- dataFrameNcols iris
  putStrLn ("iris: " ++ show rows ++ " rows x " ++ show cols ++ " cols")
  putStrLn ("formula: " ++ formula ++ "\n")

  model <- newKernelRegression iris formula -- bandwidth auto-selected via leave-one-out CV
  kernelRegressionPrintSummary model

  kernelRegressionPlotFit model iris >>= (`plotSave` "kernel_regression_iris_fit.svg")
  kernelRegressionPlotCvCurve model >>= (`plotSave` "kernel_regression_iris_cv.svg")
  putStrLn "\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg"

  -- A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
  -- bias-variance story. newKernelRegressionWith... data formula kernel bandwidth nBandwidth standardize
  small <- newKernelRegressionWithBandwidth iris formula "gaussian" 0.05
  smallR2 <- kernelRegressionRSquared small
  smallRmse <- kernelRegressionRmse small
  putStrLn ("\nbandwidth=0.05 (too small): LOO R-squared=" ++ fmt smallR2 ++ "  LOO RMSE=" ++ fmt smallRmse)
  kernelRegressionPlotFit small iris >>= (`plotSave` "kernel_regression_iris_fit_small_bandwidth.svg")

  large <- newKernelRegressionWithBandwidth iris formula "gaussian" 5.0
  largeR2 <- kernelRegressionRSquared large
  largeRmse <- kernelRegressionRmse large
  putStrLn ("bandwidth=5.0 (too large):  LOO R-squared=" ++ fmt largeR2 ++ "  LOO RMSE=" ++ fmt largeRmse)
  kernelRegressionPlotFit large iris >>= (`plotSave` "kernel_regression_iris_fit_large_bandwidth.svg")

  bandwidth <- kernelRegressionBandwidth model
  r2 <- kernelRegressionRSquared model
  rmse <- kernelRegressionRmse model
  putStrLn ("bandwidth=" ++ fmt bandwidth ++ " (CV-selected): LOO R-squared=" ++ fmt r2 ++ "  LOO RMSE=" ++ fmt rmse)
  putStrLn "\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg"

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
