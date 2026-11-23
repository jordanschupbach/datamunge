-- CMD: just TARGET=bspline_ex run-haskell
-- Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
module Main (main) where

import Datamunge

main :: IO ()
main = do
  -- Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
  let grid = [(fromIntegral ix / 7.0, fromIntegral iy / 7.0) | iy <- [0 .. 7 :: Int], ix <- [0 .. 7 :: Int]] :: [(Double, Double)]
  dat <- newDataFrame
  dataFrameAddNumericColumn dat "x" (map fst grid)
  dataFrameAddNumericColumn dat "y" (map snd grid)
  dataFrameAddNumericColumn dat "z" [sin xv + yv * yv | (xv, yv) <- grid]

  surface <- newLm dat "z ~ bs(x, y)"
  putStrLn "Fitted z ~ bs(x, y)"
  lmPrintSummary surface

  newPoints <- newDataFrame
  dataFrameAddNumericColumn newPoints "x" [0.25, 0.75]
  dataFrameAddNumericColumn newPoints "y" [0.50, 0.25]
  predictions <- lmPredict surface newPoints
  putStrLn ("Predictions: [" ++ fixed 6 (predictions !! 0) ++ ", " ++ fixed 6 (predictions !! 1) ++ "]")

-- | C#'s ToString("F<n>"): the exact value rounded to n decimals, midpoints away from zero.
fixed :: Int -> Double -> String
fixed n x
  | isNaN x = "NaN"
  | isInfinite x = if x < 0 then "-∞" else "∞"
  | otherwise = sign ++ whole ++ (if n > 0 then '.' : frac else "")
  where
    scaled = floor (abs (toRational x) * 10 ^ n + 1 / 2) :: Integer
    sign = if x < 0 || isNegativeZero x then "-" else ""
    padded = let s = show scaled in replicate (n + 1 - length s) '0' ++ s
    (whole, frac) = splitAt (length padded - n) padded
