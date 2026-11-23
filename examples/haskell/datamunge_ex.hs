-- CMD: just TARGET=datamunge_ex run-haskell
-- Smoke test of the basic bindings: vectors, pairs and a native Callback object.
module Main (main) where

import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  hello

  v <- makeDvector 1.0 2.0 3.0
  s <- sumDvector v
  putStrLn ("sum_dvector:\t" ++ fmt s)

  p <- makeDpair 1.25 2.75
  sp <- sumDpair p
  putStrLn ("sum_dpair:\t" ++ fmt sp)

  cb <- newCallback
  r <- callWithCallback 3.0 cb
  putStrLn ("call_with_callback(3.0):\t" ++ fmt r)
  v2 <- makeDvector 1.0 2.0 3.0 >>= \w -> mapDvectorWithCallback w cb
  s2 <- sumDvector v2
  putStrLn ("sum_dvector(map_dvector_with_callback(1,2,3)):\t" ++ fmt s2)
  closeObject cb

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
