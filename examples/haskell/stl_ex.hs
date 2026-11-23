-- CMD: just TARGET=stl_ex run-haskell
-- C++ std::vector and std::pair values in Haskell: vectors are lists, pairs are tuples.
module Main (main) where

import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  let n = 100 :: Int
      v = [fromIntegral i * 1.5 | i <- [0 .. n - 1]] :: [Double]
  mapM_ (putStrLn . fmt) v
  let v2 = [floor (fromIntegral i * 1.5 :: Double) | i <- [0 .. n - 1]] :: [Int]
  mapM_ print v2

  let p = (3, 4) :: (Int, Int)
  putStrLn ("p: (" ++ show (fst p) ++ ", " ++ show (snd p) ++ ")")

  let p2 = (10, 20) :: (Int, Int)
  putStrLn ("p2: (" ++ show (fst p2) ++ ", " ++ show (snd p2) ++ ")")

  -- A C++ std::pair<double,double> comes back as a tuple.
  p3@(a, b) <- makeDpair 1.25 2.75
  s <- sumDpair p3
  putStrLn ("make_dpair(1.25, 2.75): (" ++ fmt a ++ ", " ++ fmt b ++ "), sum_dpair = " ++ fmt s)

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
