-- CMD: just TARGET=autodiff_ex run-haskell
-- Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
module Main (main) where

import Control.Monad (foldM_, forM, forM_, when)
import Datamunge
import Numeric (floatToDigits)

-- Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
-- (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
-- bound. This port demonstrates the same ideas with the bound named-method API.
-- varIndex is the Var's position in the adjoint vector tapeBackward returns.

rosenbrockDual :: Dual -> Dual -> IO Dual
rosenbrockDual x0 x1 = do
  a <- newDualWithDerivative 1.0 0.0 >>= (`dualSubtract` x0)
  b <- dualMultiply x0 x0 >>= dualSubtract x1
  aa <- dualMultiply a a
  bb <- dualMultiply b b >>= (`dualMultiplyScalar` 100.0)
  dualAdd aa bb

rosenbrockValue :: Double -> Double -> Double
rosenbrockValue x0 x1 = (1.0 - x0) ^ (2 :: Int) + 100.0 * (x1 - x0 * x0) ^ (2 :: Int)

-- | Builds the Rosenbrock graph on tape; returns the output Var.
rosenbrockVar :: Tape -> Var -> Var -> IO Var
rosenbrockVar tape x0 x1 = do
  a <- newVar tape 1.0 >>= (`varSubtract` x0)
  b <- varMultiply x0 x0 >>= varSubtract x1
  aa <- varMultiply a a
  bb <- varMultiply b b >>= (`varMultiplyScalar` 100.0)
  varAdd aa bb

rosenbrockHyperDual :: HyperDual -> HyperDual -> IO HyperDual
rosenbrockHyperDual x0 x1 = do
  a <- newHyperDualWithEps1eps2 1.0 0.0 0.0 0.0 >>= (`hyperDualSubtract` x0)
  b <- hyperDualMultiply x0 x0 >>= hyperDualSubtract x1
  aa <- hyperDualMultiply a a
  bb <- hyperDualMultiply b b >>= (`hyperDualMultiplyScalar` 100.0)
  hyperDualAdd aa bb

adjoint :: [Double] -> Var -> IO Double
adjoint adjoints v = (adjoints !!) <$> varIndex v

main :: IO ()
main = do
  putStrLn "=================== Forward mode: scalar derivative ==================="
  do
    x0 <- newDualWithDerivative 2.0 1.0 -- seed derivative = 1 to read df/dx directly
    cube <- dualMultiply x0 x0 >>= dualMultiply x0
    f <- dualMultiplyScalar x0 2.0 >>= dualSubtract cube -- x^3 - 2x
    derivative <- dualDerivative f
    putStrLn ("f(x) = x^3 - 2x, f'(2) = " ++ fmt derivative ++ " (exact: 10)")

  putStrLn "\n=================== Reverse mode: build a graph by hand ==================="
  do
    tape <- newTape
    a <- newVar tape 2.0
    b <- newVar tape 3.0
    ab <- varMultiply a b
    y <- varSin a >>= varAdd ab
    yValue <- varValue y
    putStrLn ("y = a*b + sin(a) at a=2, b=3 -> y = " ++ fmt yValue)
    adjoints <- tapeBackward tape y
    da <- adjoint adjoints a
    putStrLn ("dy/da = " ++ fmt da ++ " (exact: b + cos(a))")
    db <- adjoint adjoints b
    putStrLn ("dy/db = " ++ fmt db ++ " (exact: a)")

  putStrLn "\n=================== Forward vs reverse mode agree on the Rosenbrock function ==================="
  do
    let (px, py) = (0.0, 0.0)
    -- Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
    gx <- do
      x <- newDualWithDerivative px 1.0
      y <- newDualWithDerivative py 0.0
      rosenbrockDual x y >>= dualDerivative
    gy <- do
      x <- newDualWithDerivative px 0.0
      y <- newDualWithDerivative py 1.0
      rosenbrockDual x y >>= dualDerivative
    -- Reverse mode: one Tape/Var pass computes every partial at once.
    tape <- newTape
    vx <- newVar tape px
    vy <- newVar tape py
    grad <- rosenbrockVar tape vx vy >>= tapeBackward tape
    putStrLn ("f(0,0) = " ++ fmt (rosenbrockValue px py))
    putStrLn ("gradient (forward mode): [" ++ fmt gx ++ ", " ++ fmt gy ++ "]")
    rx <- adjoint grad vx
    ry <- adjoint grad vy
    putStrLn ("gradient (reverse mode): [" ++ fmt rx ++ ", " ++ fmt ry ++ "]")

  putStrLn "\n=================== Jacobian of a vector-valued function ==================="
  -- f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
  do
    let vectorFn x y = sequence [dualMultiply x x, dualMultiply x y, dualMultiply y y >>= dualMultiply y]
    -- One column per input direction: the derivatives of every output.
    columns <- forM [(1.0, 0.0), (0.0, 1.0)] $ \(sx, sy) -> do
      x <- newDualWithDerivative 2.0 sx
      y <- newDualWithDerivative 3.0 sy
      vectorFn x y >>= mapM dualDerivative
    putStrLn "f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:"
    forM_ [0 .. 2] $ \row ->
      putStrLn ("  [" ++ fmt (columns !! 0 !! row) ++ ", " ++ fmt (columns !! 1 !! row) ++ "]")

  putStrLn "\n=================== Hessian via second-order forward mode (HyperDual) ==================="
  -- One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
  do
    let (mx, my) = (1.0, 1.0) -- the Rosenbrock function's minimum
        seeds = [(1.0, 0.0), (0.0, 1.0)]
    h <- forM seeds $ \(e1x, e1y) -> forM seeds $ \(e2x, e2y) -> do
      x <- newHyperDualWithEps1eps2 mx e1x e2x 0.0
      y <- newHyperDualWithEps1eps2 my e1y e2y 0.0
      rosenbrockHyperDual x y >>= hyperDualEps1eps2
    putStrLn "Hessian of the Rosenbrock function at its minimum (1,1):"
    forM_ h $ \row -> putStrLn ("  [" ++ fmt (row !! 0) ++ ", " ++ fmt (row !! 1) ++ "]")

  putStrLn "\n=================== Gradient descent driven by reverse-mode gradients ==================="
  do
    let learningRate = 0.001
        nSteps = 2000 :: Int
    foldM_ (\(x, y) step -> do
               tape <- newTape
               vx <- newVar tape x
               vy <- newVar tape y
               vf <- rosenbrockVar tape vx vy
               grad <- tapeBackward tape vf
               loss <- varValue vf
               x' <- (\g -> x - learningRate * g) <$> adjoint grad vx
               y' <- (\g -> y - learningRate * g) <$> adjoint grad vy
               when (step == 0 || step == nSteps - 1) $
                 putStrLn ("step " ++ show step ++ ": loss = " ++ fmt loss ++ ", x = [" ++ fmt x' ++ ", " ++ fmt y' ++ "]")
               pure (x', y'))
           (-1.2, 1.0) -- the classic Rosenbrock start
           [0 .. nSteps - 1]
    putStrLn "(true minimum is at [1, 1] with loss 0)"

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
