-- Smoke tests of the Rosetta-generated Haskell binding (just test-haskell).
module Main (main) where

import Control.Exception (Exception, throwIO, try)
import Control.Monad (unless)
import Data.IORef
import Datamunge
import System.Exit (exitFailure)

data FromHaskell = FromHaskell deriving (Show)

instance Exception FromHaskell

main :: IO ()
main = do
  checks <- newIORef (0 :: Int)
  let check ok message = do
        modifyIORef' checks (+ 1)
        unless ok $ do
          putStrLn ("check failed: " ++ message)
          exitFailure

  -- Smoke test: call into the native library.
  hello

  -- std::vector<double> and std::pair<double,double> round trips.
  v <- makeDvector 1 2 3
  check (v == [1, 2, 3]) "std::vector<double> is a list"
  s <- sumDvector v
  check (s == 6) "sumDvector"
  p <- makeDpair 1.25 2.75
  check (p == (1.25, 2.75)) "std::pair is a tuple"
  sp <- sumDpair p
  check (sp == 4) "sumDpair"

  -- The C++ Callback class, and an extended Callback whose virtual call runs Haskell (a director).
  native <- newCallback
  r1 <- callWithCallback 3 native
  check (r1 == 3) "native Callback"
  m1 <- mapDvectorWithCallback [1, 2, 3] native >>= sumDvector
  check (m1 == 6) "native map"
  doubler <- extendCallback callbackOverrides {callbackCallOverride = Just (\_ x -> pure (2 * x))}
  r2 <- callWithCallback 3 doubler
  check (r2 == 6) "the Haskell override runs"
  m2 <- mapDvectorWithCallback [1, 2, 3] doubler >>= sumDvector
  check (m2 == 12) "override from C++ loops"
  closeObject native

  -- A Haskell exception inside an override is thrown again after C++ returns.
  failing <- extendCallback callbackOverrides {callbackCallOverride = Just (\_ _ -> throwIO FromHaskell)}
  thrown <- try (callWithCallback 1 failing)
  check (either (\FromHaskell -> True) (const False) thrown) "the Haskell exception propagates"

  -- C++ exceptions throw CppError with their kind.
  bad <- try (runif 1 0 3)
  check (either ((== InvalidArgument) . errorKind) (const False) bad) "C++ exceptions throw CppError"

  -- Classes, methods and the policy namespace data.
  frame <- newDataFrame
  dataFrameAddNumericColumn frame "x" [1, 2, 3]
  rows <- dataFrameNrows frame
  check (rows == 3) "DataFrame"
  closeObject frame
  sq <- seql 0 1 5
  check (sq == [0, 0.25, 0.5, 0.75, 1]) "seql"
  iris <- dataIris
  irisRows <- dataFrameNrows iris
  check (irisRows == 150) "the data namespace"

  n <- readIORef checks
  putStrLn ("test_datamunge.hs: all " ++ show n ++ " checks passed")
