-- Copyright 2021 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

module MLIR.RewriteSpec where

import Test.Hspec

import Control.Monad.Identity

import MLIR.AST
import MLIR.AST.Builder
import MLIR.AST.Serialize
import MLIR.AST.Rewrite
import qualified MLIR.AST.Dialect.Arith as Arith
import qualified MLIR.AST.Dialect.Func as Func
import qualified MLIR.Native as MLIR


verifyAndDump :: Operation -> Expectation
verifyAndDump op =
  MLIR.withContext \ctx -> do
    MLIR.registerAllDialects ctx
    nativeOp <- fromAST ctx (mempty, mempty) op
    MLIR.dump nativeOp
    MLIR.verifyOperation nativeOp >>= (`shouldBe` True)


spec :: Spec
spec = do
  describe "Rewrite API" $ do
    it "Can replace adds with multiplies" $ do
      let m = runIdentity $ buildModule $
                buildSimpleFunction "f" [Float32Type] NoAttrs do
                  x <- blockArgument Float32Type
                  y <- blockArgument Float32Type
                  z <- Arith.addf x y
                  w <- Arith.addf z z
                  Func.return [w]
      let m' = applyClosedOpRewrite replaceAddWithMul m
      verifyAndDump m'
      where
        replaceAddWithMul op = case op of
          Arith.AddF _ _ x y -> ReplaceOne <$> Arith.mulf x y
          _                  -> return Traverse


main :: IO ()
main = hspec spec
