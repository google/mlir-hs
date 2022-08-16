-- Copyright 2022 Google LLC
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

module MLIR.AST.Dialect.ControlFlow
  ( module MLIR.AST.Dialect.ControlFlow,
    module MLIR.AST.Dialect.Generated.ControlFlow
  ) where

import Prelude hiding (return)
import Data.Array.IArray

import MLIR.AST
import MLIR.AST.Builder

import MLIR.AST.Dialect.Generated.ControlFlow

pattern Branch :: Location -> BlockName -> [Name] -> Operation
pattern Branch loc block args = Operation
  { opName = "cf.br"
  , opLocation = loc
  , opResultTypes = Explicit []
  , opOperands = args
  , opRegions = []
  , opSuccessors = [block]
  , opAttributes = NoAttrs
  }

br :: MonadBlockBuilder m => BlockName -> [Value] -> m EndOfBlock
br block args = emitOp (Branch UnknownLocation block $ operands args) >> terminateBlock

cond_br :: MonadBlockBuilder m => Value -> BlockName -> [Value] -> BlockName -> [Value] -> m EndOfBlock
cond_br cond trueBlock trueArgs falseBlock falseArgs = do
  emitOp_ $ Operation
    { opName = "cf.cond_br"
    , opLocation = UnknownLocation
    , opResultTypes = Explicit []
    , opOperands = operands $ [cond] <> trueArgs <> falseArgs
    , opRegions = []
    , opSuccessors = [trueBlock, falseBlock]
    , opAttributes = namedAttribute "operand_segment_sizes" $
                       DenseElementsAttr (VectorType [3] $ IntegerType Unsigned 32) $
                         DenseUInt32 $ listArray (0 :: Int, 2) $ fromIntegral <$> [1, length trueArgs, length falseArgs]
    }
  terminateBlock
