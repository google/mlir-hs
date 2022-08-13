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

module MLIR.AST.PatternUtil
  ( pattern I32ArrayAttr
  , pattern I64ArrayAttr
  , pattern AffineMapArrayAttr
  , DummyIx
  ) where

import Data.Traversable
import Data.Array

import MLIR.AST
import qualified MLIR.AST.Dialect.Affine as Affine

unwrapI32ArrayAttr :: Attribute -> Maybe [Int]
unwrapI32ArrayAttr (ArrayAttr vals) = for vals \case
  IntegerAttr (IntegerType Signed 32) v -> Just v
  _                                     -> Nothing
unwrapI32ArrayAttr _ = Nothing

pattern I32ArrayAttr :: [Int] -> Attribute
pattern I32ArrayAttr vals <- (unwrapI32ArrayAttr -> Just vals)
  where I32ArrayAttr vals = ArrayAttr $ fmap (IntegerAttr (IntegerType Signed 32)) vals

unwrapI64ArrayAttr :: Attribute -> Maybe [Int]
unwrapI64ArrayAttr (ArrayAttr vals) = for vals \case
  IntegerAttr (IntegerType Signed 64) v -> Just v
  _                                     -> Nothing
unwrapI64ArrayAttr _ = Nothing

pattern I64ArrayAttr :: [Int] -> Attribute
pattern I64ArrayAttr vals <- (unwrapI64ArrayAttr -> Just vals)
  where I64ArrayAttr vals = ArrayAttr $ fmap (IntegerAttr (IntegerType Signed 64)) vals

unwrapAffineMapArrayAttr :: Attribute -> Maybe [Affine.Map]
unwrapAffineMapArrayAttr (ArrayAttr vals) = for vals \case
  AffineMapAttr m -> Just m
  _               -> Nothing
unwrapAffineMapArrayAttr _ = Nothing

pattern AffineMapArrayAttr :: [Affine.Map] -> Attribute
pattern AffineMapArrayAttr vals <- (unwrapAffineMapArrayAttr -> Just vals)
  where AffineMapArrayAttr vals = ArrayAttr $ fmap AffineMapAttr vals

data DummyIx
deriving instance Eq DummyIx
deriving instance Ord DummyIx
deriving instance Show DummyIx
instance Ix DummyIx where
  range   _   = error "Invalid index"
  index   _ _ = error "Invalid index"
  inRange _ _ = error "Invalid index"
