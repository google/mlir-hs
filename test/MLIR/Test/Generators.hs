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

{-# OPTIONS_GHC -Wno-orphans #-}
module MLIR.Test.Generators where

import Control.Monad
import Test.QuickCheck
import GHC.Generics
import Generic.Random
import Data.Array.IArray
import qualified Data.Map.Strict as M
import qualified Data.ByteString.Char8 as BS8

import MLIR.AST
import qualified MLIR.AST.Dialect.Affine as Affine

instance Arbitrary Name where
  arbitrary = BS8.pack <$> arbitrary

instance Arbitrary Attribute where
  arbitrary = recursiveArbitrary leafGenerators recGenerators
    where
      leafGenerators =
        [ FloatAttr     <$> arbitraryFloatType   <*> arbitrary
        , IntegerAttr   <$> arbitraryIntegerType <*> arbitrary
        , BoolAttr      <$> arbitrary
        , StringAttr    <$> arbitrary
        , TypeAttr      <$> arbitrary
        , AffineMapAttr <$> arbitrary
        , pure UnitAttr
        , do
            values <- arbitrary
            return $ DenseElementsAttr
              (VectorType [length values] (IntegerType Signless 32))
              (DenseUInt32 $ listArray (1, length values) $ values)
        ]
      recGenerators =
        [ ArrayAttr <$> arbitrarySubtrees
        , DictionaryAttr . M.fromList <$> (traverse arbitraryName =<< arbitrarySubtrees)
        ]
      arbitraryName :: Attribute -> Gen (Name, Attribute)
      arbitraryName attr = (,attr) <$> arbitrary

arbitrarySubtrees :: Arbitrary a => Gen [a]
arbitrarySubtrees = sized $ \size -> do
  numSubtrees <- chooseInt (0, size)
  replicateM numSubtrees do
    subsize <- chooseInt (0, size - 1)
    resize subsize arbitrary

recursiveArbitrary :: [Gen a] -> [Gen a] -> Gen a
recursiveArbitrary leafGenerators recGenerators = sized $ \size -> do
  case size > 0 of
    False -> oneof leafGenerators
    True  -> frequency $ ((9,) <$> leafGenerators) <> ((1,) <$> recGenerators)

arbitraryFloatType :: Gen Type
arbitraryFloatType = oneof $ fmap pure
  [ BFloat16Type
  , Float128Type
  , Float16Type
  , Float32Type
  , Float64Type
  , Float80Type
  ]

arbitraryIntegerType :: Gen Type
arbitraryIntegerType = IntegerType <$> arbitrary <*> elements [1, 8, 16, 32, 64]

scalarTypeGenerators :: [Gen Type]
scalarTypeGenerators =
  [ pure BFloat16Type
  , ComplexType <$> arbitraryFloatType
  , pure Float128Type
  , pure Float16Type
  , pure Float32Type
  , pure Float64Type
  , pure Float80Type
  , pure IndexType
  , arbitraryIntegerType
  ]

arbitraryScalarType :: Gen Type
arbitraryScalarType = oneof scalarTypeGenerators

instance Arbitrary Type where
  arbitrary = recursiveArbitrary leafGenerators recGenerators
    where
      leafGenerators = scalarTypeGenerators ++
        [ MemRefType <$> arbitrary
                     <*> arbitraryScalarType
                     <*> frequency [(9, pure []     ), (1, arbitrary)]
                     <*> frequency [(9, pure Nothing), (1, arbitrary)]
        , pure NoneType
        , OpaqueType <$> arbitrary <*> arbitrary
        , RankedTensorType <$> frequency [(1, listOf $ Just <$> arbitrary), (1, arbitrary)]
                           <*> arbitraryScalarType
                           <*> arbitrary
        , UnrankedMemRefType <$> arbitraryScalarType <*> arbitrary
        , UnrankedTensorType <$> arbitraryScalarType
        , VectorType <$> frequency [(1, (:[]) <$> arbitrary), (1, arbitrary)]
                     <*> arbitraryScalarType
        ]
      recGenerators =
        [ FunctionType <$> arbitrarySubtrees <*> arbitrarySubtrees
        , TupleType    <$> arbitrarySubtrees
        ]


instance Arbitrary Signedness where
  arbitrary = genericArbitrary uniform

instance Arbitrary Affine.Map where
  arbitrary = genericArbitrary uniform

instance Arbitrary Affine.Expr where
  arbitrary = sized $ \size -> do
    case size > 0 of
      False -> oneof leafGenerators
      True -> do
        l <- choose (0, size)
        frequency $ ((9,) <$> leafGenerators) <> ((1,) <$> recGenerators l (size - l))
    where
      leafGenerators = fmap (<$> arbitrary)
          [Affine.Dimension, Affine.Symbol, Affine.Constant]
      recGenerators l r = smallerArbitrary2 l r <$>
          [Affine.Add, Affine.Mul, Affine.Mod, Affine.FloorDiv, Affine.CeilDiv]
      smallerArbitrary2 l r f = f <$> (resize l arbitrary) <*> (resize r arbitrary)

deriving instance Show Attribute
deriving instance Show Signedness
deriving instance Show Affine.Map
deriving instance Show Affine.Expr
deriving instance Show DenseElements

instance Show Type where
  show _ = "<MLIR type>"

deriving instance Generic Signedness
deriving instance Generic Affine.Map
deriving instance Generic Affine.Expr
