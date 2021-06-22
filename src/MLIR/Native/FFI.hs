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

{-# OPTIONS_HADDOCK hide #-}
module MLIR.Native.FFI where

import Foreign.Ptr
import Foreign.Storable
import qualified Language.C.Inline as C
import qualified Language.C.Types as C
import qualified Language.C.Inline.Context as C.Context

import Text.RawString.QQ

import Data.Int
import Data.Coerce
import qualified Data.Map as Map

C.include "<string.h>"
C.include "<stdlib.h>"
C.include "mlir-c/Support.h"

-- TODO(apaszke): Better buffering?
C.verbatim [r|
void HaskellMlirStringCallback(MlirStringRef ref, void* ctxRaw) {
  void** ctx = ctxRaw;
  char** data_ptr = ctxRaw;
  size_t* size_ptr = ctx[1];
  size_t old_size = *size_ptr;
  size_t new_size = old_size + ref.length;
  if (new_size == 0) return;
  *data_ptr = realloc(*data_ptr, new_size);
  *size_ptr = new_size;
  memcpy((*data_ptr) + old_size, ref.data, ref.length);
}
|]

stringCallbackDecl :: String
stringCallbackDecl = [r|
void HaskellMlirStringCallback(MlirStringRef ref, void* ctxRaw);
|]

data MlirContextObject
data MlirLocationObject
data MlirModuleObject
data MlirOperationObject
data MlirPassManagerObject
data MlirPassObject
data MlirExecutionEngineObject
data MlirTypeObject
data MlirBlockObject
data MlirRegionObject
data MlirAttributeObject
data MlirValueObject
data MlirIdentifierObject
data MlirAffineExprObject
data MlirAffineMapObject

-- | A native MLIR context.
newtype Context = ContextPtr (Ptr MlirContextObject)
                  deriving Storable via (Ptr ())
-- | A native MLIR pass instance.
newtype Pass = PassPtr (Ptr MlirPassObject)
               deriving Storable via (Ptr ())
-- | A native MLIR pass manager instance.
newtype PassManager = PassManagerPtr (Ptr MlirPassManagerObject)
                      deriving Storable via (Ptr ())
-- | A native MLIR location object.
newtype Location = LocationPtr (Ptr MlirLocationObject)
                   deriving Storable via (Ptr ())
-- | A native MLIR operation instance.
newtype Operation = OperationPtr (Ptr MlirOperationObject)
                    deriving Storable via (Ptr ())
-- | A native MLIR module operation.
-- Since every module is an operation, it can be converted to
-- an 'Operation' using 'MLIR.Native.moduleAsOperation'.
newtype Module = ModulePtr (Ptr MlirModuleObject)
                 deriving Storable via (Ptr ())
-- | A native MLIR execution engine.
newtype ExecutionEngine = ExecutionEnginePtr (Ptr MlirExecutionEngineObject)
                          deriving Storable via (Ptr ())
-- | A native MLIR type object.
newtype Type = TypePtr (Ptr MlirTypeObject)
               deriving Storable via (Ptr ())
-- | A native MLIR block object.
-- Every block is a list of 'Operation's.
newtype Block = BlockPtr (Ptr MlirBlockObject)
                deriving Storable via (Ptr ())
-- | A native MLIR region.
newtype Region = RegionPtr (Ptr MlirRegionObject)
                 deriving Storable via (Ptr ())
-- | A native MLIR attribute.
newtype Attribute = AttributePtr (Ptr MlirAttributeObject)
                    deriving Storable via (Ptr ())
-- | A native MLIR value object.
-- Every 'Value' is either a 'Block' argument or an output from an 'Operation'.
newtype Value = ValuePtr (Ptr MlirValueObject)
                deriving Storable via (Ptr ())
-- | A native MLIR identifier.
-- Identifiers are strings interned in the MLIR context.
newtype Identifier = IdentifierPtr (Ptr MlirIdentifierObject)
                     deriving Storable via (Ptr ())
-- | A native MLIR affine expression object.
newtype AffineExpr = AffineExprPtr (Ptr MlirAffineExprObject)
                     deriving Storable via (Ptr ())
-- | A native MLIR affine map object.
newtype AffineMap = AffineMapPtr (Ptr MlirAffineMapObject)
                    deriving Storable via (Ptr ())
data NamedAttribute  -- C structs cannot be represented in Haskell

-- | A result code for many failable MLIR operations.
-- The only valid cases are 'Success' and 'Failure'.
newtype LogicalResult = UnsafeMkLogicalResult Int8
                        deriving Storable via Int8
                        deriving Eq

instance Show LogicalResult where
  show Success = "Success"
  show Failure = "Failure"

-- | Indicates a successful completion of an MLIR operation.
pattern Success :: LogicalResult
pattern Success = UnsafeMkLogicalResult 1
-- | Indicates a filure of an MLIR operation. Inspect the diagnostics output
-- to find the cause of the issue.
pattern Failure :: LogicalResult
pattern Failure = UnsafeMkLogicalResult 0

{-# COMPLETE Success, Failure #-}

mlirCtx = mempty {
  -- This is a lie...
  -- All of those types are really C structs that hold a single pointer, but
  -- dealing with structs is just way too complicated. For simplicity, we
  -- assume that the layout of the struct is equal to the layout of a single
  -- pointer here, but I'm not 100% sure if that's a good assumption.
  C.Context.ctxTypesTable = Map.fromList [
    (C.TypeName "MlirContext", [t|Context|])
  , (C.TypeName "MlirLocation", [t|Location|])
  , (C.TypeName "MlirModule", [t|Module|])
  , (C.TypeName "MlirOperation", [t|Operation|])
  , (C.TypeName "MlirPassManager", [t|PassManager|])
  , (C.TypeName "MlirPass", [t|Pass|])
  , (C.TypeName "MlirExecutionEngine", [t|ExecutionEngine|])
  , (C.TypeName "MlirLogicalResult", [t|LogicalResult|])
  , (C.TypeName "MlirType", [t|Type|])
  , (C.TypeName "MlirBlock", [t|Block|])
  , (C.TypeName "MlirRegion", [t|Region|])
  , (C.TypeName "MlirAttribute", [t|Attribute|])
  , (C.TypeName "MlirNamedAttribute", [t|NamedAttribute|])
  , (C.TypeName "MlirValue", [t|Value|])
  , (C.TypeName "MlirIdentifier", [t|Identifier|])
  , (C.TypeName "MlirAffineExpr", [t|AffineExpr|])
  , (C.TypeName "MlirAffineMap", [t|AffineMap|])
  ]
}

nullable :: Coercible a (Ptr ()) => a -> Maybe a
nullable x = if coerce x == nullPtr then Nothing else Just x
