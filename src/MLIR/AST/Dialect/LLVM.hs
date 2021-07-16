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

module MLIR.AST.Dialect.LLVM (
  -- * Types
    Type(..)
  , pattern Ptr
  , pattern Array
  , pattern Void
  , pattern LiteralStruct
  -- * Operations
  , module MLIR.AST.Dialect.Generated.LLVM
  ) where

import MLIR.AST.Dialect.Generated.LLVM

import Data.Typeable
import Control.Monad.IO.Class
import Control.Monad.Trans.Cont
import qualified Language.C.Inline as C

import qualified MLIR.AST           as AST
import qualified MLIR.AST.Serialize as AST
import qualified MLIR.Native        as Native
import qualified MLIR.Native.FFI    as Native

C.context $ C.baseCtx <> Native.mlirCtx
C.include "mlir-c/Dialect/LLVM.h"

data Type = PointerType AST.Type
          | ArrayType Int AST.Type
          | VoidType
          | LiteralStructType [AST.Type]
          -- TODO(apaszke): Structures, functions, vectors, etc.
          deriving Eq

instance AST.FromAST Type Native.Type where
  fromAST ctx env ty = case ty of
    PointerType t -> do
      nt <- AST.fromAST ctx env t
      [C.exp| MlirType { mlirLLVMPointerTypeGet($(MlirType nt), 0) } |]
    ArrayType size t -> do
      nt <- AST.fromAST ctx env t
      let nsize = fromIntegral size
      [C.exp| MlirType { mlirLLVMArrayTypeGet($(MlirType nt), $(unsigned int nsize)) } |]
    VoidType -> [C.exp| MlirType { mlirLLVMVoidTypeGet($(MlirContext ctx)) } |]
    LiteralStructType fields -> evalContT $ do
      (numFields, nativeFields) <- AST.packFromAST ctx env fields
      liftIO $ [C.exp| MlirType {
        mlirLLVMStructTypeLiteralGet($(MlirContext ctx), $(intptr_t numFields),
                                     $(MlirType* nativeFields), false)
      } |]


castLLVMType :: AST.Type -> Maybe Type
castLLVMType ty = case ty of
  AST.DialectType dty -> cast dty
  _                   -> Nothing

pattern Ptr :: AST.Type -> AST.Type
pattern Ptr t <- (castLLVMType -> Just (PointerType t))
  where Ptr t = AST.DialectType (PointerType t)

pattern Array :: Int -> AST.Type -> AST.Type
pattern Array n t <- (castLLVMType -> Just (ArrayType n t))
  where Array n t = AST.DialectType (ArrayType n t)

pattern Void :: AST.Type
pattern Void <- (castLLVMType -> Just VoidType)
  where Void = AST.DialectType VoidType

pattern LiteralStruct :: [AST.Type] -> AST.Type
pattern LiteralStruct fields <- (castLLVMType -> Just (LiteralStructType fields))
  where LiteralStruct fields = AST.DialectType (LiteralStructType fields)
