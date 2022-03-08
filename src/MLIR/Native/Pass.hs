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

module MLIR.Native.Pass where

import qualified Language.C.Inline as C

import Control.Exception (bracket)

import MLIR.Native.FFI

C.context $ C.baseCtx <> mlirCtx

C.include "mlir-c/IR.h"
C.include "mlir-c/Pass.h"
C.include "mlir-c/Conversion.h"

-- TODO(apaszke): Flesh this out based on the header

--------------------------------------------------------------------------------
-- Pass manager

createPassManager :: Context -> IO PassManager
createPassManager ctx =
  [C.exp| MlirPassManager { mlirPassManagerCreate($(MlirContext ctx)) } |]

destroyPassManager :: PassManager -> IO ()
destroyPassManager pm =
  [C.exp| void { mlirPassManagerDestroy($(MlirPassManager pm)) } |]

withPassManager :: Context -> (PassManager -> IO a) -> IO a
withPassManager ctx = bracket (createPassManager ctx) destroyPassManager

runPasses :: PassManager -> Module -> IO LogicalResult
runPasses pm m =
  [C.exp| MlirLogicalResult { mlirPassManagerRun($(MlirPassManager pm), $(MlirModule m)) } |]

--------------------------------------------------------------------------------
-- Transform passes

--------------------------------------------------------------------------------
-- Conversion passes

addConvertMemRefToLLVMPass :: PassManager -> IO ()
addConvertMemRefToLLVMPass pm =
  [C.exp| void {
    mlirPassManagerAddOwnedPass($(MlirPassManager pm), mlirCreateConversionConvertMemRefToLLVM())
  } |]

addConvertFuncToLLVMPass :: PassManager -> IO ()
addConvertFuncToLLVMPass pm =
  [C.exp| void {
    mlirPassManagerAddOwnedPass($(MlirPassManager pm), mlirCreateConversionConvertFuncToLLVM())
  } |]

addConvertVectorToLLVMPass :: PassManager -> IO ()
addConvertVectorToLLVMPass pm =
  [C.exp| void {
    mlirPassManagerAddOwnedPass($(MlirPassManager pm), mlirCreateConversionConvertVectorToLLVM())
  } |]

addConvertReconcileUnrealizedCastsPass :: PassManager -> IO ()
addConvertReconcileUnrealizedCastsPass pm =
  [C.exp| void {
    mlirPassManagerAddOwnedPass($(MlirPassManager pm), mlirCreateConversionReconcileUnrealizedCasts())
  } |]
