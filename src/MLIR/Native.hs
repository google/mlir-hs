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

module MLIR.Native (
    LogicalResult,
    pattern Failure,
    pattern Success,
    -- Contexts
    Context,
    createContext,
    destroyContext,
    withContext,
    HasContext(..),
    -- Dialect registration
    registerAllDialects,
    getNumLoadedDialects,
    -- Location
    Location,
    getUnknownLocation,
    -- Operation
    Operation,
    getOperationName,
    showOperation,
    verifyOperation,
    -- Block
    Block,
    showBlock,
    -- Module
    Module,
    createEmptyModule,
    parseModule,
    destroyModule,
    getModuleBody,
    moduleAsOperation,
    moduleFromOperation,
    showModule,
    -- StringRef
    StringRef(..),
    withStringRef,
    -- Identifier
    Identifier,
    createIdentifier,
    identifierString,
    -- Debugging
    setDebugMode,
    HasDump(..),
  ) where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Unsafe as BS

import Foreign.Ptr
import Foreign.Storable
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import qualified Language.C.Inline as C

import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Trans.Cont
import Control.Exception (bracket)

import MLIR.Native.FFI

C.context $ C.baseCtx <> mlirCtx

C.include "mlir-c/Support.h"
C.include "mlir-c/Debug.h"
C.include "mlir-c/IR.h"
C.include "mlir-c/Pass.h"
C.include "mlir-c/Conversion.h"
C.include "mlir-c/Registration.h"

C.verbatim stringCallbackDecl

-- TODO(apaszke): Flesh this out based on the header

--------------------------------------------------------------------------------
-- Context management

createContext :: IO Context
createContext = [C.exp| MlirContext { mlirContextCreate() } |]

destroyContext :: Context -> IO ()
destroyContext ctx = [C.exp| void { mlirContextDestroy($(MlirContext ctx)) } |]

withContext :: (Context -> IO a) -> IO a
withContext = bracket createContext destroyContext

-- TODO(apaszke): Can this be pure?
class HasContext a where
  getContext :: a -> IO Context

--------------------------------------------------------------------------------
-- Dialect registration

registerAllDialects :: Context -> IO ()
registerAllDialects ctx = [C.exp| void { mlirRegisterAllDialects($(MlirContext ctx)) } |]

getNumLoadedDialects :: Context -> IO Int
getNumLoadedDialects ctx = fromIntegral <$>
  [C.exp| intptr_t { mlirContextGetNumLoadedDialects($(MlirContext ctx)) } |]

--------------------------------------------------------------------------------
-- Locations

getUnknownLocation :: Context -> IO Location
getUnknownLocation ctx =
  [C.exp| MlirLocation { mlirLocationUnknownGet($(MlirContext ctx)) } |]

-- TODO(apaszke): No destructor for locations?

--------------------------------------------------------------------------------
-- Operation

getOperationName :: Operation -> IO Identifier
getOperationName op =
  [C.exp| MlirIdentifier { mlirOperationGetName($(MlirOperation op)) } |]

showOperation :: Operation -> IO BS.ByteString
showOperation op = showSomething \ctx ->
  [C.block| void {
    MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
    mlirOperationPrintWithFlags($(MlirOperation op), flags,
                                HaskellMlirStringCallback, $(void* ctx));
    mlirOpPrintingFlagsDestroy(flags);
  } |]

verifyOperation :: Operation -> IO Bool
verifyOperation op =
  (1==) <$> [C.exp| bool { mlirOperationVerify($(MlirOperation op)) } |]

--------------------------------------------------------------------------------
-- Block

showBlock :: Block -> IO BS.ByteString
showBlock block = showSomething \ctx -> [C.exp| void {
    mlirBlockPrint($(MlirBlock block), HaskellMlirStringCallback, $(void* ctx))
  } |]

--------------------------------------------------------------------------------
-- Module

instance HasContext Module where
  getContext m = [C.exp| MlirContext { mlirModuleGetContext($(MlirModule m)) } |]

createEmptyModule :: Location -> IO Module
createEmptyModule loc =
  [C.exp| MlirModule { mlirModuleCreateEmpty($(MlirLocation loc)) } |]

parseModule :: Context -> StringRef -> IO (Maybe Module)
parseModule ctx (StringRef sPtr len) = nullable <$>
  [C.exp| MlirModule {
    mlirModuleCreateParse($(MlirContext ctx),
                          (MlirStringRef){$(char* sPtr), $(size_t len)})
  } |]

destroyModule :: Module -> IO ()
destroyModule m =
  [C.exp| void { mlirModuleDestroy($(MlirModule m)) } |]

getModuleBody :: Module -> IO Block
getModuleBody m = [C.exp| MlirBlock { mlirModuleGetBody($(MlirModule m)) } |]

-- TODO(apaszke): Can this be pure?
moduleAsOperation :: Module -> IO Operation
moduleAsOperation m =
  [C.exp| MlirOperation { mlirModuleGetOperation($(MlirModule m)) } |]

moduleFromOperation :: Operation -> IO (Maybe Module)
moduleFromOperation op =
  nullable <$> [C.exp| MlirModule { mlirModuleFromOperation($(MlirOperation op)) } |]

showModule :: Module -> IO BS.ByteString
showModule = moduleAsOperation >=> showOperation

--------------------------------------------------------------------------------
-- StringRef

data StringRef = StringRef (Ptr C.CChar) C.CSize

withStringRef :: BS.ByteString -> (StringRef -> IO a) -> IO a
withStringRef s f = BS.unsafeUseAsCStringLen s \(ptr, len) -> f $ StringRef ptr $ fromIntegral len

peekStringRef :: StringRef -> IO BS.ByteString
peekStringRef (StringRef ref size) = BS.packCStringLen (ref, fromIntegral size)

--------------------------------------------------------------------------------
-- Identifier

identifierString :: Identifier -> IO StringRef
identifierString ident = evalContT $ do
  namePtrPtr <- ContT alloca
  sizePtr    <- ContT alloca
  liftIO $ do
    [C.block| void {
      MlirStringRef identStr = mlirIdentifierStr($(MlirIdentifier ident));
      *$(const char** namePtrPtr) = identStr.data;
      *$(size_t* sizePtr) = identStr.length;
    } |]
    StringRef <$> peek namePtrPtr <*> peek sizePtr

createIdentifier :: Context -> StringRef -> IO Identifier
createIdentifier ctx (StringRef ref size) =
  [C.exp| MlirIdentifier {
    mlirIdentifierGet($(MlirContext ctx), (MlirStringRef){$(char* ref), $(size_t size)})
  } |]

--------------------------------------------------------------------------------
-- Utilities

showSomething :: (Ptr () -> IO ()) -> IO BS.ByteString
showSomething action = do
  allocaArray @(Ptr ()) 2 \ctx ->
    alloca @C.CSize \sizePtr -> do
      poke sizePtr 0
      pokeElemOff ctx 0 nullPtr
      pokeElemOff ctx 1 $ castPtr sizePtr
      let ctxFlat = (castPtr ctx) :: Ptr ()
      action ctxFlat
      dataPtr <- castPtr <$> peek ctx
      size <- peek sizePtr
      bs <- peekStringRef $ StringRef dataPtr size
      free dataPtr
      return bs

--------------------------------------------------------------------------------
-- Debugging utilities

setDebugMode :: Bool -> IO ()
setDebugMode enable = do
  let nativeEnable = if enable then 1 else 0
  [C.exp| void { mlirEnableGlobalDebug($(bool nativeEnable)) } |]

class HasDump a where
  dump :: a -> IO ()

instance HasDump Operation where
  dump op = [C.exp| void { mlirOperationDump($(MlirOperation op)) } |]

instance HasDump Module where
  dump = moduleAsOperation >=> dump
