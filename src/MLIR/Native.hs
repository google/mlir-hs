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

{-|
This module defines a set of Haskell types wrapping references to native C++
MLIR objects along with some basic operations on them. See the submodules for
more specialized components such as an 'MLIR.Native.ExecutionEngine.ExecutionEngine'
or 'MLIR.Native.Pass.PassManager'.
-}
module MLIR.Native (
    -- * Contexts
    Context,
    createContext,
    destroyContext,
    withContext,
    HasContext(..),
    -- ** Dialect registration
    registerAllDialects,
    getNumLoadedDialects,
    -- * Type
    Type,
    -- * Location
    Location,
    getFileLineColLocation,
    getNameLocation,
    getUnknownLocation,
    -- * Operation
    Operation,
    getOperationName,
    showOperation,
    showOperationWithLocation,
    verifyOperation,
    -- * Region
    Region,
    getOperationRegions,
    getRegionBlocks,
    -- * Block
    Block,
    showBlock,
    getBlockOperations,
    -- * Module
    Module,
    createEmptyModule,
    parseModule,
    destroyModule,
    getModuleBody,
    moduleAsOperation,
    moduleFromOperation,
    showModule,
    -- * StringRef
    StringRef(..),
    withStringRef,
    -- * Identifier
    Identifier,
    createIdentifier,
    identifierString,
    -- * LogicalResult
    LogicalResult,
    pattern Failure,
    pattern Success,
    -- * Debugging utilities
    setDebugMode,
    HasDump(..),
  ) where

import qualified Data.ByteString as BS

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

-- | Creates a native MLIR context.
createContext :: IO Context
createContext = [C.exp| MlirContext { mlirContextCreate() } |]

-- | Destroys a native MLIR context.
destroyContext :: Context -> IO ()
destroyContext ctx = [C.exp| void { mlirContextDestroy($(MlirContext ctx)) } |]

-- | Wraps an IO action that gets access to a fresh MLIR context.
withContext :: (Context -> IO a) -> IO a
withContext = bracket createContext destroyContext

-- TODO(apaszke): Can this be pure?
-- | A typeclass for retrieving MLIR contexts managing other native types.
class HasContext a where
  -- | Retrieve the MLIR context that manages the storage of the native value.
  getContext :: a -> IO Context

--------------------------------------------------------------------------------
-- Dialect registration

-- | Register all builtin MLIR dialects in the specified 'Context'.
registerAllDialects :: Context -> IO ()
registerAllDialects ctx = [C.exp| void { mlirRegisterAllDialects($(MlirContext ctx)) } |]

-- | Retrieve the count of dialects currently registered in the 'Context'.
getNumLoadedDialects :: Context -> IO Int
getNumLoadedDialects ctx = fromIntegral <$>
  [C.exp| intptr_t { mlirContextGetNumLoadedDialects($(MlirContext ctx)) } |]

--------------------------------------------------------------------------------
-- Locations

-- | Create an unknown source location.
getUnknownLocation :: Context -> IO Location
getUnknownLocation ctx =
  [C.exp| MlirLocation { mlirLocationUnknownGet($(MlirContext ctx)) } |]

getFileLineColLocation :: Context -> StringRef -> C.CUInt -> C.CUInt -> IO Location
getFileLineColLocation ctx (StringRef sPtr len) line col  =
  [C.exp| MlirLocation {
    mlirLocationFileLineColGet(
      $(MlirContext ctx),
      (MlirStringRef){$(char* sPtr), $(size_t len)},
      $(unsigned int line),
      $(unsigned int col)) } |]

getNameLocation :: Context -> StringRef -> Location -> IO Location
getNameLocation ctx (StringRef sPtr len) childLoc =
  [C.exp| MlirLocation {
    mlirLocationNameGet(
      $(MlirContext ctx),
      (MlirStringRef){$(char* sPtr), $(size_t len)},
      $(MlirLocation childLoc)) } |]

-- TODO(apaszke): No destructor for locations?

--------------------------------------------------------------------------------
-- Operation

-- | Retrieve the name of the given operation.
getOperationName :: Operation -> IO Identifier
getOperationName op =
  [C.exp| MlirIdentifier { mlirOperationGetName($(MlirOperation op)) } |]

-- | Show the operation using the MLIR printer.
showOperation :: Operation -> IO BS.ByteString
showOperation op = showSomething \ctx ->
  [C.block| void {
    MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
    mlirOperationPrintWithFlags($(MlirOperation op), flags,
                                HaskellMlirStringCallback, $(void* ctx));
    mlirOpPrintingFlagsDestroy(flags);
  } |]

-- TODO(jpienaar): This should probably be more general options supported.
-- | Show the operation with location using the MLIR printer.
showOperationWithLocation :: Operation -> IO BS.ByteString
showOperationWithLocation op = showSomething \ctx ->
  [C.block| void {
    MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
    mlirOpPrintingFlagsEnableDebugInfo(flags, /*prettyForm=*/false);
    mlirOperationPrintWithFlags($(MlirOperation op), flags,
                                HaskellMlirStringCallback, $(void* ctx));
    mlirOpPrintingFlagsDestroy(flags);
  } |]

-- | Check validity of the operation.
verifyOperation :: Operation -> IO Bool
verifyOperation op =
  (1==) <$> [C.exp| bool { mlirOperationVerify($(MlirOperation op)) } |]

--------------------------------------------------------------------------------
-- Region

-- | Returns the first Region in a Operation.
getOperationFirstRegion :: Operation -> IO (Maybe Region)
getOperationFirstRegion op = nullable <$> [C.exp| MlirRegion {
    mlirOperationGetFirstRegion($(MlirOperation op))
  } |]

-- | Returns the next Block in a Region.
getOperationNextRegion :: Region -> IO (Maybe Region)
getOperationNextRegion region = nullable <$> [C.exp| MlirRegion {
    mlirRegionGetNextInOperation($(MlirRegion region))
  } |]

-- | Returns the regions of an Operation.
getOperationRegions :: Operation -> IO [Region]
getOperationRegions op = unrollIOMaybe getOperationNextRegion (getOperationFirstRegion op)

-- | Returns the first Block in a Region.
getRegionFirstBlock :: Region -> IO (Maybe Block)
getRegionFirstBlock region = nullable <$> [C.exp| MlirBlock {
    mlirRegionGetFirstBlock($(MlirRegion region))
  } |]

-- | Returns the next Block in a Region.
getRegionNextBlock :: Block -> IO (Maybe Block)
getRegionNextBlock block = nullable <$> [C.exp| MlirBlock {
    mlirBlockGetNextInRegion($(MlirBlock block))
  } |]

-- | Returns the Blocks in a Region.
getRegionBlocks :: Region -> IO [Block]
getRegionBlocks region = unrollIOMaybe getRegionNextBlock (getRegionFirstBlock region)

--------------------------------------------------------------------------------
-- Block

-- | Show the block using the MLIR printer.
showBlock :: Block -> IO BS.ByteString
showBlock block = showSomething \ctx -> [C.exp| void {
    mlirBlockPrint($(MlirBlock block), HaskellMlirStringCallback, $(void* ctx))
  } |]

-- | Returns the first operation in a block.
getFirstOperationBlock :: Block -> IO (Maybe Operation)
getFirstOperationBlock block = nullable <$>
  [C.exp| MlirOperation { mlirBlockGetFirstOperation($(MlirBlock block)) } |]

-- | Returns the next operation in the block. Returns 'Nothing' if last
-- operation in block.
getNextOperationBlock :: Operation -> IO (Maybe Operation)
getNextOperationBlock childOp = nullable <$> [C.exp| MlirOperation {
  mlirOperationGetNextInBlock($(MlirOperation childOp)) } |]

-- | Returns the Operations in a Block.
getBlockOperations :: Block -> IO [Operation]
getBlockOperations block = unrollIOMaybe getNextOperationBlock (getFirstOperationBlock block)

--------------------------------------------------------------------------------
-- Module

instance HasContext Module where
  getContext m = [C.exp| MlirContext { mlirModuleGetContext($(MlirModule m)) } |]

-- | Create an empty module.
createEmptyModule :: Location -> IO Module
createEmptyModule loc =
  [C.exp| MlirModule { mlirModuleCreateEmpty($(MlirLocation loc)) } |]

-- | Parse a module from a string. Returns 'Nothing' in case of parse failure.
parseModule :: Context -> StringRef -> IO (Maybe Module)
parseModule ctx (StringRef sPtr len) = nullable <$>
  [C.exp| MlirModule {
    mlirModuleCreateParse($(MlirContext ctx),
                          (MlirStringRef){$(char* sPtr), $(size_t len)})
  } |]

-- | Destroy all resources associated with a 'Module'.
destroyModule :: Module -> IO ()
destroyModule m =
  [C.exp| void { mlirModuleDestroy($(MlirModule m)) } |]

-- | Retrieve the block containg all module definitions.
getModuleBody :: Module -> IO Block
getModuleBody m = [C.exp| MlirBlock { mlirModuleGetBody($(MlirModule m)) } |]

-- TODO(apaszke): Can this be pure?
-- | Convert a module to an 'Operation'.
moduleAsOperation :: Module -> IO Operation
moduleAsOperation m =
  [C.exp| MlirOperation { mlirModuleGetOperation($(MlirModule m)) } |]

-- | Inverse of 'moduleAsOperation'. Returns 'Nothing' if the operation is not a
-- builtin MLIR module operation.
moduleFromOperation :: Operation -> IO (Maybe Module)
moduleFromOperation op =
  nullable <$> [C.exp| MlirModule { mlirModuleFromOperation($(MlirOperation op)) } |]

-- | Show the module using the MLIR printer.
showModule :: Module -> IO BS.ByteString
showModule = moduleAsOperation >=> showOperation

--------------------------------------------------------------------------------
-- StringRef

data StringRef = StringRef (Ptr C.CChar) C.CSize

-- MLIR sometimes expects null-terminated StringRefs, so we can't use
-- unsafeUseAsCStringLen, because ByteStrings are not guaranteed to have a terminator
-- | Use a 'BS.ByteString' as a 'StringRef'. This is O(n) due to MLIR sometimes
-- requiring the 'StringRef's to be null-terminated.
withStringRef :: BS.ByteString -> (StringRef -> IO a) -> IO a
withStringRef s f = BS.useAsCString s \ptr -> f $ StringRef ptr $ fromIntegral $ BS.length s

-- | Copy a 'StringRef' as a 'BS.ByteString'. This is an O(n) operation.
peekStringRef :: StringRef -> IO BS.ByteString
peekStringRef (StringRef ref size) = BS.packCStringLen (ref, fromIntegral size)

--------------------------------------------------------------------------------
-- Identifier

-- | View an identifier as a 'StringRef'. The result is valid for as long as the
-- 'Context' managing the identifier.
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

-- | Create an identifier from a 'StringRef'.
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

-- | Unroll using a function that is equivalent to "get next" inside IO.
unrollIOMaybe :: (a -> IO (Maybe a)) -> IO (Maybe a) -> IO [a]
unrollIOMaybe fn z = do
  x <- z
  case x of
      Nothing -> return []
      Just x' -> (x':) <$> unrollIOMaybe fn (fn x')

--------------------------------------------------------------------------------
-- Debugging utilities

-- | Enable or disable debug logging in MLIR.
setDebugMode :: Bool -> IO ()
setDebugMode enable = do
  let nativeEnable = if enable then 1 else 0
  [C.exp| void { mlirEnableGlobalDebug($(bool nativeEnable)) } |]


-- | A class for native objects that can be dumped to standard error output.
class HasDump a where
  -- | Display the value in the standard error output.
  dump :: a -> IO ()

instance HasDump Operation where
  dump op = [C.exp| void { mlirOperationDump($(MlirOperation op)) } |]

instance HasDump Module where
  dump = moduleAsOperation >=> dump
