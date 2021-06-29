module MLIR.AST.IStorableArray (IStorableArray, unsafeWithIStorableArray) where

import Data.Ix
import Data.Array.Storable
import Data.Array.Base
import Foreign.Ptr
import Foreign.Storable
import System.IO.Unsafe

newtype IStorableArray i e = UnsafeIStorableArray (StorableArray i e)

unsafeWithIStorableArray :: IStorableArray i e -> (Ptr e -> IO c) -> IO c
unsafeWithIStorableArray (UnsafeIStorableArray arr) = withStorableArray arr

instance Storable e => IArray IStorableArray e where
  bounds (UnsafeIStorableArray arr) = unsafeDupablePerformIO $ getBounds arr
  numElements = rangeSize . bounds
  unsafeArray bs inits = unsafeDupablePerformIO $ do
    arr <- newArray_ bs
    mapM_ (uncurry $ unsafeWrite arr) inits
    return $ UnsafeIStorableArray arr
  unsafeAt (UnsafeIStorableArray arr) i = unsafeDupablePerformIO $ unsafeRead arr i

instance (Ix i, Show i, Show e, Storable e) => Show (IStorableArray i e) where
  showsPrec = showsIArray

instance (Ix i, Eq e, Storable e) => Eq (IStorableArray i e) where
  a == b = (bounds a == bounds b) &&
    (all id [unsafeAt a i == unsafeAt b i | i <- [0 .. numElements a - 1]])
