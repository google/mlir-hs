# mlir-hs - Haskell bindings for MLIR

**🚨 This is an early-stage project. All details are subject to arbitrary changes. 🚨**

Note that the `main` branch tracks the current HEAD of [LLVM](https://github.com/llvm/llvm-project)
and so it is likely to be incompatible with any past releases. We are planning to
provide release-specifi branches in the future, but only once the API stabilizes.
For now your best bet is to develop against MLIR built from source. See the
[Building MLIR from source](#building-mlir-from-source) section for guidance.

## Building

The only prerequisite for building mlir-hs is that you have MLIR installed
somewhere, and the `llvm-config` binary from that installation is available
in your `PATH` (a good way to verify this is to run `which llvm-config`).

If that's looking reasonable, we recommend using [Stack](https://haskellstack.org)
for development. To build the project simply run `stack build`, while the test
suite can be executed using `stack test`.

### Building MLIR from source

The instructions below assume that you have `cmake` and `ninja` installed.
You should be able to get them from your favorite package manager.

  1. Clone the latest LLVM code (or use `git pull` if you cloned it before) into the root of this repository
     ```bash
     git clone https://github.com/llvm/llvm-project
     ```

  2. Create a temporary build directory
     ```bash
     mkdir llvm-project/build
     ```

  3. Configure the build using CMake. Remember to replace `$PREFIX` with the directory
     where you want MLIR to be installed. See [LLVM documentation](https://llvm.org/docs/CMake.html)
     for extended explanation and other potentially interesting build flags.
     ```bash
     cmake -B llvm-project/build           \
       -G Ninja                            \ # Use the Ninja build system
       -DLLVM_ENABLE_PROJECTS=mlir         \ # Enable build MLIR
       -DCMAKE_INSTALL_PREFIX=$PREFIX      \ # Install prefix
       -DMLIR_BUILD_MLIR_C_DYLIB=ON        \ # Build shared libraries
       -DLLVM_BUILD_LLVM_DYLIB=ON          \
       llvm-project/llvm
     ```
     For development purposes we additionally recommend using
     `-DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON`
     to retain debug information and enable internal LLVM assertions. If one changes
     the install directory (CMAKE_INSTALL_PREFIX) then one needs to add this directory
     to PATH and LD_LIBRARY_PATH for the subsequent builds (e.g., `stack`) to find it.

  4. [Build and install MLIR]. Note that it uses the installation prefix specified
     in the previous step.
     ```bash
     cmake --build llvm-project/build -t install
     ```

## Contributing

Contributions of all kinds are welcome! If you're planning to implement a larger feature,
consider posting an issue so that we can discuss it before you put in the work.

## License

See the LICENSE file.

mlir-hs is an early-stage project, not an official Google product.
