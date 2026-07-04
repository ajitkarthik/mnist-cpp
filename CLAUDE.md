# mnist-cpp

A mini tensor library in C++23 with reverse-mode autograd (backprop), plus an MLP
built on top of it, trained on MNIST.

> Status: greenfield. Most of the structure below is the *intended* design, not
> yet code. Update this file as the real layout solidifies.

## Goal

1. `Tensor` — an n-d array with a small op set and a computation graph.
2. Autograd — reverse-mode backprop over that graph.
3. `MLP` — a fully-connected network built from library primitives.
4. Train the MLP to classify MNIST digits.

## Proposed layout

```
mnist-cpp/
  include/tensor/   # public headers (Tensor, autograd, ops, nn layers)
  src/              # implementation (.cpp) if not header-only
  apps/train.cpp    # MNIST training entry point
  tests/            # unit tests (gradient checks, op correctness)
  data/             # MNIST idx files (gitignored)
  CMakeLists.txt
```

## Build & run

Toolchain requires a C++23 compiler (clang 17+ / gcc 13+). Assume CMake:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # run tests
./build/apps/train              # train on MNIST
```

Adjust these once the actual build system lands.

## Design notes / conventions

- **C++23**, no ML dependencies — the point is to build autograd from scratch.
  A minimal MNIST loader is fine; avoid pulling in a DL framework.
- **Autograd model:** each op records inputs and a local backward closure;
  `backward()` walks the graph in reverse topological order accumulating grads
  into `.grad`. Decide early: value-semantics vs. shared-ptr graph nodes, and
  document the choice here.
- **Verify gradients** with finite-difference checks in `tests/` for every op
  before trusting training. This is the single highest-leverage safety net.
- Keep the core (`Tensor` + ops + autograd) independent of anything MNIST- or
  MLP-specific so the library stays reusable.
- Prefer clarity over micro-optimization; correctness of gradients first,
  performance second.

## MNIST data

Standard IDX files (`train-images-idx3-ubyte`, etc.) under `data/`. Keep the raw
data out of git.
