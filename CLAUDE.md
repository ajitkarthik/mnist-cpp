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

## Learning context — READ THIS

This is a **learning exercise**, not a ship-it project. It implements Week 12–17
of a personal ML foundations plan
(`/Users/ajit/Code/ml-checklist/ml-fm/ml-foundation-models-learning-plan-2.md`,
section "Week 12–17: Extend C++ micrograd to Tensors + MNIST"). The user is
writing the code themselves to learn.

**How to help here:** explain, review, and unblock — but do NOT write the
implementation unless explicitly asked. Point out bugs, answer design questions,
and let the user type the code. When they ask "what's next", map against the
roadmap below.

### Roadmap (do these in order)

1. **Tensor / 2D matrix class** ← current step. Storage node holds both
   `std::vector<float> data` and `std::vector<float> grad`. Methods: `zeros`,
   `randn`, `at(i, j)`, `set(i, j, val)`, and `operator<<` for printing.
   Done = `std::cout << Tensor::randn(2, 3);` prints a matrix and a `grad`
   buffer sits ready alongside `data`.
2. **Matmul** forward `C = A @ B` (triple loop) + backward
   `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC`. Verify against a hand calculation.
3. **Element-wise ops + broadcasting** (add/sub/mul, bias add
   `(batch, features) + (1, features)`). Broadcast-backward sums the gradient
   back over the broadcast dim (e.g. bias grad sums across the batch).
4. **Activations** — ReLU, softmax (`exp(x - max) / sum`).
5. **Cross-entropy loss** — fuse softmax+CE; backward is
   `(softmax_output - one_hot) / batch_size`.
6. **MNIST loader** — parse IDX binary (16-byte header, then raw bytes),
   normalize to [0, 1], flatten to 784.
7. **2-layer MLP** — 784 → 128 (ReLU) → 10 (softmax), SGD
   `w -= lr * grad`. Target ~95% test accuracy.

Prerequisite before step 2: watch makemore Part 4 ("Backprop Ninja") — it derives
the matmul / broadcasting / softmax-CE backward passes.

Design constraints from the plan: **float only** (do not template on dtype),
row-major storage, `shared_ptr` graph nodes with per-node backward closures.
(The user chose `row_stride`/`col_stride` over the plan's single `stride`.)

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
