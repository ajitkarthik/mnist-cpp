# mnist-cpp

A from-scratch mini tensor library in **C++23** with reverse-mode autograd
(backprop), plus a 2-layer MLP built on top of it and trained to classify MNIST
digits. No ML dependencies — the point is to build the autograd engine by hand.

## Results

A 784 → 128 (ReLU) → 128 (ReLU) → 10 (softmax) MLP trained with plain SGD
reaches **98.05% test accuracy** (9413 / 9600 correct) after 20,000 mini-batch
iterations (batch size 32), comfortably past the 95% target.

Learning rate is 0.1 for the first 10,000 iterations, then decays to 0.01.
Weights use He/Kaiming init (`sqrt(2 / fan_in)`) on the two ReLU-feeding layers
and Xavier (`sqrt(1 / fan_in)`) on the output layer; biases start at zero.

All three RNG streams are seeded to fixed constants — weight init
(`Tensor::seed`), training mini-batches, and evaluation mini-batches — so the
run is **reproducible**: repeated invocations produce byte-identical output.
That makes A/B comparisons of architecture or hyperparameter changes
meaningful, since batch order and init can no longer explain a difference.

Training run (Release build, ~86s total):

```
Reading label file ... done
Number of labels: 60000
Reading image file ... done
Number of images: 60000
[0s]Iteration:0      Loss:   2.37746
[4s]Iteration:1000   Loss:   0.04152
[9s]Iteration:2000   Loss:   0.09568
[13s]Iteration:3000  Loss:   0.44890
[18s]Iteration:4000  Loss:   0.04246
[22s]Iteration:5000  Loss:   0.01949
[27s]Iteration:6000  Loss:   0.04889
[31s]Iteration:7000  Loss:   0.00220
[36s]Iteration:8000  Loss:   0.01590
[40s]Iteration:9000  Loss:   0.02551
[45s]Iteration:10000 Loss:   0.01280
[49s]Iteration:11000 Loss:   0.00175
[54s]Iteration:12000 Loss:   0.00344
[58s]Iteration:13000 Loss:   0.01457
[63s]Iteration:14000 Loss:   0.03735
[68s]Iteration:15000 Loss:   0.00155
[72s]Iteration:16000 Loss:   0.02464
[77s]Iteration:17000 Loss:   0.01448
[81s]Iteration:18000 Loss:   0.01362
[86s]Iteration:19000 Loss:   0.13759
Running testing ...
Number of labels: 10000
Number of images: 10000
Correct: 9413
Incorrect: 187
% correct: 0.98052
```

Note: evaluation samples 300 random mini-batches of 32 (9,600 draws with
replacement) from the 10,000-image test set rather than doing one exact pass,
so the accuracy is an estimate over a resample, not a strict full-test-set
score.

## Design

Value-semantics handle (`Tensor`) over a shared graph node (`Tensor::Node`).
Each op records its inputs and a local backward closure that pushes gradients
onto its inputs' `grad` buffers. `backward()` seeds the output gradient, walks
the graph in reverse topological order, and runs each closure once. Data is a
flat row-major buffer with explicit `row_stride` / `col_stride`.

Key constraints (from the learning plan):

- **float only** — no templating on dtype.
- Row-major storage; `shared_ptr` graph nodes with per-node backward closures.
- Gradients accumulate with `+=`; call `zero_grad()` on parameters each step.
- Correctness of gradients first, performance second.

## Layout

```
mnist-cpp/
  include/tensor/tensor.hpp   # public Tensor API + autograd
  src/tensor.cpp              # implementation
  apps/train.cpp              # MNIST training + test-set evaluation
  tests/grad_check.cpp        # finite-difference gradient checks
  data/                       # MNIST IDX files (gitignored)
  CMakeLists.txt
```

## Tensor ops

Forward + backward implemented and gradient-checked:

| Op | Notes |
|---|---|
| `matmul` | `C = A @ B`; `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC` |
| `add_bias` | `(rows, cols) + (1, cols)`; bias grad sums over the batch dim |
| `relu` | `max(0, x)` |
| `softmax` | per-row `exp(x − rowmax) / rowsum` (forward only) |
| `fused_cross_entropy_loss` | fused softmax + CE; backward `(probs − one_hot) / batch_size` |

Helpers: `randn`, `zeros`, `seed`, `at`, `set`, `grad_at`, `val`, `sum`,
`max_idx`, `transpose`, `clone`, `zero_grad`, `adjust_weights`
(SGD `w -= lr * grad`), `backward`, `operator<<`.

`Tensor::seed(uint32_t)` reseeds the generator behind `randn`. By default it is
seeded from `std::random_device`, so determinism is opt-in from the application
rather than baked into the library.

## Build & run

Requires a C++23 compiler (clang 17+ / gcc 13+) and CMake.

Two build trees — **Debug for tests** (keeps asserts and shape checks live),
**Release for training** (10×+ faster on the strided matmul):

```bash
# Debug — run the gradient checks
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build

# Release — train + evaluate on MNIST
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
cd build-release && ./train
```

`train` reads the IDX files from `../data/` (relative to the working dir), so
run it from inside the build tree as shown.

## MNIST data

Standard IDX files under `data/` (kept out of git):

```
train-images.idx3-ubyte  train-labels.idx1-ubyte
t10k-images.idx3-ubyte   t10k-labels.idx1-ubyte
```

Big-endian format: 16-byte header for images (magic `0x803`), 8-byte header for
labels (magic `0x801`), then raw `uint8` pixel / label bytes. Pixels are
normalized to `[0, 1]` at batch-assembly time.
