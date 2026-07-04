// tensor.hpp — a tiny reverse-mode autograd tensor.
//
// Design: value-semantics handle (Tensor) over a shared graph node
// (Tensor::Node). Ops record their inputs and a local backward closure that
// pushes gradients onto its inputs' `grad` buffers. `backward()` seeds the
// output gradient, walks the graph in reverse topological order, and runs each
// closure once. Data is a flat row-major buffer described by `shape`.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace tn {
class Tensor {};
}  // namespace tn
