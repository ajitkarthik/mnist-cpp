// tensor.hpp — a tiny reverse-mode autograd tensor.
//
// Design: value-semantics handle (Tensor) over a shared graph node
// (Tensor::Node). Ops record their inputs and a local backward closure that
// pushes gradients onto its inputs' `grad` buffers. `backward()` seeds the
// output gradient, walks the graph in reverse topological order, and runs each
// closure once. Data is a flat row-major buffer described by `shape`.
#pragma once

#include <memory>
#include <vector>

namespace tn {
class Tensor {
   private:
    struct Node {
        std::vector<float> data;
        std::vector<float> grad;
        int rows;
        int cols;
        int row_stride;
        int col_stride;

        Node(int rows, int cols, float fill)
            : data(rows * cols, fill),
              grad(rows * cols, 0.0),
              rows(rows),
              cols(cols),
              row_stride(cols),
              col_stride(1) {}
    };

   public:
    // Empty/undefined tensor (no storage). Assign into it later.
    Tensor() = default;

    // Allocate a rows x cols tensor, elements zero-initialized.
    Tensor(int rows, int cols);

    // Allocate a rows x cols tensor with every element set to `fill`.
    Tensor(int rows, int cols, float fill);

    // Wrap existing row-major data; data.size() must equal rows * cols.
    Tensor(int rows, int cols, std::vector<float> data);

    // set element [i, j]
    void set(int i, int j, float val);

    // return element [i, j]
    float at (int i, int j);

    static Tensor randn (int rows, int cols);
    static Tensor zeros (int rows, int cols);

    std::shared_ptr<Node> node_;
};
}  // namespace tn
