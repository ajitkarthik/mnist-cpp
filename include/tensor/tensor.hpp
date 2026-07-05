// tensor.hpp — a tiny reverse-mode autograd tensor.
//
// Design: value-semantics handle (Tensor) over a shared graph node
// (Tensor::Node). Ops record their inputs and a local backward closure that
// pushes gradients onto its inputs' `grad` buffers. `backward()` seeds the
// output gradient, walks the graph in reverse topological order, and runs each
// closure once. Data is a flat row-major buffer described by `shape`.
#pragma once

#include <memory>
#include <ostream>
#include <vector>

namespace tn {
class Tensor {
   private:
    enum class OpCode { NONE, MATMUL };
    struct Node {
        std::vector<float> data;
        std::vector<float> grad;
        int rows;
        int cols;
        int row_stride;
        int col_stride;
        OpCode opcode;
        // track the computational graph
        std::vector<std::shared_ptr<Node>> prev;

        Node(int rows, int cols, float fill)
            : data(rows * cols, fill),
              grad(rows * cols, 0.0f),
              rows(rows),
              cols(cols),
              row_stride(cols),
              col_stride(1),
              opcode(OpCode::NONE) {}

        Node(int rows, int cols, std::vector<float> data)
            : data(std::move(data)),
              grad(
                  rows * cols,
                  0.0f),  // can't use data.size() because of the move operation
              rows(rows),
              cols(cols),
              row_stride(cols),
              col_stride(1),
              opcode(OpCode::NONE) {}

        Node(int rows, int cols, int row_stride, int col_stride,
             std::vector<float> data)
            : data(std::move(data)),
              grad(
                  rows * cols,
                  0.0f),  // can't use data.size() because of the move operation
              rows(rows),
              cols(cols),
              row_stride(row_stride),
              col_stride(col_stride),
              opcode(OpCode::NONE) {}
    };

    // Lambdas to calculate local derivatives
    void matmul_backward(Tensor& upstream_gradient);

   public:
    // Empty/undefined tensor (no storage). Assign into it later.
    Tensor() = default;

    // Allocate a rows x cols tensor, elements zero-initialized.
    Tensor(int rows, int cols);

    // Allocate a rows x cols tensor with every element set to `fill`.
    Tensor(int rows, int cols, float fill);

    // Wrap existing row-major data; data.size() must equal rows * cols.
    Tensor(int rows, int cols, std::vector<float> data);

    // Given a node, get it's corresponding Tensor handle
    Tensor(Node& n);

    // set element [i, j]
    void set(int i, int j, float val);

    // return element [i, j]
    float at(int i, int j) const;

    // flatten a Tensor to a vector<float>
    std::vector<float> flatten() const;

    static Tensor randn(int rows, int cols);
    static Tensor zeros(int rows, int cols);

    // backward pass
    void local_backward();

    // Transpose
    // NOTE: This operation does not maintain gradients
    Tensor transpose() const;

    // GRADIENT MAINTAINING FUNCTIONS
    // Matrix multiplication
    Tensor matmul(const Tensor&) const;

    std::shared_ptr<Node> node_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);
}  // namespace tn
