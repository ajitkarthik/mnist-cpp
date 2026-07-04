// tensor.cpp — implementations for tensor.hpp. Definitions go here.
#include "tensor/tensor.hpp"

#include <memory>
#include <random>
#include <assert.h>

namespace tn {

// TODO: implement declarations from tensor.hpp
Tensor::Tensor(int rows, int cols, float fill) {
    node_ = std::make_shared<Node>(rows, cols, fill);
}

Tensor::Tensor(int rows, int cols) {
    node_ = std::make_shared<Node>(rows, cols, 0.0);
}

void Tensor::set(int i, int j, float val) {
    assert (i >= 0 && i < node_->rows);
    assert (j >= 0 && j < node_->cols);
    node_->data[i*node_->row_stride + j] = val;
}

float Tensor::at(int i, int j) {
    assert (i >= 0 && i < node_->rows);
    assert (j >= 0 && j < node_->cols);
    return node_->data[i*node_->row_stride + j];
}

Tensor Tensor::zeros(int rows, int cols) {
    return Tensor(rows, cols);
}

Tensor Tensor::randn(int rows, int cols) {
    // One RNG shared across calls, seeded once. Not thread-safe; fine here.
    static std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<float> dist(0.0f, 1.0f);  // mean 0, stddev 1

    Tensor t(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) t.set(i, j, dist(gen));
    return t;
}

}  // namespace tn
