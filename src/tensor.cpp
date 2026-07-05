// tensor.cpp — implementations for tensor.hpp. Definitions go here.
#include "tensor/tensor.hpp"

#include <assert.h>

#include <iomanip>
#include <memory>
#include <random>
#include <vector>

namespace tn {

// TODO: implement declarations from tensor.hpp
Tensor::Tensor(int rows, int cols, float fill) {
    node_ = std::make_shared<Node>(rows, cols, fill);
}

Tensor::Tensor(int rows, int cols) {
    node_ = std::make_shared<Node>(rows, cols, 0.0f);
}

void Tensor::set(int i, int j, float val) {
    assert(i >= 0 && i < node_->rows);
    assert(j >= 0 && j < node_->cols);
    node_->data[(i * node_->row_stride) + (j * node_->col_stride)] = val;
}

float Tensor::at(int i, int j) const {
    assert(i >= 0 && i < node_->rows);
    assert(j >= 0 && j < node_->cols);
    return node_->data[(i * node_->row_stride) + (j * node_->col_stride)];
}

Tensor Tensor::zeros(int rows, int cols) { return Tensor(rows, cols); }

Tensor Tensor::randn(int rows, int cols) {
    // One RNG shared across calls, seeded once. Not thread-safe; fine here.
    static std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<float> dist(0.0f, 1.0f);  // mean 0, stddev 1

    Tensor t(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) t.set(i, j, dist(gen));
    return t;
}

Tensor::Tensor(int rows, int cols, std::vector<float> data) {
    assert(data.size() == rows * cols);
    node_ = std::make_shared<Node>(rows, cols, std::move(data));
}

Tensor::Tensor(Node& n) { node_ = std::make_shared<Node>(n); }

Tensor Tensor::matmul(const Tensor& t) const {
    assert(this->node_->cols == t.node_->rows);
    // output tensor is of shape: this->node_->rows, t.node_->cols
    Tensor output(this->node_->rows, t.node_->cols);

    // Save away for the backward pass
    output.node_->prev.push_back(this->node_);
    output.node_->prev.push_back(t.node_);
    output.node_->opcode = OpCode::MATMUL;

    for (int p = 0; p < this->node_->rows; p++) {
        for (int q = 0; q < t.node_->cols; q++) {
            float val = 0.0f;
            for (int k = 0; k < this->node_->cols; k++) {
                val += (this->at(p, k) * t.at(k, q));
            }
            output.set(p, q, val);
        }
    }
    return output;
}

Tensor Tensor::transpose() const {
    Tensor t;
    t.node_ = std::make_shared<Node>(
        this->node_->cols, this->node_->rows, this->node_->col_stride,
        this->node_->row_stride, this->node_->data);
    return t;
}

void Tensor::matmul_backward(Tensor& upstream_gradient) {
    Tensor dA = upstream_gradient.matmul(
        Tensor(*(this->node_->prev[1])).transpose() /* B.T */);
    auto& ga = this->node_->prev[0]->grad;
    for (int i = 0; i < dA.node_->rows; i++)
        for (int j = 0; j < dA.node_->cols; j++)
            ga[i * this->node_->prev[0]->row_stride +
               j * this->node_->prev[0]->col_stride] += dA.at(i, j);

    Tensor dB =
        Tensor(*(this->node_->prev[0])).transpose().matmul(upstream_gradient);
    auto& gb = this->node_->prev[1]->grad;
    for (int i = 0; i < dB.node_->rows; i++)
        for (int j = 0; j < dB.node_->cols; j++)
            gb[i * this->node_->prev[1]->row_stride +
               j * this->node_->prev[1]->col_stride] += dB.at(i, j);
}

std::vector<float> Tensor::flatten() const {
    std::vector<float> output;
    output.reserve(this->node_->rows * this->node_->cols);
    if (this->node_->row_stride == this->node_->cols &&
        this->node_->col_stride == 1)
        output = this->node_->data;
    else {
        for (int i = 0; i < this->node_->rows; i++) {
            for (int j = 0; j < this->node_->cols; j++) {
                output.push_back(at(i, j));
            }
        }
    }
    return output;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << std::fixed << std::setprecision(5);
    for (int i = 0; i < t.node_->rows; ++i) {
        for (int j = 0; j < t.node_->cols; ++j) {
            os << std::setw(10) << t.at(i, j);
        }
        os << std::endl;
    }
    return os;
}

}  // namespace tn
