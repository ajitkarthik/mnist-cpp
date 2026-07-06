// tensor.cpp — implementations for tensor.hpp. Definitions go here.
#include "tensor/tensor.hpp"

#include <assert.h>

#include <algorithm>
#include <iomanip>
#include <memory>
#include <random>
#include <stack>
#include <unordered_set>
#include <vector>

namespace tn {

// TODO: implement declarations from tensor.hpp
Tensor::Tensor(int rows, int cols, float fill) {
    node_ = std::make_shared<Node>(rows, cols, fill);
}

Tensor::Tensor(int rows, int cols) { node_ = std::make_shared<Node>(rows, cols, 0.0f); }

void Tensor::set(int i, int j, float val) {
    assert(i >= 0 && i < rows());
    assert(j >= 0 && j < cols());
    node_->data[(i * node_->row_stride) + (j * node_->col_stride)] = val;
}

float Tensor::at(int i, int j) const {
    assert(i >= 0 && i < rows());
    assert(j >= 0 && j < cols());
    return node_->data[(i * node_->row_stride) + (j * node_->col_stride)];
}

float Tensor::grad_at(int i, int j) const {
    assert(i >= 0 && i < rows());
    assert(j >= 0 && j < cols());
    return node_->grad[(i * node_->row_stride) + (j * node_->col_stride)];
}

void Tensor::zero_grad() { std::fill(node_->grad.begin(), node_->grad.end(), 0.0f); }

int Tensor::rows() const { return node_->rows; }

int Tensor::cols() const { return node_->cols; }

float Tensor::sum() const {
    float sum = 0.0f;
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) {
            sum += at(i, j);
        }
    }
    return sum;
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
    assert(cols() == t.rows());
    // output tensor is of shape: node_->rows, t.node_->cols
    Tensor output(node_->rows, t.node_->cols);

    for (int p = 0; p < node_->rows; p++) {
        for (int q = 0; q < t.node_->cols; q++) {
            float val = 0.0f;
            for (int k = 0; k < node_->cols; k++) {
                val += (at(p, k) * t.at(k, q));
            }
            output.set(p, q, val);
        }
    }

    // Save away for the backward pass
    auto a = node_;
    auto b = t.node_;
    /* Note: here we save the raw pointer otherwise we get a circular ref with
     * shared pointers */
    auto out = output.node_.get();

    output.node_->backward = [a, b, out]() {
        // compute local gradient
        Tensor dC = Tensor(out->rows, out->cols, out->grad);
        Tensor dA = dC.matmul(Tensor(*b).transpose()); /* dA = dC @ B.T */
        Tensor dB = Tensor(*a).transpose().matmul(dC); /* dB = A.T @ dC */

        // flow the gradients backward
        for (int i = 0; i < dA.node_->rows; i++)
            for (int j = 0; j < dA.node_->cols; j++)
                a->grad[i * a->row_stride + j * a->col_stride] += dA.at(i, j);

        for (int i = 0; i < dB.node_->rows; i++)
            for (int j = 0; j < dB.node_->cols; j++)
                b->grad[i * b->row_stride + j * b->col_stride] += dB.at(i, j);
    };

    output.node_->prev.push_back(a);
    output.node_->prev.push_back(b);
    return output;
}

Tensor Tensor::add(const Tensor& t) const {
    assert(rows() == t.rows() && cols() == t.cols());
    Tensor output(rows(), cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) {
            output.set(i, j, at(i, j) + t.at(i, j));
        }
    }

    // Save away for the backward pass
    auto a = node_;
    auto b = t.node_;
    auto out = output.node_.get();
    output.node_->backward = [a, b, out]() {
        Tensor dC = Tensor(out->rows, out->cols, out->grad);

        for (int i = 0; i < dC.rows(); i++)
            for (int j = 0; j < dC.cols(); j++)
                a->grad[i * a->row_stride + j * a->col_stride] += dC.at(i, j);

        for (int i = 0; i < dC.node_->rows; i++)
            for (int j = 0; j < dC.node_->cols; j++)
                b->grad[i * b->row_stride + j * b->col_stride] += dC.at(i, j);
    };
    output.node_->prev.push_back(a);
    output.node_->prev.push_back(b);
    return output;
}

Tensor Tensor::add(const float val) {
    Tensor output(this->node_->rows, this->node_->cols, this->node_->data);
    for (int i = 0; i < this->node_->rows; i++) {
        for (int j = 0; j < this->node_->cols; j++) {
            output.set(i, j, this->at(i, j) + val);
        }
    }
    return output;
}

Tensor Tensor::clone() const {
    Tensor output(this->node_->rows, this->node_->cols, this->node_->data);
    return output;
}

/* Construct a topological sort of the graph, then walk through the sorted
 * nodes, calling the lambda for each node */
void Tensor::backward(void) {
    std::stack<Node*> stack;
    std::unordered_set<Node*> visited;

    // initialize gradient for the root node
    std::fill(this->node_->grad.begin(), this->node_->grad.end(), 1.0f);

    // Initialize root of toposort with the current node
    DFSVisit(this->node_.get(), visited, stack);

    while (!stack.empty()) {
        Node* top = stack.top();
        if (top->backward) top->backward();
        stack.pop();
    }
}

Tensor Tensor::transpose() const {
    Tensor t;
    t.node_ = std::make_shared<Node>(this->node_->cols, this->node_->rows,
                                     this->node_->col_stride, this->node_->row_stride,
                                     this->node_->data);
    return t;
}

std::vector<float> Tensor::flatten() const {
    std::vector<float> output;
    output.reserve(this->node_->rows * this->node_->cols);
    if (this->node_->row_stride == this->node_->cols && this->node_->col_stride == 1)
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

void Tensor::DFSVisit(Tensor::Node* curr, std::unordered_set<Node*>& visited,
                      std::stack<Node*>& stack) {
    visited.insert(curr);
    for (const auto& p : curr->prev) {
        if (!visited.contains(p.get())) {
            DFSVisit(p.get(), visited, stack);
        }
    }
    stack.push(curr);
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
