#include <cmath>
#include <iostream>
#include <print>

#include "../include/tensor/tensor.hpp"

using namespace tn;

// Combined absolute/relative tolerance, the criterion PyTorch's gradcheck uses:
//   pass if |numeric - analytic| <= atol + rtol * |analytic|
// Pure relative error is fragile near zero — a fixed float noise floor divided
// by a shrinking gradient blows up. atol absorbs that floor; rtol still catches
// genuinely wrong large gradients. Counts failures into `failures`.
static constexpr float kAtol = 1e-3f;
static constexpr float kRtol = 1e-2f;

static void check(const char* name, int i, int j, float analytic, float numeric, int& failures) {
    float abs_err = std::abs(numeric - analytic);
    float thresh = kAtol + kRtol * std::abs(analytic);
    bool pass = abs_err <= thresh;
    if (!pass) failures++;
    std::print("{}[{}, {}]={} dL={} abs_err={} thresh={} {}\n", name, i, j, analytic, numeric,
               abs_err, thresh, pass ? "ok" : "FAIL");
}

// File compares the analytic gradient with the numerical gradient
int main() {
    // Large step: every op here makes L linear in the perturbed input, so the
    // central difference has zero truncation error and a big e just cuts the
    // float cancellation noise. Revisit once nonlinear ops (ReLU/softmax) land.
    float e = 1e-2;
    int failures = 0;

    Tensor A = Tensor::randn(3, 4);
    Tensor B = Tensor::randn(4, 2);
    Tensor C = Tensor::randn(3, 4);
    Tensor D = Tensor::randn(1, 4);
    Tensor logits = Tensor::randn(32, 10);  // batchsize=32, features=10

    // matmul
    std::cout << "===MATMUL===\n";
    Tensor test1 = A.matmul(B);
    test1.backward();

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            Tensor A_plus = A.clone();
            Tensor A_minus = A.clone();
            A_plus.set(i, j, A.at(i, j) + e);
            A_minus.set(i, j, A.at(i, j) - e);
            float L_plus = (A_plus.matmul(B)).sum();
            float L_minus = (A_minus.matmul(B)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dA", i, j, A.grad_at(i, j), numeric, failures);
        }
    }
    A.zero_grad();

    for (int i = 0; i < B.rows(); i++) {
        for (int j = 0; j < B.cols(); j++) {
            Tensor B_plus = B.clone();
            Tensor B_minus = B.clone();
            B_plus.set(i, j, B.at(i, j) + e);
            B_minus.set(i, j, B.at(i, j) - e);
            float L_plus = (A.matmul(B_plus)).sum();
            float L_minus = (A.matmul(B_minus)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dB", i, j, B.grad_at(i, j), numeric, failures);
        }
    }
    B.zero_grad();

    // add
    std::cout << "===ADD===\n";
    Tensor test2 = A.add(C);
    test2.backward();

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            Tensor A_plus = A.clone();
            Tensor A_minus = A.clone();
            A_plus.set(i, j, A.at(i, j) + e);
            A_minus.set(i, j, A.at(i, j) - e);
            float L_plus = (A_plus.add(C)).sum();
            float L_minus = (A_minus.add(C)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dA", i, j, A.grad_at(i, j), numeric, failures);
        }
    }
    A.zero_grad();

    for (int i = 0; i < C.rows(); i++) {
        for (int j = 0; j < C.cols(); j++) {
            Tensor C_plus = C.clone();
            Tensor C_minus = C.clone();
            C_plus.set(i, j, C.at(i, j) + e);
            C_minus.set(i, j, C.at(i, j) - e);
            float L_plus = (A.add(C_plus)).sum();
            float L_minus = (A.add(C_minus)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dC", i, j, C.grad_at(i, j), numeric, failures);
        }
    }
    C.zero_grad();

    // sub
    std::cout << "===SUB===\n";
    Tensor test3 = A.sub(C);
    test3.backward();

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            Tensor A_plus = A.clone();
            Tensor A_minus = A.clone();
            A_plus.set(i, j, A.at(i, j) + e);
            A_minus.set(i, j, A.at(i, j) - e);
            float L_plus = (A_plus.sub(C)).sum();
            float L_minus = (A_minus.sub(C)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dA", i, j, A.grad_at(i, j), numeric, failures);
        }
    }
    A.zero_grad();

    for (int i = 0; i < C.rows(); i++) {
        for (int j = 0; j < C.cols(); j++) {
            Tensor C_plus = C.clone();
            Tensor C_minus = C.clone();
            C_plus.set(i, j, C.at(i, j) + e);
            C_minus.set(i, j, C.at(i, j) - e);
            float L_plus = (A.sub(C_plus)).sum();
            float L_minus = (A.sub(C_minus)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dC", i, j, C.grad_at(i, j), numeric, failures);
        }
    }
    C.zero_grad();

    // mul (Hadamard)
    std::cout << "===HADAMARD PRODUCT===\n";
    Tensor test4 = A.mul(C);
    test4.backward();

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            Tensor A_plus = A.clone();
            Tensor A_minus = A.clone();
            A_plus.set(i, j, A.at(i, j) + e);
            A_minus.set(i, j, A.at(i, j) - e);
            float L_plus = (A_plus.mul(C)).sum();
            float L_minus = (A_minus.mul(C)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dA", i, j, A.grad_at(i, j), numeric, failures);
        }
    }
    A.zero_grad();

    for (int i = 0; i < C.rows(); i++) {
        for (int j = 0; j < C.cols(); j++) {
            Tensor C_plus = C.clone();
            Tensor C_minus = C.clone();
            C_plus.set(i, j, C.at(i, j) + e);
            C_minus.set(i, j, C.at(i, j) - e);
            float L_plus = (A.mul(C_plus)).sum();
            float L_minus = (A.mul(C_minus)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dC", i, j, C.grad_at(i, j), numeric, failures);
        }
    }
    C.zero_grad();

    // bias addition
    std::cout << "===BIAS ADD===\n";
    Tensor test5 = A.add_bias(D);
    test5.backward();

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            Tensor A_plus = A.clone();
            Tensor A_minus = A.clone();
            A_plus.set(i, j, A.at(i, j) + e);
            A_minus.set(i, j, A.at(i, j) - e);
            float L_plus = (A_plus.add_bias(D)).sum();
            float L_minus = (A_minus.add_bias(D)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dA", i, j, A.grad_at(i, j), numeric, failures);
        }
    }
    A.zero_grad();

    for (int i = 0; i < D.rows(); i++) {
        for (int j = 0; j < D.cols(); j++) {
            Tensor D_plus = D.clone();
            Tensor D_minus = D.clone();
            D_plus.set(i, j, D.at(i, j) + e);
            D_minus.set(i, j, D.at(i, j) - e);
            float L_plus = (A.add_bias(D_plus)).sum();
            float L_minus = (A.add_bias(D_minus)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dD", i, j, D.grad_at(i, j), numeric, failures);
        }
    }
    D.zero_grad();

    // ReLU
    std::cout << "===RELU===\n";
    Tensor test6 = A.relu();
    test6.backward();

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            if (std::abs(A.at(i, j)) < e)  // in other words, if A + e > 0 and A - e < 0
                continue;                  // straddles the ReLU knee where derivative is undefined
            Tensor A_plus = A.clone();
            Tensor A_minus = A.clone();
            A_plus.set(i, j, A.at(i, j) + e);
            A_minus.set(i, j, A.at(i, j) - e);
            float L_plus = (A_plus.relu()).sum();
            float L_minus = (A_minus.relu()).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dA", i, j, A.grad_at(i, j), numeric, failures);
        }
    }
    A.zero_grad();

    // cross-entropy loss
    std::cout << "===Cross Entropy Loss===\n";
    Tensor labels(32, 1);
    // initialize the labels to some number between 0 and 9
    for (int i = 0; i < labels.rows(); i++) {
        labels.set(i, 0, i % 10);
    }

    Tensor loss = logits.fused_cross_entropy_loss(labels);
    loss.backward();

    for (int i = 0; i < logits.rows(); i++) {
        for (int j = 0; j < logits.cols(); j++) {
            Tensor logits_plus = logits.clone();
            Tensor logits_minus = logits.clone();
            logits_plus.set(i, j, logits.at(i, j) + e);
            logits_minus.set(i, j, logits.at(i, j) - e);
            float L_plus = (logits_plus.fused_cross_entropy_loss(labels)).sum();
            float L_minus = (logits_minus.fused_cross_entropy_loss(labels)).sum();
            float numeric = (L_plus - L_minus) / (2 * e);

            check("dlogits", i, j, logits.grad_at(i, j), numeric, failures);
        }
    }
    logits.zero_grad();

    if (failures > 0) {
        std::print("FAIL: {} gradient element(s) exceeded tolerance (atol={}, rtol={})\n", failures,
                   kAtol, kRtol);
        return 1;
    }
    std::print("PASS: all gradient elements within tolerance (atol={}, rtol={})\n", kAtol, kRtol);
    return 0;
}
