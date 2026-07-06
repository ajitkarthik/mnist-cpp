#include <algorithm>
#include <cmath>
#include <iostream>
#include <print>

#include "../include/tensor/tensor.hpp"

using namespace tn;

// Max relative error between the two, tracked into `worst`.
static float check(const char* name, int i, int j, float analytic, float numeric,
                   float& worst) {
    float rel_error =
        std::abs(numeric - analytic) / (std::abs(numeric) + std::abs(analytic) + 1e-8f);
    worst = std::max(worst, rel_error);
    std::print("{}[{}, {}]={} dL={} rel_err={}\n", name, i, j, analytic, numeric,
               rel_error);
    return rel_error;
}

// File compares the analytic gradient with the numerical gradient
int main() {
    float e = 1e-4;
    const float tol = 1e-2f;  // loose: central difference over float storage
    float worst = 0.0f;

    Tensor A = Tensor::randn(3, 4);
    Tensor B = Tensor::randn(4, 2);
    Tensor C = Tensor::randn(3, 4);

    // matmul
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

            check("dA", i, j, A.grad_at(i, j), numeric, worst);
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

            check("dB", i, j, B.grad_at(i, j), numeric, worst);
        }
    }
    B.zero_grad();

    // add
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

            check("dA", i, j, A.grad_at(i, j), numeric, worst);
        }
    }
    A.zero_grad();

    if (worst > tol) {
        std::print("FAIL: max relative error {} exceeds tolerance {}\n", worst, tol);
        return 1;
    }
    std::print("PASS: max relative error {} within tolerance {}\n", worst, tol);
    return 0;
}
