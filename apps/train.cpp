#include <tensor/tensor.hpp>
#include <iostream>

using namespace tn;

int main() {
    Tensor t = Tensor::randn(5, 2);
    std::cout << t << std::endl;
    return 0;
}