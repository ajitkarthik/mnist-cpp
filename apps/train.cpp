#include <assert.h>
#include <sys/types.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string_view>
#include <tensor/tensor.hpp>

using namespace tn;

inline constexpr std::string_view TRAIN_IMAGE_FILENAME = "../data/train-images.idx3-ubyte";
inline constexpr std::string_view TRAIN_LABEL_FILENAME = "../data/train-labels.idx1-ubyte";
inline constexpr std::string_view TEST_IMAGE_FILENAME = "../data/t10k-images.idx3-ubyte";
inline constexpr std::string_view TEST_LABEL_FILENAME = "../data/t10k-labels.idx1-ubyte";
inline constexpr int NUM_CLASSES = 10;

template <std::integral T>
[[nodiscard]] constexpr T convert_endian(T value) {
    if constexpr (std::endian::native == std::endian::big) {
        return value;
    } else {
        return std::byteswap(value);
    }
}

int read_images(std::string_view filename, std::vector<std::vector<uint8_t>>& images, int& rows,
                int& cols) {
    // Big endian format
    // Image Files(idx3 - ubyte)
    //   The first 16 bytes contain the file metadata
    //   Bytes 0–3 : Magic number(2051)
    //   Bytes 4–7 : number of images
    //   Bytes 8–11: number of rows(28)
    //   Bytes 12–15 : number of columns(28)
    //   Remaining bytes : Pixel intensities from 0 to 255.
    uint32_t magic = 0;
    uint32_t num_images = 0;

    std::ifstream file(std::filesystem::path(filename), std::ios::binary | std::ios::beg);
    std::cout << "Reading image file ...";
    // magic number
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    magic = convert_endian(magic);
    assert(magic == 0x803);
    // number of images
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = convert_endian(num_images);
    // number of rows
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    rows = convert_endian(rows);
    // number of cols
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    cols = convert_endian(cols);

    std::vector<uint8_t> image;

    image.resize(rows * cols);
    while (file.read(reinterpret_cast<char*>(image.data()), image.size())) {
        images.push_back(image);
    }
    std::cout << " done" << std::endl;
    std::cout << "Number of images: " << images.size() << std::endl;

    file.close();
    return images.size();
}

int read_labels(std::string_view filename, std::vector<uint8_t>& labels) {
    // Big endian format
    // Label Files(idx1 - ubyte)
    //   The first 8 bytes contain the metadata.
    //   Bytes 0–3 : Magic number(2049)
    //   Bytes 4–7 : number of items
    //   Remaining bytes : Unsigned byte digits from 0 to 9.
    uint32_t magic = 0;
    uint32_t num_labels = 0;

    std::ifstream file(std::filesystem::path(filename), std::ios::binary | std::ios::beg);
    std::cout << "Reading label file ...";

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    magic = convert_endian(magic);
    assert(magic == 0x801);
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = convert_endian(num_labels);
    std::cout << " done" << std::endl;

    uint8_t label = 0;

    while (file.read(reinterpret_cast<char*>(&label), sizeof(label))) {
        labels.push_back(label);
    }
    std::cout << "Number of labels: " << labels.size() << std::endl;

    file.close();
    return labels.size();
}

// The caller owns the generator so the draw sequence is reproducible across runs
// (and so we don't pay to re-seed a Mersenne twister on every call).
void assemble_minibatch(Tensor& X, Tensor& y, std::vector<std::vector<uint8_t>>& images,
                        std::vector<uint8_t>& labels, int batchsize, std::mt19937& g) {
    std::uniform_int_distribution<int> distrib(0, static_cast<int>(images.size()) - 1);

    for (int b = 0; b < batchsize; b++) {
        int idx = distrib(g);
        const std::vector<uint8_t>& img = images[idx];
        for (int p = 0; p < X.cols(); p++) {
            X.set(b, p, img[p] / 255.0f);
        }
        y.set(b, 0, static_cast<float>(labels[idx]));
    }
}

int main() {
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;
    int rows = 0;
    int cols = 0;

    [[maybe_unused]] int num_labels = read_labels(TRAIN_LABEL_FILENAME, labels);
    [[maybe_unused]] int num_images = read_images(TRAIN_IMAGE_FILENAME, images, rows, cols);
    assert(num_labels == num_images);

    // MLP
    constexpr int N_HIDDEN = 128;
    // Fix the weight-init stream so the whole run is reproducible.
    constexpr uint32_t INIT_SEED = 2024;
    Tensor::seed(INIT_SEED);
    Tensor W1 = Tensor::randn(rows * cols, N_HIDDEN);
    Tensor b1 = Tensor::zeros(1, N_HIDDEN);
    Tensor W2 = Tensor::randn(N_HIDDEN, N_HIDDEN);
    Tensor b2 = Tensor::zeros(1, N_HIDDEN);
    Tensor W3 = Tensor::randn(N_HIDDEN, NUM_CLASSES);
    Tensor b3 = Tensor::zeros(1, NUM_CLASSES);

    std::vector<Tensor> params;
    params.push_back(W1);
    params.push_back(b1);
    params.push_back(W2);
    params.push_back(b2);
    params.push_back(W3);
    params.push_back(b3);

    // Kaiming init for W1 (since it feeds RELU)
    for (int i = 0; i < W1.rows(); i++) {
        for (int j = 0; j < W1.cols(); j++) {
            W1.set(i, j, W1.at(i, j) * sqrt(2.0f / (rows * cols)));
        }
    }

    // Kaiming init for W2 (since it feeds RELU)
    for (int i = 0; i < W2.rows(); i++) {
        for (int j = 0; j < W2.cols(); j++) {
            W2.set(i, j, W2.at(i, j) * sqrt(2.0f / N_HIDDEN));
        }
    }

    // Xavier init for W3 (since it feeds softmax)
    for (uint32_t i = 0; i < N_HIDDEN; i++) {
        for (uint32_t j = 0; j < NUM_CLASSES; j++) {
            W3.set(i, j, W3.at(i, j) * (sqrt(1.0f / (N_HIDDEN))));
        }
    }

    // Training
    constexpr int TRAINING_ITERATIONS = 20000;
    constexpr int TRAINING_ITERATIONS_LR_DECAY = 10000;
    constexpr int BATCHSIZE = 32;
    constexpr float LEARNING_RATE_SLOW = 0.01;
    constexpr float LEARNING_RATE_FAST = 0.1;
    const int features = static_cast<int>(rows * cols);  // 784

    // Fixed seeds so runs are reproducible and two configs can be compared
    // without mini-batch order accounting for the difference.
    constexpr uint32_t TRAIN_SEED = 1337;
    constexpr uint32_t TEST_SEED = 4242;
    std::mt19937 train_gen(TRAIN_SEED);

    // Training loop
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < TRAINING_ITERATIONS; i++) {
        // Assemble a mini-batch of BATCHSIZE random samples.
        Tensor X(BATCHSIZE, features);  // (BATCHSIZE, 784), pixels normalized to [0, 1]
        Tensor y(BATCHSIZE, 1);         // matching labels for the batch

        assemble_minibatch(X, y, images, labels, BATCHSIZE, train_gen);

        Tensor loss = X.matmul(W1)
                          .add_bias(b1)
                          .relu()
                          .matmul(W2)
                          .add_bias(b2)
                          .relu()
                          .matmul(W3)
                          .add_bias(b3)
                          .fused_cross_entropy_loss(y);

        if (!(i % 1000)) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start);
            std::cout << "[" << duration.count() << "s]"
                      << "Iteration:" << i << " Loss:" << loss;
        }

        // gradient descent
        loss.backward();

        // adjust weights for parameters
        for (auto& p : params) {
            if (i <= TRAINING_ITERATIONS_LR_DECAY)
                p.adjust_weights(LEARNING_RATE_FAST);
            else
                p.adjust_weights(LEARNING_RATE_SLOW);
        }

        // set gradients of params to 0 for next pass
        for (auto& p : params) {
            p.zero_grad();
        }
    }

    // Testing ...
    std::cout << "Running testing ...\n";
    labels.clear();
    images.clear();
    num_labels = read_labels(TEST_LABEL_FILENAME, labels);
    num_images = read_images(TEST_IMAGE_FILENAME, images, rows, cols);
    assert(num_labels == num_images);
    // Training
    constexpr int TEST_ITERATIONS = 300;
    int correct = 0;
    int incorrect = 0;
    // Seeded separately from training so the eval sample is identical across
    // runs regardless of how many training draws preceded it.
    std::mt19937 test_gen(TEST_SEED);

    // Testing loop
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        // Assemble a mini-batch of BATCHSIZE random samples.
        Tensor X(BATCHSIZE, features);  // (BATCHSIZE, 784), pixels normalized to [0, 1]
        Tensor y(BATCHSIZE, 1);         // matching labels for the batch

        assemble_minibatch(X, y, images, labels, BATCHSIZE, test_gen);

        Tensor probs = X.matmul(W1)
                           .add_bias(b1)
                           .relu()
                           .matmul(W2)
                           .add_bias(b2)
                           .relu()
                           .matmul(W3)
                           .add_bias(b3)
                           .softmax();

        for (int i = 0; i < BATCHSIZE; i++) {
            if (probs.max_idx(i) == y.at(i, 0)) {
                correct++;
            } else {
                incorrect++;
            }
        }
    }

    std::cout << "Correct: " << correct << "\n";
    std::cout << "Incorrect: " << incorrect << "\n";
    std::cout << "% correct: " << correct * 1.0 / (correct + incorrect) << "\n";

    return 0;
}