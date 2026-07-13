#include <assert.h>

#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string_view>
#include <tensor/tensor.hpp>

using namespace tn;

inline constexpr std::string_view TRAIN_IMAGE_FILENAME =
    "../data/train-images.idx3-ubyte";
inline constexpr std::string_view TRAIN_LABEL_FILENAME =
    "../data/train-labels.idx1-ubyte";

template <std::integral T>
[[nodiscard]] constexpr T convert_endian(T value) {
    if constexpr (std::endian::native == std::endian::big) {
        return value;
    } else {
        return std::byteswap(value);
    }
}

int main() {
    // Big endian format
    // Image Files(idx3 - ubyte)
    //   The first 16 bytes contain the file metadata
    //   Bytes 0–3 : Magic number(2051)
    //   Bytes 4–7 : number of images
    //   Bytes 8–11: number of rows(28)
    //   Bytes 12–15 : number of columns(28)
    //   Remaining bytes : Pixel intensities from 0 to 255.
    // Label Files(idx1 - ubyte)
    //   The first 8 bytes contain the metadata.
    //   Bytes 0–3 : Magic number(2049)
    //   Bytes 4–7 : number of items
    //   Remaining bytes : Unsigned byte digits from 0 to 9.
    uint32_t magic = 0;
    uint32_t num_images = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t num_labels = 0;

    std::ifstream file(std::filesystem::path(TRAIN_IMAGE_FILENAME),
                       std::ios::binary | std::ios::beg);
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

    // create a vector for the images, each vector contains rows * cols uint8s
    std::vector<uint8_t> image;
    std::vector<std::vector<uint8_t>> images;

    image.resize(rows * cols);
    while (file.read(reinterpret_cast<char*>(image.data()), image.size())) {
        images.push_back(image);
    }
    std::cout << " done" << std::endl;
    std::cout << "Number of images: " << images.size() << std::endl;

    file.close();

    std::cout << "Reading label file ...";
    file.open(std::filesystem::path(TRAIN_LABEL_FILENAME),
              std::ios::binary | std::ios::beg);
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    magic = convert_endian(magic);
    assert(magic == 0x801);
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = convert_endian(num_labels);
    assert(num_labels == num_images);
    std::cout << " done" << std::endl;

    // create a vector for the labels, each vector contains num_labels uint8s
    std::vector<uint8_t> labels;
    uint8_t label = 0;

    while (file.read(reinterpret_cast<char*>(&label), sizeof(label))) {
        labels.push_back(label);
    }
    std::cout << "Number of labels: " << labels.size() << std::endl;

    file.close();

    return 0;
}