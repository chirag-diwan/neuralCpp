#include "../include/Dataset.h"

uint32_t readBE32(std::ifstream &in) {
  uint8_t bytes[4];
  in.read(reinterpret_cast<char*>(bytes), 4);

  return (uint32_t(bytes[0]) << 24) |
    (uint32_t(bytes[1]) << 16) |
    (uint32_t(bytes[2]) << 8)  |
    (uint32_t(bytes[3]));
}

IDX1 readImageLabels(std::string filepath){
  std::ifstream in(filepath, std::ios::binary);
  if(!in.is_open()){
    ERROR_AND_EXIT("Error opening file " , filepath);
  }

  uint32_t magic = readBE32(in);

  if (magic != 0x00000801){
    ERROR_AND_EXIT("Invalid IDX1 magic number");
  }

  IDX1 data;
  data.count = readBE32(in);
  data.labels.resize(data.count);
  in.read(reinterpret_cast<char*>(data.labels.data()), data.count);

  in.close();
  return data;
}

IDX3 readImage(std::string filepath){
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()){ ERROR_AND_EXIT("Cannot open file");}

  uint32_t magic = readBE32(file);
  if (magic != 0x00000803){
    ERROR_AND_EXIT("Invalid IDX3 magic number");
  }
  IDX3 image;
  image.num_images = readBE32(file);
  image.rows       = readBE32(file);
  image.cols       = readBE32(file);

  size_t total = size_t(image.num_images) * image.rows * image.cols;

  image.data.resize(total);
  file.read(reinterpret_cast<char*>(image.data.data()), total);
  return image;
}

uint8_t getPixel(const IDX3 &img, int imageIndex, int row, int col) {
  size_t offset =
    imageIndex * img.rows * img.cols +
    row * img.cols +
    col;

  return img.data[offset];
}

// NIGHTLY


