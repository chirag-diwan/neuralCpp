#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "../Utils/Utils.h"
#include "./MatMaths.h"

uint32_t readBE32(std::ifstream &in) ;

typedef struct {
  uint32_t magic;
  uint32_t count;
  std::vector<uint8_t> labels;
} IDX1;

typedef struct {
  uint32_t magic;
  uint32_t num_images;
  uint32_t rows;
  uint32_t cols;
  std::vector<uint8_t>data;
} IDX3;


IDX1 readImageLabels(std::string filepath);

IDX3 readImage(std::string filepath);

uint8_t getPixel(const IDX3 &img, int imageIndex, int row, int col) ;

// NIGHTLY

namespace Dataset{
  enum SplitDir{
    ROW,
    COL
  };
};
