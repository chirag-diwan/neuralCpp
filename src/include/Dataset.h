#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <strings.h>
#include <vector>
#include <raylib.h>
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



IDX3 readImage(const char* filepath);
IDX1 readImageLabels(const char* filepath);
uint8_t getPixel(const IDX3 &img, int imageIndex, int row, int col) ;

// NIGHTLY

namespace Dataset{
  void Display(const char* imagesPath, const char* labelsPath) ;
};


// NIGHTLY

