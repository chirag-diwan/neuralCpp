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


uint32_t readBE32(std::ifstream &in) {
  uint8_t bytes[4];
  in.read(reinterpret_cast<char*>(bytes), 4);

  return (uint32_t(bytes[0]) << 24) |
    (uint32_t(bytes[1]) << 16) |
    (uint32_t(bytes[2]) << 8)  |
    (uint32_t(bytes[3]));
}

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



IDX3 readImage(const char* filepath){
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



IDX1 readImageLabels(const char* filepath){
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




uint8_t getPixel(const IDX3 &img, int imageIndex, int row, int col) {
  size_t offset =
    imageIndex * img.rows * img.cols +
    row * img.cols +
    col;

  return img.data[offset];
}

// NIGHTLY

namespace Dataset{
  void Display(const char* imagesPath, const char* labelsPath) {
    IDX3 images = readImage(imagesPath);
    IDX1 labels = readImageLabels(labelsPath);
    std::cout << " --- Read Complete --- \n";

    const auto rows = images.rows;
    const auto cols = images.cols;

    uint64_t imageIndex = 0;
    const int scale = 20; // Extracted magic number to prevent arbitrary hardcoding

    InitWindow(800, 600, "Dataset Viewer");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
      if (IsKeyPressed(KEY_LEFT) && imageIndex > 0) {
        imageIndex--;
      } else if (IsKeyPressed(KEY_RIGHT) && imageIndex < images.num_images - 1) { // Prevent out of bounds
        imageIndex++;
      }

      BeginDrawing();
      ClearBackground(RAYWHITE); 

      for(auto i = 0 ; i < rows ; i++){
        for(auto j = 0 ; j < cols ; j++){
          auto pixelIndex = imageIndex * rows * cols + i * cols + j;
          unsigned char brightness = images.data[pixelIndex];

          DrawRectangle(j * scale, i * scale, scale, scale, {brightness, brightness, brightness, 255});
        }
      }

      int currentLabel = labels.labels[imageIndex];

      int textX = (cols * scale) + 20; 
      DrawText(TextFormat("Label: %d", currentLabel), textX, 40, 40, DARKGRAY);
      DrawText(TextFormat("Index: %llu", imageIndex), textX, 90, 20, GRAY);

      EndDrawing();
    }
    CloseWindow();
  }
};


// NIGHTLY

