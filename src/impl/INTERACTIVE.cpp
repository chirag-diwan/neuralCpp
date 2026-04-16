#ifdef INTERACTIVE_INFERENCE
#include <cstdint>
#include <cstring>
#include <iostream>
#include <raylib.h>
#include <vector>
#include "../include/Model.h"
#include "../include/Weights.h"


thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(1024*1024);

float Infer(NeuralNetwork& model, Mat& input) {
  Forward(model, input);
  auto predicted_indx = ArgMax(model.layers[model.layers.size() - 1].A);
  return predicted_indx;
}

class InteractiveInferEngine{
  private:
    uint32_t width = 800;
    uint32_t height = 600;

    uint32_t tile_size = 20;

    NeuralNetwork& model;

    std::vector<uint8_t> grid;

    uint32_t rows;
    uint32_t cols;

  public:
    InteractiveInferEngine(NeuralNetwork& model) : model(model){}

    void Init(uint32_t rows , uint32_t cols){
      this->rows = rows;
      this->cols = cols;


      grid.resize(rows*cols);

      tile_size = 600/rows;

      InitWindow(width, height, "Interactive Inference");
      SetTargetFPS(60);
    }

    void Run(){
      bool Recording = false;
      while (!WindowShouldClose()) {
        if(IsKeyPressed(KEY_ENTER)){
          {
            DeferFree df;
            Mat input;
            input.Populate(1, 28*28, false);
            input.Cpy(grid.data() , 28*28);

            const auto scale = 1.0f / 255.0f;
            MatScale(input, scale);

            std::cout << "Prediction: " << Infer(model, input) << '\n';
          }
        }

        Recording = false;
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
          Recording = true;
        }

        if(IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)){
          memset(grid.data(), 0, grid.size());
        }

        if(Recording){
          Vector2 mousepos = GetMousePosition();
          int gridX = static_cast<int>(mousepos.x) / tile_size;
          int gridY = static_cast<int>(mousepos.y) / tile_size;

          // 4. Bounds Checking
          if (gridX >= 0 && gridX < cols && gridY >= 0 && gridY < rows) {

            // Core brush stroke
            grid[gridY * cols + gridX] = 255;

            // 3. Fake Anti-Aliasing / Brush Thickening to match MNIST domain
            // Apply half-intensity to adjacent pixels if they are within bounds
            if (gridX + 1 < cols && grid[gridY * cols + (gridX + 1)] == 0) 
              grid[gridY * cols + (gridX + 1)] = 128;
            if (gridX - 1 >= 0 && grid[gridY * cols + (gridX - 1)] == 0) 
              grid[gridY * cols + (gridX - 1)] = 128;
            if (gridY + 1 < rows && grid[(gridY + 1) * cols + gridX] == 0) 
              grid[(gridY + 1) * cols + gridX] = 128;
            if (gridY - 1 >= 0 && grid[(gridY - 1) * cols + gridX] == 0) 
              grid[(gridY - 1) * cols + gridX] = 128;
          }
        }

        BeginDrawing();
        ClearBackground(BLACK); // Ensure background clears properly
        for(int i = 0 ; i < grid.size() ; i++){
          uint32_t x = i % cols;
          uint32_t y = i / cols;
          uint8_t color = grid[i];

          // Draw the grid cells
          DrawRectangle(x * tile_size, y * tile_size, tile_size, tile_size, {color, color, color, 255});
        }
        EndDrawing();
      }
      CloseWindow();
    }
};

int main(){
  NeuralNetwork model  = NNLoadModel("./Mnist-128-10.bin" , 1);
  InteractiveInferEngine engine(model);
  engine.Init(28 , 28);
  engine.Run();
  delete __Global_Mat_Allocator;
}
#endif
