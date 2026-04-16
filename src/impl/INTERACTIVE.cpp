#ifdef INTERACTIVE_INFERENCE
#include <cstdint>
#include <cstring>
#include <iostream>
#include <raylib.h>
#include <vector>
#include "../include/Model.h"
#include "../include/Weights.h"


inline float Activation(float a){
  return Sigmoid(a);
}

inline float ActivationPrime(float a){
  return SigmoidPrime(a) ;
}

inline void Activation(Mat& src , Mat& dst){
  assert(src.rows == dst.rows && src.cols == dst.cols);
  for(size_t i = 0 ; i < src.rows*src.cols ; i++){
    dst[i] = Activation(src[i]);
  }
}

inline void ActivationPrime(Mat& src , Mat& dst){
  assert(src.rows == dst.rows && src.cols == dst.cols);
  for(size_t i = 0 ; i < src.rows*src.cols ; i++){
    dst[i] = ActivationPrime(src[i]);
  }
}



thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(1024*1024);


float Infer(NeuralNetwork& model, Mat& inputData) {
  constexpr auto img_size = 28 * 28;
  const size_t BATCH_SIZE = model.batchsize;

    DeferFree df;
    Mat input;
    input.ViewNoAlloc(BATCH_SIZE, img_size, inputData.data);
    Forward(model, input);

    auto& final_layer_A = model.layers[model.layers.size() - 1].A;
    PrintMat("Final A", final_layer_A, true);
    Mat view;
    view.ViewNoAlloc(1,final_layer_A.cols,final_layer_A.data);
    return ArgMax(view);
}


class InteractiveInferEngine{
  private:
    uint32_t tile_size = 20;

    uint32_t batchsize = 0;

    NeuralNetwork& model;

    std::vector<uint8_t> grid;

    uint32_t rows;
    uint32_t cols;

  public:
    InteractiveInferEngine(NeuralNetwork& model) : model(model){}

    void Init(uint32_t rows , uint32_t cols , uint32_t batchsize){
      this->rows = rows;
      this->cols = cols;
      this->batchsize = batchsize;


      grid.resize(batchsize*rows*cols);

      tile_size = 600/rows;

      InitWindow(cols*tile_size, rows*tile_size, "Interactive Inference");
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

          
          if (gridX >= 0 && gridX < cols && gridY >= 0 && gridY < rows) {

            
            grid[gridY * cols + gridX] = 255;

            
            
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
        ClearBackground(BLACK); 
        for(int i = 0 ; i < grid.size() ; i++){
          uint32_t x = i % cols;
          uint32_t y = i / cols;
          uint8_t color = grid[i];
      
          if(x < 4 || x > 23 || y < 4 || y > 23){

            DrawRectangle(x * tile_size, y * tile_size, tile_size, tile_size, RED);
          }else{
            DrawRectangle(x * tile_size, y * tile_size, tile_size, tile_size, {color, color, color, 255});
          }
        }
        EndDrawing();
      }
      CloseWindow();
    }
};

int main(){
  NeuralNetwork model  = NNLoadModel("./MNIST-TEST.bin");
  InteractiveInferEngine engine(model);
  engine.Init(28 , 28 , model.batchsize);
  engine.Run();
  delete __Global_Mat_Allocator;
}
#endif
