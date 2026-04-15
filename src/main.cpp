#include "include/Dataset.h"
#include "include/MatMaths.h"
#include "include/Model.h"
#include <iostream>
#include <iomanip>

thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(256*1024*1024);

template <size_t inputParamCount, size_t layerCount>
void Infer(NeuralNetwork<inputParamCount, layerCount>& model, Mat& inputData, Mat& outputData, size_t test_count) {
  constexpr auto img_size = 28 * 28;

  std::cout << "--- Initiating Inference ---\n";

  size_t correct_predictions = 0;

  for(size_t j = 0; j < test_count; j++) {
    
    DeferFree df;

    Mat input;
    input.ViewNoAlloc(1, img_size, inputData.data + (j * img_size));

    Mat output;
    output.ViewNoAlloc(1, 10, outputData.data + (j * 10));

    Forward(model, input);

    auto predicted_indx = ArgMax(model.layers[model.layers.size() - 1].A);
    
    auto target_indx = ArgMax(output);

    if (predicted_indx == target_indx) {
      correct_predictions++;
    }
  }

  float accuracy = (static_cast<float>(correct_predictions) / test_count) * 100.0f;
  std::cout << "Test accuracy: " << std::fixed << std::setprecision(2) << accuracy << "% \n";
}


int main(){
  IDX3 images = readImage("/home/chirag/datasets/train-images.idx3-ubyte");
  IDX1 labels = readImageLabels("/home/chirag/datasets/train-labels.idx1-ubyte");
  constexpr auto rows = 28; 
  constexpr auto cols = 28;
  constexpr auto img_size = rows * cols; // 784

  NeuralNetwork<img_size, 2> model({128, 10});
  model.Init();

  // Load the images and labels into memory
  Mat inputData;
  inputData.Populate(1, images.data.size(), false);
  inputData.Cpy(images.data.data(), images.data.size());

  // Normalize the images 
  const auto scale = 1.0f / 255.0f;
  MatScale(inputData, scale);

  Mat outputData;
  outputData.Populate(1, labels.labels.size() * 10, false);
  for(size_t i = 0; i < labels.labels.size(); i++) {
    for(int j = 0; j < 10; j++) {
      outputData.data[(i * 10) + j] = (j == labels.labels[i]) ? 1.0f : 0.0f;
    }
  }

  const auto epochs = 50; 
  const auto learning_rate = 1e-3;
  const auto train_count = 10000; // 10K

  std::cout << "--- Initiating Training ---\n";

  for(size_t i = 0; i < epochs; i++) {
    float epoch_cost = 0;

    for(auto j = 0; j < train_count; j++) {

      Mat input;
      input.ViewNoAlloc( 1, img_size , inputData.data + (j * img_size));

      Mat output;
      output.ViewNoAlloc(1, 10 , outputData.data + (j * 10));

      epoch_cost += Cost(model, input, output);

      BackProp(model, input, output, learning_rate);
    }
    std::cout << "Epoch Cost: " << std::setw(3) <<(epoch_cost / train_count) << "\n";
  }

  Infer(model,inputData,outputData ,100);

  std::cout << " Total Memory used " << __Global_Mat_Allocator->GetStrider() * sizeof(float)/1e6 << " mb .\n";
  delete  __Global_Mat_Allocator;
  return 0;
}
