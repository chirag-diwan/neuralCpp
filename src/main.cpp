#ifndef MNIST_TEST
#ifndef XOR_TEST
#include "./include/Weights.h"
#include "./include/Model.h"
#include "./include/Dataset.h"
#include "include/MatMaths.h"
#include <raylib.h>

thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(128*1024*1024);

void Infer(NeuralNetwork& model, Mat& inputData, Mat& outputData, size_t test_count) {
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

  IDX3 testImages = readImage("/home/chirag/datasets/t10k-images.idx3-ubyte");
  IDX1 testLabels = readImageLabels("/home/chirag/datasets/t10k-labels.idx1-ubyte");

  NeuralNetwork model = NNLoadModel("Mnist-128-10.bin");



  {
    DeferFree df;

    Mat testInputData;
    testInputData.Populate(1, testImages.data.size(), false);
    testInputData.Cpy(testImages.data.data(), testImages.data.size());

    const auto scale = 1.0f/255.0f;
    MatScale(testInputData, scale);

    Mat testOutputData;
    testOutputData.Populate(1, testLabels.labels.size() * 10, false);
    for(size_t i = 0; i < testLabels.labels.size(); i++) {
      for(int j = 0; j < 10; j++) {
        testOutputData.data[(i * 10) + j] = (j == testLabels.labels[i]) ? 1.0f : 0.0f;
      }
    }

    Infer(model , testInputData , testOutputData ,100);
    std::cout << " Total Memory used to test " << __Global_Mat_Allocator->GetStrider() * sizeof(float)/1e6 << " mb .\n";
  }

  delete  __Global_Mat_Allocator;
  return 0;
}
#endif
#endif
