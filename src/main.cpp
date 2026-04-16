#ifndef MNIST_TEST
#ifndef XOR_TEST
#ifndef INTERACTIVE_INFERENCE

#include "./include/Model.h"
#include "include/Dataset.h"
#include "include/MatMaths.h"
#include "include/Weights.h"
#include <cassert>


thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(128*1024*1024);

void Infer(NeuralNetwork& model, Mat& inputData, Mat& outputData, size_t test_count) {
  constexpr auto img_size = 28 * 28;

  const size_t BATCH_SIZE = model.batchsize;

  std::cout << "--- Initiating Inference ---\n";
  size_t correct_predictions = 0;
  size_t processed_count = 0;

  for (size_t batch_index = 0; batch_index < test_count; batch_index += BATCH_SIZE) {
    DeferFree df;
    size_t current_batch_size = std::min(BATCH_SIZE, test_count - batch_index);
    Mat input;
    input.ViewNoAlloc(BATCH_SIZE, img_size, inputData.data + (batch_index * img_size));

    Mat output;
    output.ViewNoAlloc(BATCH_SIZE, 10, outputData.data + (batch_index * 10));

    Forward(model, input);

    auto& final_layer_A = model.layers[model.layers.size() - 1].A;

    for (size_t b = 0; b < current_batch_size; b++) {
      float max_pred_val = -1e9f;
      size_t pred_idx = 0;

      float max_target_val = -1e9f;
      size_t target_idx = 0;

      for (size_t c = 0; c < 10; c++) {
        float p_val = final_layer_A.data[b * 10 + c];
        if (p_val > max_pred_val) {
          max_pred_val = p_val;
          pred_idx = c;
        }

        float t_val = output.data[b * 10 + c];
        if (t_val > max_target_val) {
          max_target_val = t_val;
          target_idx = c;
        }
      }

      if (pred_idx == target_idx) {
        correct_predictions++;
      }
    }
    processed_count += current_batch_size;
  }

  float accuracy = (static_cast<float>(correct_predictions) / processed_count) * 100.0f;
  std::cout << "Test accuracy: " << std::fixed << std::setprecision(2) << accuracy 
    << "% (" << correct_predictions << "/" << processed_count << ")\n";
}

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


int main(){

  IDX3 testImages = readImage("/home/chirag/datasets/train-images.idx3-ubyte");
  IDX1 testLabels = readImageLabels("/home/chirag/datasets/train-labels.idx1-ubyte");

  //const auto img_rows = testImages.rows;
  //const auto img_cols = testImages.cols;
  //const auto img_size = img_rows*img_cols;

  //const auto BATCH_SIZE = 20;

  NeuralNetwork model = NNLoadModel("./MNIST-BATCH-20-128-10.bin");

  //{
  //  DeferFree df;
  //
  //  Mat testInputData;
  //  testInputData.Populate(1, testImages.data.size(), false);
  //  testInputData.Cpy(testImages.data.data(), testImages.data.size());
  //
  //  const auto scale = 1.0f/255.0f;
  //  MatScale(testInputData, scale);
  //
  //  Mat testOutputData;
  //  testOutputData.Populate(1, testLabels.labels.size() * 10, false);
  //  for(size_t i = 0; i < testLabels.labels.size(); i++) {
  //    for(int j = 0; j < 10; j++) {
  //      testOutputData.data[(i * 10) + j] = (j == testLabels.labels[i]) ? 1.0f : 0.0f;
  //    }
  //  }
  //
  //  constexpr uint32_t epoch = 400;
  //  constexpr uint32_t train_count = 10000;
  //  constexpr float learning_rate = 1e-2;
  //
  //  for(int e = 0 ; e < epoch ; e++){
  //    float epoch_cost = 0.0f;
  //    for(int batch_index = 0 ; batch_index < train_count ; batch_index += BATCH_SIZE){
  //      DeferFree df;
  //      Mat input;
  //      input.Populate(BATCH_SIZE,img_size ,false);
  //      auto index = batch_index*img_size;
  //      input.Cpy(&testInputData.data[index] , BATCH_SIZE*img_size);
  //
  //      Mat output;
  //      output.Populate(BATCH_SIZE, 10, false);
  //      index = batch_index*10;
  //      output.Cpy(&testOutputData.data[index] , BATCH_SIZE*10);
  //      epoch_cost += Cost(model, input, output);
  //      BackProp(model,input, output,  learning_rate , BATCH_SIZE);
  //    }
  //    std::cout << "Epoch Cost :: " << epoch_cost/train_count << "\n";
  //  }
  //  std::cout << " Memory Usage :: " << __Global_Mat_Allocator->GetStrider()*sizeof(float)/1e6 << "mb\n";
  //}
  //
  //SaveModel(model,28*28,"MNIST-BATCH-20-128-10.bin");

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

  delete __Global_Mat_Allocator;
}
#endif
#endif
#endif
