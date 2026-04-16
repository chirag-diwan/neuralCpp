// I want to see how to weights changes with respect to the input that was provided , based on that we can further see what is the thing that makes the network learn and is there is any pattern

#include "../include/Model.h"
#include "../include/Dataset.h"
#include "../include/MatMaths.h"
#include "../include/Weights.h"
#include <cassert>


thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(128*1024*1024);

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

  
  delete __Global_Mat_Allocator;
}
