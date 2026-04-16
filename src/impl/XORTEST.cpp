#ifdef XOR_TEST
#include "../include/MatMaths.h"
#include "../include/Model.h"
#include <iostream>



inline void Activation(Mat& src , Mat& dst){
  assert(src.rows == dst.rows && src.cols == dst.cols);
  for(size_t i = 0 ; i < src.rows*src.cols ; i++){
    dst[i] = Sigmoid(src[i]);
  }
}

inline void ActivationPrime(Mat& src , Mat& dst){
  assert(src.rows == dst.rows && src.cols == dst.cols);
  for(size_t i = 0 ; i < src.rows*src.cols ; i++){
    dst[i] = SigmoidPrime(src[i]);
  }
}


thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(64);

int main() {
  ActivationProfile Profile = {
    .Activation = Activation,
    .ActivationPrime = ActivationPrime,
  };

  NeuralNetwork model({2, 3, 1});
  model.Init(2 , 1 , Profile);
  
  Mat input;
  input.Populate(1, 2, false); 

  Mat output;
  output.Populate(1, 1, false);

  const float inputData[4][2] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
  const float outputData[4][1] = { {0}, {1}, {1}, {0} };

  float learningRate = 0.5f; 
  int epochs = 20000;        

  std::cout << "--- Initiating Training ---\n";

  for (int epoch = 0; epoch < epochs; epoch++) {
    float epochCost = 0.0f;

    for (size_t i = 0; i < 4; i++) {
      
      input[0] = inputData[i][0];
      input[1] = inputData[i][1];
      output[0] = outputData[i][0];

      
      epochCost += Cost(model, input, output);
      BackProp(model, input, output, learningRate , 1);
    }

    if (epoch % 2000 == 0) {
      std::cout << "Epoch: " << std::setw(5) << epoch 
                << " | Average Cost: " << std::setprecision(6) << (epochCost / 4.0f) 
                << "\n";
    }
  }

  std::cout << "\n--- Training Complete. Final Predictions ---\n";
  for (size_t i = 0; i < 4; i++) {
    input[0] = inputData[i][0];
    input[1] = inputData[i][1];
    
    
    
    {
      DeferFree df; 
      Forward(model, input);
    }
    
    float prediction = model.layers[model.layers.size() - 1].A[0];
    
    std::cout << "Input: [" << inputData[i][0] << ", " << inputData[i][1] 
              << "] -> Target: " << outputData[i][0] 
              << " | Prediction: " << std::setprecision(4) << prediction << "\n";
  }

  std::cout << " Total Memory used " << __Global_Mat_Allocator->GetStrider() * sizeof(float)/1e6 << " mb .\n";
  delete  __Global_Mat_Allocator;
  return 0;
}
#endif

