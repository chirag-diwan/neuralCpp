#ifdef XOR_TEST
#include "../include/MatMaths.h"
#include "../include/Model.h"
#include <array>
#include <iostream>

int main() {
  NeuralNetwork<2, 3> model({2, 3, 1});
  model.Init();
  
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
      // 4. Zero-allocation data transfer. Direct hardware-level memory write.
      input[0] = inputData[i][0];
      input[1] = inputData[i][1];
      output[0] = outputData[i][0];

      // Cost and BackProp handle their own temporary memory via DeferFree.
      epochCost += Cost(model, input, output);
      BackProp(model, input, output, learningRate);
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
    
    // 5. You forgot to scope your final inference pass. 
    // Without this DeferFree, your final predictions will leak memory into the global allocator.
    {
      DeferFree df; 
      Forward(model, input);
    }
    
    float prediction = model.layers[model.layers.size() - 1].A[0];
    
    std::cout << "Input: [" << inputData[i][0] << ", " << inputData[i][1] 
              << "] -> Target: " << outputData[i][0] 
              << " | Prediction: " << std::setprecision(4) << prediction << "\n";
  }

  std::cout << "Total memory usage " << __Global_Mat_Allocator->GetStrider() << "units \n";
  delete __Global_Mat_Allocator;
  return 0;
}
#endif

