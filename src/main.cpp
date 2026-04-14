#include "include/MatMaths.h"
#include "include/Model.h"
#include "include/Dataset.h"
#include <array>
#include <cmath>
#include <iostream>
#include <openblas/cblas.h>

thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(256*1024*1024);

template <size_t inputParamCount, size_t layerCount>
void Infer(Model<inputParamCount, layerCount>& model , size_t testcount) {
  IDX3 testimgs = readImage("/home/chirag/Learn/NeuralNetwork/dataset/t10k-images.idx3-ubyte");
  IDX1 testlabels = readImageLabels("/home/chirag/Learn/NeuralNetwork/dataset/t10k-labels.idx1-ubyte");

  Matrix testinput;
  testinput.Populate(1, 28 * 28, false); 

  const auto inputStrider = 28 * 28;
  
  // Set to 20 for your current test constraints, expand to testlabels.labels.size() later
  size_t correct_predictions = 0;

  const auto alpha = 1.0f/255.0f;

  for (size_t i = 0; i < testcount; i++) {
      testinput.Cpy(testimgs.data.data() + i * inputStrider, inputStrider);
      cblas_sscal(testinput.rows*testinput.cols,alpha ,testinput.data,1);
      {
        // DeferFree scopes the lifetime of temporary matrices spawned during Forward()
        DeferFree df; 
        Forward(model, testinput);
        
        // 1. Extract discrete prediction from the final layer
        auto& final_layer_A = model.layers[model.layers.size() - 1].A;
        size_t predicted_class = ArgMax(final_layer_A);
        
        // 2. Read raw integer ground truth directly from the dataset
        size_t target_class = testlabels.labels[i];
        
        // 3. Binary state evaluation
        if (predicted_class == target_class) {
            correct_predictions++;
        }
      }
  }
  
  float accuracy = (static_cast<float>(correct_predictions) / testcount) * 100.0f;
  std::cout << "Accuracy: " << accuracy << "% (" << correct_predictions << "/" << testcount << ")\n";
}

#ifdef XOR_TEST
int main() {
  Model<2, 3> model({2, 3, 1});
  model.Init();
  
  Matrix input;
  input.Populate(1, 2, false); 

  Matrix output;
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

  std::cout << "Total memory usage";
  delete __Global_Mat_Allocator;
  return 0;
}
#else

int main(){
  Model<28*28, 3> model({28*28 , 128 ,  10});
  model.Init();

  IDX3 imgs =  readImage("/home/chirag/Learn/NeuralNetwork/dataset/train-images.idx3-ubyte");
  IDX1 labels =  readImageLabels("/home/chirag/Learn/NeuralNetwork/dataset/train-labels.idx1-ubyte");

  std::vector<std::array<float , 10>> outputData;

  for(const auto label : labels.labels){
    std::array<float,10> buf = {0.0f};
    buf[label] = 1;
    outputData.push_back(buf);
  }

  Matrix input;
  input.Populate(1, 28*28, false); 

  Matrix output;
  output.Populate(1, 10, false);


  std::cout << "--- Initiating Training ---\n";
  float learningRate = 0.001f; 
  int epochs = 500;        
  const auto inputStrider = 28*28;
  const auto traincount = 80;
  const auto alpha = 1.0f/255.0f;
  const auto nin = 28*28;
  const auto nout = 10;
  const float range = std::sqrt(6/(nin + nout));
  std::cout << "--- Done ---\n--- Starting Traning ---\n";

  for (int epoch = 0; epoch < epochs; epoch++) {
    for (size_t i = 0; i < traincount; i++) {
      input.Cpy(imgs.data.data() + i*inputStrider, inputStrider);
      cblas_sscal(input.rows*input.cols,alpha ,input.data,1);
      output.Cpy(outputData[i]);

      BackProp(model, input, output, learningRate);
    }
  }

  std::cout << "Total memory usage " << __Global_Mat_Allocator->GetStrider()  << '\n';

  Infer(model , traincount);
  delete __Global_Mat_Allocator;
}

#endif
