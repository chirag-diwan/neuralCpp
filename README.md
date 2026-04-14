# NeuralCpp
C++ framework for neural networks . Making model and traning them.

----

# How it works

The whole framework can be divided into four major parts , the matrix , the maths , the Layer and Model and the memory management pool.


**MEMORY MANAGEMENT**
The whole memory management is a memory pool that is controlled by MatAllocator (A class) . There is two types of memory allocation , first the persistent ones and then the temporary ones. 
The memory pool is resuable , thanks to RAII , a struct DeferFree is responsible to clean the memory allocated in the scope that it lives
A global thread local pointer called __Global_Mat_Allocator is used for allocation .

**LAYER AND MODEL**
Layers is a struct that contains Matrices
Model is a struct that contains arrays of layers and provided information to the layer about the shape of the layer;

**MATHS**
Functions like MatMult that have O(N^3) complexity are NOT implemented and cblas is used for them
Functions like dot product , sum , addition , sigmoid , sigmoid prime etc are implamented 

**MATRIX**
Struct with basic information and pointer to an offset in data pool

**DATASET LOADING**
This implementation contains the data reader and praser implementation only for MNIST dataset .


----

# XOR Traning Example

While building the binary , if you provide option -DXOR_TEST=ON then the binary will execute a XOR train and inference test.
![XOR Test Image](./imgs/XORTEST.png)


----

# Example 
```cpp

    #include "include/MatMaths.h"
    #include "include/Model.h"
    #include "include/Dataset.h"
    #include <array>
    #include <cmath>
    #include <iostream>
    #include <openblas/cblas.h>
    
    thread_local MatAllocator* __Global_Mat_Allocator = new MatAllocator(1024*1024);
    
    template <size_t inputParamCount, size_t layerCount>
    void Infer(Model<inputParamCount, layerCount>& model , size_t testcount) {
      IDX3 testimgs = readImage("/home/chirag/Learn/NeuralNetwork/dataset/t10k-images.idx3-ubyte");
      IDX1 testlabels = readImageLabels("/home/chirag/Learn/NeuralNetwork/dataset/t10k-labels.idx1-ubyte");
    
      Matrix testinput;
      testinput.Populate(1, 28 * 28, false); 
    
      const auto inputStrider = 28 * 28;
      
      size_t correct_predictions = 0;
    
      const auto alpha = 1.0f/255.0f;
    
      for (size_t i = 0; i < testcount; i++) {
          testinput.Cpy(testimgs.data.data() + i * inputStrider, inputStrider);
          cblas_sscal(testinput.rows*testinput.cols,alpha ,testinput.data,1);
          {
            DeferFree df; 
            Forward(model, testinput);
            
            auto& final_layer_A = model.layers[model.layers.size() - 1].A;
            size_t predicted_class = ArgMax(final_layer_A);
            
            size_t target_class = testlabels.labels[i];
            
            if (predicted_class == target_class) {
                correct_predictions++;
            }
          }
      }
      
      float accuracy = (static_cast<float>(correct_predictions) / testcount) * 100.0f;
      std::cout << "Accuracy: " << accuracy << "% (" << correct_predictions << "/" << testcount << ")\n";
    }
    
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

```

----

# Testing
```bash
git clone https://github.com/chirag-diwan/neuralCpp.git
cd neuralCpp
mkdir build && cd build
cmake -DXOR_TEST=OFF ..
make 
./NeuralCpp
```
