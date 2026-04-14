#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <openblas/cblas.h>
#include <random>
#include <vector>
#include "./MatMaths.h"

struct Layer {
  Matrix A;
  Matrix W;
  Matrix B;
  Matrix Z;
  Matrix delta; 
  Matrix zSigmaPrime;

  Layer() = default;

  void Populate(size_t currNeurons, size_t prevNeurons) ;
};

template <size_t inputParamCount, size_t layerCount>
struct Model {
  std::array<Layer, layerCount> layers;
  std::array<size_t, layerCount> layerSizes;

  Model();

  Model(std::array<size_t, layerCount>&& ls) : layerSizes(ls) {}

  void Init() {
    for (size_t i = 0; i < layerSizes.size(); i++) {
      if (i == 0) {
        layers.at(i).Populate(layerSizes[i], inputParamCount );
        continue;
      }
      layers.at(i).Populate(layerSizes[i], layerSizes[i - 1] );
    }
  }

  ~Model(){}
};



// Pure transpose logic separated from multiplication





template <size_t inputParamCount, size_t layerCount>
void Forward(Model<inputParamCount, layerCount>& model, Matrix& input) {
  MatMul(input, model.layers[0].W, model.layers[0].Z , CblasNoTrans , CblasNoTrans);
  MatAddInplace(model.layers[0].B, model.layers[0].Z);
  Sigmoid(model.layers[0].Z, model.layers[0].A);

  for (size_t i = 1; i < model.layers.size(); i++) {
    auto& prev = model.layers[i - 1];
    auto& layer = model.layers[i];
    MatMul(prev.A, layer.W, layer.Z , CblasNoTrans , CblasNoTrans);
    MatAddInplace(layer.B, layer.Z);
    Sigmoid(layer.Z, layer.A);
  }
}

template <size_t inputParamCount, size_t layerCount>
float Cost(Model<inputParamCount, layerCount>& model, Matrix& input, Matrix& output) {
  {
    DeferFree df;
    Forward(model, input);
    Matrix C;
    auto& A = model.layers[model.layers.size() - 1].A;
    C.Populate(A.rows, A.cols, false);
    MatSub(A, output, C);
    return ReduceSqr(C);
  }
}

void CostGradFinalLayer(Matrix& Activations, Matrix& Outputs, Matrix& out) ;
void ErrorFinalLayer(Layer& last, Matrix& Outputs ) ;
void ErrorLayer(Layer& currLayer, Layer& nextLayer ) ;


template <size_t inputParamCount, size_t layerCount>
void BackProp(Model<inputParamCount, layerCount>& model, Matrix& input, Matrix& output, float learningRate ) {
  for (int i = static_cast<int>(model.layers.size()) - 1; i >= 0; i--) {
    if (i == model.layers.size() - 1) {
      ErrorFinalLayer(model.layers[i], output);
    } else {
      ErrorLayer(model.layers[i], model.layers[i + 1]);
    }
  }

  for (size_t i = 0; i < model.layers.size(); i++) {
    {
      DeferFree df;

      auto& curr = model.layers[i];


      Matrix dw;
      if (i == 0) {
        dw.Populate(input.cols, curr.delta.cols, false );
        MatMul(input, curr.delta, dw , CblasTrans , CblasNoTrans);
      } else {
        auto& prev = model.layers[i - 1];
        dw.Populate(prev.A.cols, curr.delta.cols, false );
        MatMul(prev.A, curr.delta, dw , CblasTrans , CblasNoTrans);
      }
      UpdateParameter(curr.W, dw, learningRate);
      UpdateParameter(curr.B, curr.delta, learningRate);
    }
  }
}


