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
  Mat A;
  Mat W;
  Mat B;
  Mat Z;
  Mat delta; 
  Mat zSigmaPrime;

  Layer() = default;
  Layer(const Layer& layer) = default;
    

  void Populate(size_t currNeurons, size_t prevNeurons) {
    A.Populate(1, currNeurons, false );
    W.Populate(prevNeurons, currNeurons, true );
    B.Populate(1, currNeurons, true );
    Z.Populate(1, currNeurons, false );
    delta.Populate(1, currNeurons, false );
    zSigmaPrime.Populate(1, currNeurons, false );
  }
};

struct NeuralNetwork {
  std::vector<Layer> layers;
  std::vector<size_t> layerSizes;

  NeuralNetwork() = delete;

  NeuralNetwork(const std::vector<size_t>&& ls) : layerSizes(ls) {}

  void Init(size_t inputParamCount) {
    layers.resize(layerSizes.size());
    for (size_t i = 0; i < layerSizes.size(); i++) {
      if (i == 0) {
        layers[i].Populate(layerSizes[i], inputParamCount );
      }else{
        layers[i].Populate(layerSizes[i], layerSizes[i - 1] );
      }
    }
  }

  ~NeuralNetwork(){}
};



void Forward(NeuralNetwork& model, Mat& input) {
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

float Cost(NeuralNetwork& model, Mat& input, Mat& output) {
  {
    DeferFree df;
    Forward(model, input);
    Mat C;
    auto& A = model.layers[model.layers.size() - 1].A;
    C.Populate(A.rows, A.cols, false);
    MatSub(A, output, C);
    return ReduceSqr(C);
  }
}

void CostGradFinalLayer(Mat& Activations, Mat& Outputs, Mat& out) {
  assert(Activations.rows == Outputs.rows && Activations.cols == Outputs.cols);
  assert(Activations.rows == out.rows && Activations.cols == out.cols);

  for (size_t i = 0; i < Activations.rows * Activations.cols; i++) {
    out[i] = 2.0f * (Activations[i] - Outputs[i]);
  }
}


void ErrorFinalLayer(Layer& last, Mat& Outputs ) {
  {
    DeferFree df;
    SigmoidPrime(last.Z, last.zSigmaPrime);
    Mat costGrad;
    costGrad.Populate(last.A.rows, last.A.cols, false);
    CostGradFinalLayer(last.A, Outputs, costGrad);
    HarmardProduct(costGrad, last.zSigmaPrime, last.delta);
  }
}

void ErrorLayer(Layer& currLayer, Layer& nextLayer ) {
  {
    DeferFree df;

    Mat buf;
    buf.Populate(nextLayer.delta.rows, nextLayer.W.rows, false );

    // buf = delta^(l+1) * W^(l+1)^T
    MatMul(nextLayer.delta, nextLayer.W, buf , CblasNoTrans , CblasTrans);

    SigmoidPrime(currLayer.Z, currLayer.zSigmaPrime);
    HarmardProduct(buf, currLayer.zSigmaPrime, currLayer.delta);
  }
}

void BackProp(NeuralNetwork& model, Mat& input, Mat& output, float learningRate ) {
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


      Mat dw;
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




