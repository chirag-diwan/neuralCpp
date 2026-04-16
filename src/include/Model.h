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

struct ActivationProfile{
  ActivationFn Activation;
  ActivationPrimeFn ActivationPrime;
};

struct Layer {
  Mat A;
  Mat W;
  Mat B;
  Mat Z;
  Mat delta; 
  Mat zSigmaPrime;

  Layer() = default;
  Layer(const Layer& layer) = default;
    

  void Populate(size_t currNeurons, size_t prevNeurons , size_t batchSize) ;
};

struct NeuralNetwork {
  std::vector<Layer> layers;
  std::vector<size_t> layerSizes;
 
  ActivationProfile ActivationFuncs;

  uint32_t batchsize;

  NeuralNetwork() = delete;

  NeuralNetwork(const std::vector<size_t>&& ls) : layerSizes(ls) {}

  void Init(size_t inputParamCount , size_t batchSize , ActivationProfile Profile) ;

  ~NeuralNetwork(){}
};



void Forward(NeuralNetwork& model, Mat& input) ;
float Cost(NeuralNetwork& model, Mat& input, Mat& output) ;
void CostGradFinalLayer(Mat& Activations, Mat& Outputs, Mat& out) ;
void ErrorFinalLayer(Layer& last, Mat& Outputs ) ;
void ErrorLayer(Layer& currLayer, Layer& nextLayer ) ;
void BackProp(NeuralNetwork& model, Mat& input, Mat& output, float learningRate , size_t BATCH_SIZE) ;
void PrintModel(NeuralNetwork& model , bool showSize);
