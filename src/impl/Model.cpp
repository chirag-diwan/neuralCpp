#include "../include/Model.h"


void Layer::Populate(size_t currNeurons, size_t prevNeurons , size_t batchSize) {
  A.Populate(batchSize, currNeurons, false );
  W.Populate(prevNeurons, currNeurons, true );
  B.Populate(1, currNeurons, true );
  Z.Populate(batchSize, currNeurons, false );
  delta.Populate(batchSize, currNeurons, false );
  zSigmaPrime.Populate(batchSize, currNeurons, false );
}

void NeuralNetwork::Init(size_t inputParamCount , size_t batchSize ,ActivationProfile Profile) {
  ActivationFuncs = Profile;
  batchsize = batchSize;
  layers.resize(layerSizes.size());
  for (size_t i = 0; i < layerSizes.size(); i++) {
    if (i == 0) {
      layers[i].Populate(layerSizes[i], inputParamCount  , batchSize);
    }else{
      layers[i].Populate(layerSizes[i], layerSizes[i - 1] , batchSize);
    }
  }
}



void Forward(NeuralNetwork& model, Mat& input) {
  MatMul(input, model.layers[0].W, model.layers[0].Z , CblasNoTrans , CblasNoTrans);
  MatAddBias(model.layers[0].B, model.layers[0].Z);
  model.ActivationFuncs.Activation(model.layers[0].Z, model.layers[0].A);

  for (size_t i = 1; i < model.layers.size(); i++) {
    auto& prev = model.layers[i - 1];
    auto& layer = model.layers[i];
    MatMul(prev.A, layer.W, layer.Z , CblasNoTrans , CblasNoTrans);
    MatAddBias(layer.B, layer.Z);
    model.ActivationFuncs.Activation(layer.Z, layer.A);
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



void ErrorFinalLayer(Layer& last, Mat& Outputs , ActivationPrimeFn ActivationPrime) {
  {
    DeferFree df;
    ActivationPrime(last.Z, last.zSigmaPrime);
    Mat costGrad;
    costGrad.Populate(last.A.rows, last.A.cols, false);
    CostGradFinalLayer(last.A, Outputs, costGrad);
    HarmardProduct(costGrad, last.zSigmaPrime, last.delta);
  }
}

void ErrorLayer(Layer& currLayer, Layer& nextLayer , ActivationPrimeFn ActivationPrime) {
  {
    DeferFree df;

    Mat buf;
    buf.Populate(nextLayer.delta.rows, nextLayer.W.rows, false );

    // buf = delta^(l+1) * W^(l+1)^T
    MatMul(nextLayer.delta, nextLayer.W, buf , CblasNoTrans , CblasTrans);

    ActivationPrime(currLayer.Z, currLayer.zSigmaPrime);
    HarmardProduct(buf, currLayer.zSigmaPrime, currLayer.delta);
  }
}

void BackProp(NeuralNetwork& model, Mat& input, Mat& output, float learningRate , size_t BATCH_SIZE) {
  for (int i = static_cast<int>(model.layers.size()) - 1; i >= 0; i--) {
    if (i == model.layers.size() - 1) {
      ErrorFinalLayer(model.layers[i], output, model.ActivationFuncs.ActivationPrime);
    } else {
      ErrorLayer(model.layers[i], model.layers[i + 1] ,  model.ActivationFuncs.ActivationPrime);
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
      UpdateParameter(curr.W, dw, learningRate , BATCH_SIZE);
      UpdateParameterBias(curr.B, curr.delta, learningRate , BATCH_SIZE);
    }
  }
}


void PrintModel(NeuralNetwork& model , bool showSize){
  for(const auto& layer : model.layers){
    PrintMat("A" , layer.A,showSize);
    PrintMat("W" , layer.W,showSize);
    PrintMat("B" , layer.B,showSize);
    PrintMat("Z" , layer.Z,showSize);
    PrintMat("delta" , layer.delta , showSize); 
    PrintMat("zSigmaPrime" , layer.zSigmaPrime , showSize);
  }
}
