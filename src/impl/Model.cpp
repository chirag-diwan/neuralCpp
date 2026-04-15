#include "../include/Model.h"


void Layer::Populate(size_t currNeurons, size_t prevNeurons) {
  A.Populate(1, currNeurons, false );
  W.Populate(prevNeurons, currNeurons, true );
  B.Populate(1, currNeurons, true );
  Z.Populate(1, currNeurons, false );
  delta.Populate(1, currNeurons, false );
  zSigmaPrime.Populate(1, currNeurons, false );
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
