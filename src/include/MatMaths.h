#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>
#include <openblas/cblas.h>
#include "../Utils/Utils.h"

class MatAllocator{
  std::unique_ptr<float[]> Pool;
  uint64_t strider = 0;
  size_t capacity;

  public:

  MatAllocator(size_t poolSize);
  float* allocate(uint64_t nrows , uint64_t ncols);
  float* operator()(uint64_t nrows , uint64_t ncols);
  uint64_t GetStrider()const;
  void SetStrider(uint64_t val);
};

extern thread_local MatAllocator* __Global_Mat_Allocator;

// RAII
class DeferFree{
  public:
    uint64_t saved;
    DeferFree();
    ~DeferFree();
};

struct Mat {
  float* data;
  uint32_t rows;
  uint32_t cols;

  Mat() = default;
  Mat(Mat&&) = default;
  Mat(Mat&) = default;
  Mat(const Mat&) = default;

  void ViewNoAlloc(uint32_t r, uint32_t c , float* data);

  void Populate(uint32_t r, uint32_t c, bool rand , float rangeMax = 1) ;

  float& operator[](size_t i) const ;

  void operator=(Mat&& mat){
    this->data = std::move(mat.data);
    this->rows = mat.rows;
    this->cols = mat.cols;
  }

  template<typename T>
    void operator|=(const std::vector<T>& vals) {
      for (size_t i = 0; i < rows * cols && i < vals.size(); i++) {
        data[i] = static_cast<float>(vals[i]);
      }
    }


  template<typename T>
    void Cpy(T* vals , size_t NumElements){
      for(size_t i = 0 ; i < NumElements ; i++){
        data[i] = static_cast<float>(vals[i]);
      }
    }

  template<typename T , size_t s>
    void Cpy(std::array<T,s> vals){
      for(size_t i = 0 ; i < s ; i++){
        data[i] = static_cast<float>(vals[i]);
      }
    }
};


inline float* TempAlloc(uint64_t rows , uint64_t cols){
  ERRORIF(__Global_Mat_Allocator == nullptr,"Global Allocator pointer not set");
  return __Global_Mat_Allocator->allocate(rows, cols);
}



void PrintMat(const std::string& name, const Mat& m , bool onlySize) ;

inline void MatMul( Mat& A,  Mat& B, Mat& C , CBLAS_TRANSPOSE TransA , CBLAS_TRANSPOSE TransB) {
  size_t M = (TransA == CblasNoTrans) ? A.rows : A.cols;
  size_t K = (TransA == CblasNoTrans) ? A.cols : A.rows;

  size_t K_B = (TransB == CblasNoTrans) ? B.rows : B.cols;
  size_t N   = (TransB == CblasNoTrans) ? B.cols : B.rows;

  assert(K == K_B && "Inner dimensions must match for multiplication");
  assert(C.rows == M && C.cols == N && "Output matrix incorrectly sized");
  size_t lda = A.cols; 
  size_t ldb = B.cols;
  size_t ldc = C.cols;

  cblas_sgemm(CblasRowMajor, TransA, TransB, 
      M, N, K, 
      1.0f, 
      A.data, lda, 
      B.data, ldb, 
      0.0f, 
      C.data, ldc);
}

inline void MatAddInplace(Mat& A, Mat& B) {
  assert(A.cols == B.cols);
  assert(A.rows == B.rows);
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    B[i] = A[i] + B[i];
  }
}


inline void MatAddBias(const Mat& B , Mat& Z) {
  assert(Z.cols == B.cols && B.rows == 1);
  
  for (size_t row = 0; row < Z.rows; row++) {
    for (size_t col = 0; col < Z.cols; col++) {
      Z.data[row * Z.cols + col] += B.data[col];
    }
  }
}


inline void UpdateParameter(Mat& param, const Mat& gradient, float learningRate, size_t BATCH_SIZE) {
  assert(param.rows == gradient.rows && param.cols == gradient.cols);
  float effective_rate = -(learningRate / static_cast<float>(BATCH_SIZE));
  cblas_saxpy(param.rows * param.cols, effective_rate, gradient.data, 1, param.data, 1);
}

inline float ReduceSqr(Mat& A) {
  float result = 0;
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    result += A[i] * A[i];
  }
  return result;
}

inline void UpdateParameterBias(Mat &bias, const Mat &gradient, float learningRate, size_t BATCH_SIZE){
  assert(bias.rows == 1 && bias.cols == gradient.cols && gradient.rows == BATCH_SIZE);
  {
    DeferFree df;
    Mat buf;
    buf.Populate(bias.rows, bias.cols,false);

    for(size_t i = 0; i < buf.rows * buf.cols; i++) {
      buf.data[i] = 0.0f;
    }

    for(size_t r = 0; r < BATCH_SIZE; r++) {
      for(size_t c = 0; c < gradient.cols; c++) {
        buf.data[c] += gradient.data[r * gradient.cols + c];
      }
    }
    UpdateParameter(bias, buf,learningRate, BATCH_SIZE);
  }
}


inline float Reduce(Mat& A) {
  float result = 0;
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    result += A[i];
  }
  return result;
}

using ActivationFn =void(*)(Mat& src , Mat& dst); //forward declare activation
using ActivationPrimeFn =void(*)(Mat& mat , Mat& dst); //forward declare activation prime

inline float leakyReLU(float a){
  return a > 0 ? a : 0.01*a;
}

inline float leakyReLUPrime(float a){
  return a > 0 ? 1 : 0.01;
}


inline float ReLU(float a){
  return a > 0 ? a : 0;
}

inline float ReLUPrime(float a){
  return a > 0 ? 1 : 0;
}

inline float Sigmoid(float a) {
  return 1.0f / (1.0f + std::exp(-a));
}

inline float SigmoidPrime(float a) {
  float sig = Sigmoid(a);
  return sig * (1.0f - sig);

}

void Sigmoid(Mat& A, Mat& B) ;
void SigmoidPrime(Mat& A, Mat& B) ;
void MatSub(Mat& A, Mat& B, Mat& C) ;
size_t ArgMax(const Mat& m) ;
void HarmardProduct(Mat& A, Mat& B, Mat& out);
inline void MatScale(Mat& A , float scale , uint64_t strider = 1){
  cblas_sscal(A.rows*A.cols,scale ,A.data,strider);
}
