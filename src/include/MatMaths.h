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

  Mat(Mat&) = delete;

  void ViewNoAlloc(uint32_t r, uint32_t c , float* data);

  void Populate(uint32_t r, uint32_t c, bool rand , float rangeMax = 1) ;

  float& operator[](size_t i) const ;


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

inline void UpdateParameter(Mat& param, const Mat& gradient, float learningRate) {
  assert(param.rows == gradient.rows && param.cols == gradient.cols);
  for (size_t i = 0; i < param.rows * param.cols; i++) {
    param[i] = param[i] - (learningRate * gradient[i]);
  }
}



inline float ReduceSqr(Mat& A) {
  float result = 0;
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    result += A[i] * A[i];
  }
  return result;
}

inline float Reduce(Mat& A) {
  float result = 0;
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    result += A[i];
  }
  return result;
}

void PrintMat(const std::string& name, const Mat& m) ;




float Sigmoid(float a) ;

float SigmoidPrime(float a) ;

void Sigmoid(Mat& A, Mat& B) ;

void SigmoidPrime(Mat& A, Mat& B) ;

void MatSub(Mat& A, Mat& B, Mat& C) ;

size_t ArgMax(const Mat& m) ;

inline void MatScale(Mat& A , float scale , uint64_t strider = 1){
  cblas_sscal(A.rows*A.cols,scale ,A.data,strider);
}

void HarmardProduct(Mat& A, Mat& B, Mat& out) ;
