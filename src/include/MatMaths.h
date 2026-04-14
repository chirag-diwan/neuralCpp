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

  MatAllocator(size_t poolSize){
    Pool = std::make_unique<float[]>(poolSize);
    for(uint64_t i = 0 ; i < poolSize ; i++){
      Pool[i] = 0;
    }
    capacity = poolSize;
  }

  float* allocate(uint64_t nrows , uint64_t ncols){
    if(strider + nrows*ncols > capacity){
      ERROR_AND_EXIT("Usage more than allocation");
    }
    auto* data = Pool.get() + strider;
    strider += nrows * ncols;
    return data;
  }

  float* operator()(uint64_t nrows , uint64_t ncols){
    auto* data = Pool.get() + strider;
    strider += nrows * ncols;
    return data;
  }

  uint64_t GetStrider()const{
    return strider;
  }
  
  void SetStrider(uint64_t val){
    strider =  val;
  }
};

extern thread_local MatAllocator* __Global_Mat_Allocator;

// RAII
class DeferFree{
  public:
  uint64_t saved;
  DeferFree(){
    ERRORIF(__Global_Mat_Allocator == nullptr,"Global Allocator pointer not set");
    saved = __Global_Mat_Allocator->GetStrider();
  }
  ~DeferFree(){
    ERRORIF(__Global_Mat_Allocator == nullptr,"Global Allocator pointer not set");
    __Global_Mat_Allocator->SetStrider(saved);
  }
};

struct Matrix {
  float* data;
  uint32_t rows;
  uint32_t cols;

  Matrix() = default;
  Matrix(Matrix&&) = default;

  Matrix(Matrix&) = delete;


  void Populate(uint32_t r, uint32_t c, bool rand) ;
 
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


  float& operator[](size_t i) const ;
};



void PrintMat(const std::string& name, const Matrix& m) ;
inline float* TempAlloc(uint64_t rows , uint64_t cols){
  ERRORIF(__Global_Mat_Allocator == nullptr,"Global Allocator pointer not set");
  return __Global_Mat_Allocator->allocate(rows, cols);
}
inline void MatMul( Matrix& A,  Matrix& B, Matrix& C , CBLAS_TRANSPOSE TransA , CBLAS_TRANSPOSE TransB) {
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

inline void MatAddInplace(Matrix& A, Matrix& B) {
  assert(A.cols == B.cols);
  assert(A.rows == B.rows);
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    B[i] = A[i] + B[i];
  }
}

inline void UpdateParameter(Matrix& param, const Matrix& gradient, float learningRate) {
  assert(param.rows == gradient.rows && param.cols == gradient.cols);
  for (size_t i = 0; i < param.rows * param.cols; i++) {
    param[i] = param[i] - (learningRate * gradient[i]);
  }
}



float Sigmoid(float a) ;
float SigmoidPrime(float a) ;
void Sigmoid(Matrix& A, Matrix& B) ;
void SigmoidPrime(Matrix& A, Matrix& B) ;
void MatSub(Matrix& A, Matrix& B, Matrix& C) ;
size_t ArgMax(const Matrix& m) ;


inline float ReduceSqr(Matrix& A) {
  float result = 0;
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    result += A[i] * A[i];
  }
  return result;
}

inline float Reduce(Matrix& A) {
  float result = 0;
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    result += A[i];
  }
  return result;
}
void HarmardProduct(Matrix& A, Matrix& B, Matrix& out) ;

