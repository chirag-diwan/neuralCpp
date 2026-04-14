#include "../include/MatMaths.h"


void Matrix::Populate(uint32_t r, uint32_t c, bool rand) {
  this->rows = r;
  this->cols = c;
  this->data = __Global_Mat_Allocator->allocate(r,c);

  if (rand) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1,1);
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = dist(gen);
    }
  } 
}

float& Matrix::operator[](size_t i) const {
  assert(i >= 0 && i < rows * cols);
  return data[i];
}



void PrintMat(const std::string& name, const Matrix& m) {
  std::cout << "Matrix " << name 
    << " (" << m.rows << " x " << m.cols << ")\n";

  for (uint32_t r = 0; r < m.rows; ++r) {
    std::cout << "  [";
    for (uint32_t c = 0; c < m.cols; ++c) {
      std::cout << std::setw(8) << std::setprecision(4) << std::fixed
        << m.data[r * m.cols + c] << " ";
      if (c != m.cols - 1) {
        std::cout << ',';
      }
    }
    std::cout << "]\n";
  }
}




float Sigmoid(float a) {
  return 1.0f / (1.0f + std::exp(-a));
}

float SigmoidPrime(float a) {
  float sig = Sigmoid(a);
  return sig * (1.0f - sig); // Mathematically more stable evaluation
}

void Sigmoid(Matrix& A, Matrix& B) {
  assert(A.rows == B.rows && A.cols == B.cols);
  for (size_t i = 0; i < A.rows * A.cols; i++) {
    B[i] = Sigmoid(A[i]);
  }
}

void SigmoidPrime(Matrix& A, Matrix& B) {
  assert(A.rows == B.rows && A.cols == B.cols);
  for (size_t i = 0; i < A.rows * A.cols; i++) {
    B[i] = SigmoidPrime(A[i]);
  }
}

void MatSub(Matrix& A, Matrix& B, Matrix& C) {

  assert(A.cols == B.cols && A.rows == B.rows);
  assert(A.cols == C.cols && A.rows == C.rows);
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    C[i] = A[i] - B[i];
  }
}

size_t ArgMax(const Matrix& m) {
  // Ensure we are scanning a 1D vector representation
  assert(m.rows == 1 || m.cols == 1); 

  size_t max_index = 0;
  float max_val = m[0];
  size_t total_elements = m.rows * m.cols;

  for (size_t i = 1; i < total_elements; i++) {
    if (m[i] > max_val) {
      max_val = m[i];
      max_index = i;
    }
  }
  return max_index;
}



void HarmardProduct(Matrix& A, Matrix& B, Matrix& out) {
  assert(A.rows == B.rows && A.cols == B.cols);
  assert(A.rows == out.rows && A.cols == out.cols);

  for (size_t i = 0; i < A.rows * A.cols; i++) {
    out[i] = A[i] * B[i];
  }
}

