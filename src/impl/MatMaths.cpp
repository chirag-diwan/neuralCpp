#include "../include/MatMaths.h"


MatAllocator::MatAllocator(size_t poolSize){
  Pool = std::make_unique<float[]>(poolSize);
  for(uint64_t i = 0 ; i < poolSize ; i++){
    Pool[i] = 0;
  }
  capacity = poolSize;
}

float*MatAllocator:: allocate(uint64_t nrows , uint64_t ncols){
  if(strider + nrows*ncols > capacity){
    ERROR_AND_EXIT("Usage more than allocation");
  }
  auto* data = Pool.get() + strider;
  strider += nrows * ncols;
  return data;
}

float* MatAllocator::operator()(uint64_t nrows , uint64_t ncols){
  auto* data = Pool.get() + strider;
  strider += nrows * ncols;
  return data;
}

uint64_t MatAllocator::GetStrider()const{
  return strider;
}

void MatAllocator::SetStrider(uint64_t val){
  strider =  val;
}


extern thread_local MatAllocator* __Global_Mat_Allocator;

// RAII
DeferFree::DeferFree(){
  ERRORIF(__Global_Mat_Allocator == nullptr,"Global Allocator pointer not set");
  saved = __Global_Mat_Allocator->GetStrider();
}
DeferFree::~DeferFree(){
  ERRORIF(__Global_Mat_Allocator == nullptr,"Global Allocator pointer not set");
  __Global_Mat_Allocator->SetStrider(saved);
}

void Mat::ViewNoAlloc(uint32_t r, uint32_t c , float* data){
  this->rows = r;
  this->cols = c;
  this->data = data;
}

void Mat::Populate(uint32_t r, uint32_t c, bool rand , float rangeMax) {
  this->rows = r;
  this->cols = c;
  this->data = __Global_Mat_Allocator->allocate(r,c);

  if (rand) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-rangeMax,rangeMax);
    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = dist(gen);
    }
  } 
}

float& Mat::operator[](size_t i) const {
  assert(i >= 0 && i < rows * cols);
  return data[i];
}





void PrintMat(const std::string& name, const Mat& m , bool onlySize) {
  std::cout << "Mat " << name 
    << " (" << m.rows << " x " << m.cols << ")\n";

  if(!onlySize){
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
}

void Sigmoid(Mat& A, Mat& B) {
  assert(A.rows == B.rows && A.cols == B.cols);
  for (size_t i = 0; i < A.rows * A.cols; i++) {
    B[i] = Sigmoid(A[i]);
  }
}

void SigmoidPrime(Mat& A, Mat& B) {
  assert(A.rows == B.rows && A.cols == B.cols);
  for (size_t i = 0; i < A.rows * A.cols; i++) {
    B[i] = SigmoidPrime(A[i]);
  }
}

void MatSub(Mat& A, Mat& B, Mat& C) {

  assert(A.cols == B.cols && A.rows == B.rows);
  assert(A.cols == C.cols && A.rows == C.rows);
  for (size_t i = 0; i < A.cols * A.rows; i++) {
    C[i] = A[i] - B[i];
  }
}

size_t ArgMax(const Mat& m) {
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

void HarmardProduct(Mat& A, Mat& B, Mat& out) {
  assert(A.rows == B.rows && A.cols == B.cols);
  assert(A.rows == out.rows && A.cols == out.cols);

  for (size_t i = 0; i < A.rows * A.cols; i++) {
    out[i] = A[i] * B[i];
  }
}
