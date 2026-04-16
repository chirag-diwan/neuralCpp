#pragma  once

#include "../include/Model.h"
#include "MatMaths.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ostream>

void writeMatrix(std::ofstream& out, const Mat& mat) ;
void readMatrix(std::ifstream& in, Mat& mat) ;
void SaveModel(const NeuralNetwork& model, uint32_t inputParamCount, const std::string& filepath) ;
NeuralNetwork NNLoadModel(const std::string& filepath , ActivationProfile& Profile) ;
